"""
View-specific Neuron Fine-tuning (VNF) -- View-PNDF
===================================================

Fine-tunes ONLY the detected view-specific pattern neurons of the medical VLM
``Mini-InternVL2-4B-DA-Medical`` (Phi-3 language decoder), leaving all other
parameters frozen.  This improves view-specific (esp. lateral) report
generation while preserving general-view performance and keeping the trainable
parameter budget tiny (paper: <5% of neurons).

Mechanism
---------
* Only the weight tensors that contain pattern neurons are set trainable
  (``qkv_proj`` / ``o_proj`` / ``gate_up_proj`` / ``down_proj`` of the relevant
  layers); everything else is frozen.
* A backward hook multiplies each such tensor's gradient by a boolean mask that
  is True only on the detected neuron rows/columns, so optimisation touches
  exactly the pattern neurons (paper VNF; see ``view_pndf.build_grad_mask``).

Example
-------
    python train.py \
        --neuron_file ./neuron_deactivation/..._lateral_neuron_0.05.json \
        --view lateral \
        --output_dir ./ckpt/lateral_vnf \
        --num_epochs 1 --learning_rate 4e-6 --micro_batch_size 8
"""

import os
import sys
import argparse

# The environment ships a broken bitsandbytes (CUDA setup failed).  We do not
# use 8-bit optimisers, so block the import before transformers.Trainer pulls
# it in, otherwise `from transformers import Trainer` crashes.
sys.modules["bitsandbytes"] = None

import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments

from view_pndf import (
    build_projection_spec,
    build_grad_mask,
    register_grad_mask_hooks,
    read_neuron,
)
from data_rrg import (
    load_dataset,
    filter_by_view,
    load_image,
    build_training_example,
    DEFAULT_PROMPT,
)
import config

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class RRGViewDataset(Dataset):
    """Image+report examples of a single view, tokenised for causal LM loss on
    the report tokens only."""

    def __init__(self, samples, model, tokenizer, cutoff_len=1024, dynamic_tiles=False):
        self.samples = samples
        self.model = model
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.dynamic_tiles = dynamic_tiles

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        pixel_values = load_image(s["image_path"], dynamic=self.dynamic_tiles).to(torch.bfloat16)
        ex = build_training_example(
            self.model, self.tokenizer, pixel_values.shape[0],
            DEFAULT_PROMPT, s["report"], cutoff_len=self.cutoff_len,
        )
        ex["pixel_values"] = pixel_values
        return ex


class RRGCollator:
    """Right-pad input_ids/labels/attention_mask (InternVL training convention);
    stack pixel_values.  Also records ``image_flags`` (all-ones) required by
    InternVLChatModel's native forward."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
        num_tiles = sum(b["pixel_values"].shape[0] for b in batch)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_flags": torch.ones(num_tiles, 1, dtype=torch.long),
        }


def freeze_except_pattern_tensors(model, masks):
    """Freeze everything, then unfreeze only the weight tensors that carry
    pattern neurons (their grads are further masked by backward hooks)."""
    trainable_names = set(masks.keys())
    for name, p in model.named_parameters():
        p.requires_grad_(name in trainable_names)
    n_train = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    print(f"trainable tensors: {len(trainable_names)}, trainable params: {n_train:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--neuron_file", type=str, required=True)
    parser.add_argument("--view", type=str, required=True, help="frontal or lateral")
    parser.add_argument("--dataset", type=str, default="iu_xray", choices=list(config.DATASETS))
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="override config placeholder for the dataset annotation JSON")
    parser.add_argument("--image_root", type=str, default=None,
                        help="override config placeholder for the dataset image root")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-6)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.5)
    parser.add_argument("--dynamic_tiles", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=-1)
    args = parser.parse_args()

    # DDP-aware device placement: each torchrun process owns one GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    spec = build_projection_spec(model)
    detected = read_neuron(args.neuron_file)

    # Build gradient masks + freeze non-pattern tensors, then register hooks.
    masks = build_grad_mask(model, detected, spec, device=device)
    freeze_except_pattern_tensors(model, masks)
    register_grad_mask_hooks(model, masks)
    model.config.use_cache = False
    model.language_model.config.use_cache = False
    # NOTE: gradient checkpointing is intentionally NOT enabled.  The vision
    # tower and token embeddings are frozen, so a checkpointed LM segment whose
    # inputs (inputs_embeds) have requires_grad=False breaks the autograd graph
    # ("element 0 does not require grad").  The trainable projections live
    # inside the LM and receive gradients fine without checkpointing.

    annotation_path, image_root = config.get_dataset_paths(
        args.dataset, args.annotation_path, args.image_root
    )
    splits = load_dataset(args.dataset, annotation_path, image_root)
    train_samples = filter_by_view(splits["train"], args.view)
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    print(f"training on {len(train_samples)} {args.view} samples")

    train_ds = RRGViewDataset(train_samples, model, tokenizer,
                              cutoff_len=args.cutoff_len, dynamic_tiles=args.dynamic_tiles)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_only_model=True,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False if ddp else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=RRGCollator(tokenizer),
    )
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
