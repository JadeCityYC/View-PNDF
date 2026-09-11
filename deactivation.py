"""
View-specific Neuron Verification (VNV) -- View-PNDF
====================================================

Verifies that the neurons detected by VND are genuinely responsible for
view-specific report generation.  Following the paper we:

  1. Zero (deactivate) the detected view-specific neurons and observe the
     change in generated reports (paper Eq. 9).
  2. Build a RANDOM control set of the same per-layer size and zero it instead;
     if the pattern-neuron deactivation degrades generation substantially more
     than the random deactivation, the detected neurons are validated.

Works on ``Mini-InternVL2-4B-DA-Medical`` (Phi-3 language decoder) using the
fused-projection-aware zeroing in ``view_pndf.build_deactivate_indices_dict``.

Example
-------
    # generate a random control set matching the frontal neuron file
    python deactivation.py make-random \
        --neuron_file ./neuron_deactivation/..._frontal_neuron_0.05.json \
        --out ./neuron_deactivation/..._frontal_random_0.05.json

    # qualitatively compare original vs pattern-deactivated vs random-deactivated
    python deactivation.py verify \
        --view lateral \
        --neuron_file ./neuron_deactivation/..._lateral_neuron_0.05.json \
        --num_samples 5
"""

import os
import json
import math
import random
import argparse

import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig

from view_pndf import (
    ProjectionSpec,
    build_projection_spec,
    build_deactivate_indices_dict,
    apply_zero_mask_to_model,
    read_neuron,
    merge_image_embeds,
    GROUPS,
)
from data_rrg import (
    load_dataset,
    filter_by_view,
    load_image,
    build_image_prompt,
    DEFAULT_PROMPT,
)
import config

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def load_model_and_tokenizer(model_path):
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Random control set: same #neurons per layer per group, drawn uniformly.
# --------------------------------------------------------------------------- #
def generate_random_control(detected_neuron, spec: ProjectionSpec, seed=42):
    """Return a random neuron set matching the per-layer group sizes of
    ``detected_neuron``.  Widths: attn groups use q_dim/kv_dim, ffn groups use
    ``intermediate``."""
    rng = random.Random(seed)
    width = {
        "attn_q": spec.q_dim,
        "attn_k": spec.kv_dim,
        "attn_v": spec.kv_dim,
        "attn_o": spec.q_dim,          # o_proj input dim = n_heads*head_dim
        "fwd_up": spec.intermediate,
        "fwd_down": spec.intermediate,
    }
    out = {g: {} for g in GROUPS}
    for group in GROUPS:
        for layer, idxs in detected_neuron[group].items():
            n = len(idxs)
            out[group][str(layer)] = rng.sample(range(width[group]), min(n, width[group]))
    return out


def save_neuron_lists(neurons, path):
    formatted = {g: {str(k): sorted(int(x) for x in v) for k, v in neurons[g].items()} for g in GROUPS}
    with open(path, "w") as f:
        json.dump(formatted, f)
    print(f"saved {path}")


# --------------------------------------------------------------------------- #
# Generation helper (qualitative observation; GPT-4o scoring is separate)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_report(model, tokenizer, image_path, prompt=DEFAULT_PROMPT,
                    max_new_tokens=200, dynamic_tiles=False):
    pixel_values = load_image(image_path, dynamic=dynamic_tiles).to(torch.bfloat16).cuda()
    input_ids, attention_mask = build_image_prompt(model, tokenizer, pixel_values.shape[0], question=prompt)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    gen_cfg = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False,
                               pad_token_id=tokenizer.eos_token_id, bos_token_id=bos_id,
                               eos_token_id=tokenizer.eos_token_id)
    out = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        generation_config=gen_cfg,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


def deactivate_model(model, neuron, spec):
    """Zero the given neurons on the model in place."""
    dd = build_deactivate_indices_dict(neuron, spec)
    apply_zero_mask_to_model(model, dd)
    n = sum(len(v["indices"]) for v in dd.values())
    print(f"  deactivated {n} neuron slices across {len(dd)} weight tensors")


# --------------------------------------------------------------------------- #
# Sub-commands
# --------------------------------------------------------------------------- #
def cmd_make_random(args):
    model, _ = load_model_and_tokenizer(args.model_path)
    spec = build_projection_spec(model)
    detected = read_neuron(args.neuron_file)
    ctrl = generate_random_control(detected, spec, seed=args.seed)
    save_neuron_lists(ctrl, args.out)


def cmd_verify(args):
    """Qualitative VNV: print original vs pattern-deactivated vs random-deactivated
    reports for a few view-specific samples.  Reload the model between conditions
    so zeroing does not accumulate."""
    annotation_path, image_root = config.get_dataset_paths(
        args.dataset, args.annotation_path, args.image_root
    )
    splits = load_dataset(args.dataset, annotation_path, image_root, seed=args.seed)
    samples = filter_by_view(splits[args.split], args.view)
    random.seed(args.seed)
    samples = random.sample(samples, min(args.num_samples, len(samples)))

    detected = read_neuron(args.neuron_file)

    # Condition 1: original
    print("\n########## ORIGINAL ##########")
    model, tok = load_model_and_tokenizer(args.model_path)
    spec = build_projection_spec(model)
    originals = [generate_report(model, tok, s["image_path"]) for s in samples]
    for s, r in zip(samples, originals):
        print(f"[{s['id']}] {r}\n")
    del model; torch.cuda.empty_cache()

    # Condition 2: pattern-neuron deactivation
    print("\n########## PATTERN-NEURON DEACTIVATED ##########")
    model, tok = load_model_and_tokenizer(args.model_path)
    deactivate_model(model, detected, spec)
    for s in samples:
        print(f"[{s['id']}] {generate_report(model, tok, s['image_path'])}\n")
    del model; torch.cuda.empty_cache()

    # Condition 3: random deactivation (control)
    print("\n########## RANDOM DEACTIVATED (control) ##########")
    model, tok = load_model_and_tokenizer(args.model_path)
    ctrl = generate_random_control(detected, spec, seed=args.seed)
    deactivate_model(model, ctrl, spec)
    for s in samples:
        print(f"[{s['id']}] {generate_report(model, tok, s['image_path'])}\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_path", type=str, default=config.DEFAULT_MODEL_PATH)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--dataset", type=str, default="iu_xray", choices=list(config.DATASETS))
    common.add_argument("--annotation_path", type=str, default=None,
                        help="override config placeholder for the dataset annotation JSON")
    common.add_argument("--image_root", type=str, default=None,
                        help="override config placeholder for the dataset image root")

    p1 = sub.add_parser("make-random", parents=[common])
    p1.add_argument("--neuron_file", type=str, required=True)
    p1.add_argument("--out", type=str, required=True)
    p1.set_defaults(func=cmd_make_random)

    p2 = sub.add_parser("verify", parents=[common])
    p2.add_argument("--view", type=str, required=True)
    p2.add_argument("--neuron_file", type=str, required=True)
    p2.add_argument("--split", type=str, default="test")
    p2.add_argument("--num_samples", type=int, default=5)
    p2.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
