"""
View-specific report generation for View-PNDF.
==============================================

Runs a (optionally VNF-fine-tuned) VLM over a test split and writes one report
per image, grouped by view.  The resulting JSON is consumed by:

  * ``integrate.py``     -- LLM consolidation of frontal+lateral into one report
  * ``evaluate_nlg.py``  -- BLEU / METEOR / ROUGE-L on integrated reports
  * ``evaluate_llm.py``  -- GPT-4o clinical scoring on view-specific reports

You can point ``--model_path`` at either the base backbone or a VNF checkpoint
(``./ckpt/lateral_vnf_0.05``) to compare "w/o" vs "w/" View-PNDF.

Output schema (``predictions.json``)::

    [
      {"id": "...", "view": "frontal", "gt": "...", "pred": "..."},
      {"id": "...", "view": "lateral", "gt": "...", "pred": "..."},
      ...
    ]

Frontal and lateral views of the SAME study share an ``id`` (or a ``study_id``
if present), which ``integrate.py`` uses to pair them.

Example
-------
    python generate.py \
        --model_path ./ckpt/lateral_vnf_0.05 \
        --dataset iu_xray --split test \
        --out ./outputs/iu_xray_vnf_predictions.json
"""

import os
import json
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, GenerationConfig

from data_rrg import (
    load_dataset,
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
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@torch.no_grad()
def generate_report(model, tokenizer, image_path, prompt=DEFAULT_PROMPT,
                    max_new_tokens=256, dynamic_tiles=False):
    pixel_values = load_image(image_path, dynamic=dynamic_tiles).to(torch.bfloat16).cuda()
    input_ids, attention_mask = build_image_prompt(model, tokenizer, pixel_values.shape[0], question=prompt)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id, bos_token_id=bos_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids.cuda(),
        attention_mask=attention_mask.cuda(),
        generation_config=gen_cfg,
    )
    # Strip the prompt tokens so only the freshly generated report remains.
    gen_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=config.DEFAULT_MODEL_PATH,
                        help="base backbone OR a VNF checkpoint dir")
    parser.add_argument("--dataset", type=str, default="iu_xray", choices=list(config.DATASETS))
    parser.add_argument("--annotation_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--views", type=str, default="frontal,lateral")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=-1, help="cap per view (debug); -1 = all")
    parser.add_argument("--dynamic_tiles", action="store_true")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    annotation_path, image_root = config.get_dataset_paths(
        args.dataset, args.annotation_path, args.image_root
    )
    splits = load_dataset(args.dataset, annotation_path, image_root, seed=args.seed)
    pool = splits[args.split]

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    wanted = set(args.views.split(","))
    records = []
    per_view_count = {v: 0 for v in wanted}
    for s in tqdm(pool, desc="generating"):
        if s["view"] not in wanted:
            continue
        if 0 < args.max_samples <= per_view_count[s["view"]]:
            continue
        try:
            pred = generate_report(
                model, tokenizer, s["image_path"],
                max_new_tokens=args.max_new_tokens, dynamic_tiles=args.dynamic_tiles,
            )
        except Exception as e:  # noqa: BLE001 - skip unreadable images / OOM
            print(f"[warn] {s.get('id')} failed: {e}")
            continue
        records.append({"id": s["id"], "view": s["view"], "gt": s["report"], "pred": pred})
        per_view_count[s["view"]] += 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"wrote {len(records)} predictions -> {args.out}  ({per_view_count})")


if __name__ == "__main__":
    main()
