"""
View-specific Neuron Detection (VND) -- View-PNDF
=================================================

Detects the pattern neurons that are selectively responsive to a particular
chest X-ray view (frontal / lateral) in a medical VLM whose language decoder is
Phi-3 (``Mini-InternVL2-4B-DA-Medical``).

For each detection sample (image + prompt) of a given view we run a full VLM
forward pass and, via forward hooks (see ``view_pndf.ActivationCollector``),
record the per-neuron activation magnitude for every decoder layer and every
projection (paper Eq. 6).  Per sample we keep the top-ratio neurons per layer
(paper Eq. 7 candidate set); intersecting these across all samples yields the
view's pattern neurons, which are written to a JSON file consumed by
``deactivation.py`` (VNV) and ``train.py`` (VNF).

Example
-------
    python detect.py \
        --model_path OpenGVLab/Mini-InternVL2-4B-DA-Medical \
        --dataset iu_xray \
        --views frontal,lateral \
        --atten_ratio 0.05 --ffn_ratio 0.05 \
        --sample_size 100 \
        --output_path ./neuron_deactivation/
"""

import os
import json
import random
import argparse

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from view_pndf import (
    ActivationCollector,
    build_projection_spec,
    topk_indices_per_layer,
    intersect_sample_sets,
    merge_image_embeds,
    save_neuron,
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
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    return model, tokenizer


@torch.no_grad()
def detect_view_neurons(model, tokenizer, samples, spec,
                        atten_ratio=0.05, ffn_ratio=0.05,
                        prompt=DEFAULT_PROMPT, dynamic_tiles=False):
    """Detect the pattern neurons for one view.

    ``samples`` is a list of ``{image_path, report, view, ...}`` all of the
    same view.  Returns ``{group: {layer_str: set}}`` (intersection over
    samples).
    """
    collector = ActivationCollector(model, spec).attach()
    per_sample_selected = []
    errors = 0

    for s in tqdm(samples, desc="detecting"):
        try:
            pixel_values = load_image(s["image_path"], dynamic=dynamic_tiles).to(torch.bfloat16).cuda()
            input_ids, _ = build_image_prompt(model, tokenizer, pixel_values.shape[0], question=prompt)
            input_ids = input_ids.cuda()

            embeds = merge_image_embeds(model, pixel_values, input_ids)
            model.language_model.model(inputs_embeds=embeds)
            scores = collector.pop()

            selected = topk_indices_per_layer(scores, atten_ratio, ffn_ratio)
            per_sample_selected.append(selected)
        except Exception as e:  # noqa: BLE001 - keep detecting on OOM / bad image
            errors += 1
            collector.pop()  # discard partial scores
            print(f"[warn] sample {s.get('id')} failed: {e}")

    collector.detach()
    print(f"Detection complete; {len(per_sample_selected)} ok, {errors} errors")
    return intersect_sample_sets(per_sample_selected)


def get_dataset_splits(args):
    annotation_path, image_root = config.get_dataset_paths(
        args.dataset, args.annotation_path, args.image_root
    )
    return load_dataset(args.dataset, annotation_path, image_root, seed=args.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=str, default="iu_xray", choices=list(config.DATASETS))
    parser.add_argument("--annotation_path", type=str, default=None,
                        help="override config placeholder for the dataset annotation JSON")
    parser.add_argument("--image_root", type=str, default=None,
                        help="override config placeholder for the dataset image root")
    parser.add_argument("--views", type=str, default="frontal,lateral")
    parser.add_argument("--split", type=str, default="train", help="split to draw detection samples from")
    parser.add_argument("--atten_ratio", type=float, default=0.05)
    parser.add_argument("--ffn_ratio", type=float, default=0.05)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--dynamic_tiles", action="store_true", help="use InternVL dynamic tiling (default: single tile)")
    parser.add_argument("--output_path", type=str, default=config.NEURON_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    random.seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    spec = build_projection_spec(model)
    print(f"ProjectionSpec: {spec}")

    splits = get_dataset_splits(args)
    pool = splits[args.split]
    model_tag = os.path.basename(args.model_path.rstrip("/"))

    # Detect per-view pattern neurons, and also track the shared (view-general)
    # set as the intersection across views (for VNV random-control / VNF protect).
    per_view_neurons = {}
    for view in args.views.split(","):
        view_samples = filter_by_view(pool, view)
        if args.sample_size > 0 and len(view_samples) > args.sample_size:
            view_samples = random.sample(view_samples, args.sample_size)
        print(f"\n=== View '{view}': {len(view_samples)} detection samples ===")

        neurons = detect_view_neurons(
            model, tokenizer, view_samples, spec,
            atten_ratio=args.atten_ratio, ffn_ratio=args.ffn_ratio,
            dynamic_tiles=args.dynamic_tiles,
        )
        per_view_neurons[view] = neurons

        total = sum(len(v) for v in neurons["attn_q"].values()) + \
            sum(len(v) for v in neurons["fwd_up"].values())
        fname = f"{model_tag}_{args.dataset}_{view}_neuron_{args.atten_ratio}.json"
        out = os.path.join(args.output_path, fname)
        save_neuron(neurons, out)
        print(f"  saved {out}  (attn_q+fwd_up neurons: {total})")

    # Shared / view-general neurons = intersection of the per-view sets.
    if len(per_view_neurons) > 1:
        shared = {g: {} for g in GROUPS}
        views = list(per_view_neurons.keys())
        for group in GROUPS:
            layers = per_view_neurons[views[0]][group].keys()
            for layer in layers:
                common = None
                for v in views:
                    s = per_view_neurons[v][group].get(layer, set())
                    common = s if common is None else (common & s)
                shared[group][layer] = common if common is not None else set()
        fname = f"{model_tag}_{args.dataset}_shared_neuron_{args.atten_ratio}.json"
        save_neuron(shared, os.path.join(args.output_path, fname))
        print(f"\nsaved shared (view-general) neurons -> {fname}")


if __name__ == "__main__":
    main()
