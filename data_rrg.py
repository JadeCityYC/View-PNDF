"""
Radiology report generation (RRG) data utilities for View-PNDF.
================================================================

Loads the IU-Xray / MIMIC-CXR PromptMRG-style annotations, maps each sample to
a view (``frontal`` = PA/AP, ``lateral`` = lateral), applies InternVL2's dynamic
tiling image preprocessing, and builds the ``<img>...<IMG_CONTEXT>...</img>``
prompt expected by the InternVL medical backbone.

Both datasets are exposed through a single ``load_dataset(name, ...)`` entry
point returning ``{split: [ {id, image_path(abs), report, view}, ... ]}``.

* **IU-Xray** annotation is a flat list where each entry is a single image with
  a ``view`` field, e.g.::

      {"id": "CXR2384_IM-0942", "image_path": ["CXR2384_IM-0942/0.png"],
       "report": "...", "view": ["PA"], "split": "test"}

  Since the shipped file only carries a ``test`` split, we deterministically
  re-split into train/val/test (paper uses 70/10/20) with a fixed seed.

* **MIMIC-CXR** annotation is the PromptMRG/R2Gen-style split dict
  ``{"train":[...], "val":[...], "test":[...]}`` where each entry carries an
  image list and a view field (``view`` / ``view_position`` / ``ViewPosition``).
  We keep the provided splits and only attach absolute paths + mapped views.

Data paths are intentionally NOT hard-coded here -- they come from ``config.py``
placeholders (or the matching env vars / CLI flags), so this repo can be
published without exposing local directories.
"""

import os
import json
import random

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

# Default prompt used for detection / generation (paper Fig. 1 style).
DEFAULT_PROMPT = (
    "Provide a detailed radiological analysis of the given chest X-ray, "
    "describing the cardiac size, pulmonary vascularity, lung fields, "
    "presence of any abnormalities, and any degenerative changes in the spine."
)


def map_view(view):
    """Map raw view tag(s) to 'frontal' / 'lateral' / 'other'.

    Handles the various spellings found across IU-Xray and MIMIC-CXR
    (``view_position`` uses codes like PA/AP/LATERAL/LL/LAO)."""
    v = view[0] if isinstance(view, (list, tuple)) and view else view
    v = str(v).strip().lower()
    if v in ("pa", "ap", "frontal"):
        return "frontal"
    if v in ("lateral", "ll", "lao", "rao"):
        return "lateral"
    return "other"


# --------------------------------------------------------------------------- #
# InternVL2 dynamic-tiling preprocessing (standard recipe)
# --------------------------------------------------------------------------- #
def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_diff = float("inf")
    best = (1, 1)
    area = width * height
    for r in target_ratios:
        tar = r[0] / r[1]
        diff = abs(aspect_ratio - tar)
        if diff < best_diff or (diff == best_diff and area > 0.5 * image_size * image_size * r[0] * r[1]):
            best_diff = diff
            best = r
    return best


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """InternVL2 dynamic tiling: split a high-res image into up to ``max_num``
    448x448 tiles plus an optional global thumbnail."""
    w, h = image.size
    ar = w / h
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )
    ratio = _find_closest_aspect_ratio(ar, target_ratios, w, h, image_size)
    tw, th = image_size * ratio[0], image_size * ratio[1]
    blocks = ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = []
    cols = tw // image_size
    for i in range(blocks):
        box = ((i % cols) * image_size, (i // cols) * image_size,
               ((i % cols) + 1) * image_size, ((i // cols) + 1) * image_size)
        tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def load_image(image_file, input_size=448, max_num=12, dynamic=True):
    """Load one image -> pixel_values tensor (num_tiles, 3, H, W)."""
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size)
    if dynamic:
        tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        tiles = [image.resize((input_size, input_size))]
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values


# --------------------------------------------------------------------------- #
# Annotation loading / view splitting
# --------------------------------------------------------------------------- #
def load_iu_xray(annotation_path, image_root, seed=42,
                 train_ratio=0.7, val_ratio=0.1):
    """Load IU-Xray flat annotation, attach absolute image paths and mapped
    views, and re-split into train/val/test with a fixed seed.

    Returns ``{split: [ {id, image_path(abs), report, view, split}, ... ]}``.
    """
    with open(annotation_path, "r") as f:
        data = json.load(f)

    samples = []
    for x in data:
        view = map_view(x.get("view"))
        if view == "other":
            continue  # drop notCXR / unknown
        abs_path = os.path.join(image_root, x["image_path"][0])
        if not os.path.exists(abs_path):
            continue
        samples.append({
            "id": x["id"],
            "image_path": abs_path,
            "report": x["report"],
            "view": view,
        })

    rng = random.Random(seed)
    rng.shuffle(samples)
    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = {
        "train": samples[:n_train],
        "val": samples[n_train:n_train + n_val],
        "test": samples[n_train + n_val:],
    }
    return splits


def _extract_view(entry):
    """Pull a view tag out of an annotation entry regardless of key spelling."""
    for key in ("view", "view_position", "ViewPosition", "view_pos"):
        if key in entry and entry[key] not in (None, "", []):
            return map_view(entry[key])
    return "other"


def load_mimic_cxr(annotation_path, image_root):
    """Load a PromptMRG/R2Gen-style MIMIC-CXR annotation.

    The file is a dict of splits, each a list of entries carrying an image list
    and a view field.  We keep the provided train/val/test splits and only
    attach absolute image paths + mapped views, dropping non-frontal/lateral or
    missing-image entries.

    Returns ``{split: [ {id, image_path(abs), report, view}, ... ]}``.
    """
    with open(annotation_path, "r") as f:
        data = json.load(f)

    splits = {}
    for split_name, entries in data.items():
        out = []
        for x in entries:
            view = _extract_view(x)
            if view == "other":
                continue
            img = x["image_path"][0] if isinstance(x["image_path"], list) else x["image_path"]
            abs_path = os.path.join(image_root, img)
            if not os.path.exists(abs_path):
                continue
            out.append({
                "id": x.get("id", x.get("study_id", img)),
                "image_path": abs_path,
                "report": x["report"],
                "view": view,
            })
        splits[split_name] = out
    # Normalise split names ('validate'/'valid' -> 'val').
    for alias in ("validate", "valid"):
        if alias in splits and "val" not in splits:
            splits["val"] = splits.pop(alias)
    return splits


def load_dataset(name, annotation_path, image_root, **kwargs):
    """Dataset dispatcher used by detect/deactivation/train/generate.

    ``name`` is one of the keys in ``config.DATASETS`` (currently ``iu_xray``
    and ``mimic_cxr``).  Extra kwargs (e.g. ``seed``) are forwarded to the
    IU-Xray re-splitter and ignored by MIMIC-CXR.
    """
    if name == "iu_xray":
        return load_iu_xray(annotation_path, image_root, **kwargs)
    if name == "mimic_cxr":
        return load_mimic_cxr(annotation_path, image_root)
    raise ValueError(f"Unsupported dataset: {name} (known: iu_xray, mimic_cxr)")


def filter_by_view(samples, view):
    """Keep only samples of a given view ('frontal' or 'lateral')."""
    return [s for s in samples if s["view"] == view]


# --------------------------------------------------------------------------- #
# Prompt / input construction for InternVL (Phi-3 chat template)
# --------------------------------------------------------------------------- #
def build_image_prompt(model, tokenizer, num_tiles, question=DEFAULT_PROMPT,
                       system_message=None):
    """Build the input_ids for one image+question using the Phi-3 chat template
    that the model ships (``model.conv_template``)."""
    from copy import deepcopy

    n_img_tok = model.num_image_token * num_tiles
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * n_img_tok + IMG_END_TOKEN

    template = deepcopy(model.conv_template)
    if system_message is not None:
        template.system_message = system_message
    template.append_message(template.roles[0], f"<image>\n{question}")
    template.append_message(template.roles[1], None)
    query = template.get_prompt().replace("<image>", image_tokens)

    model_inputs = tokenizer(query, return_tensors="pt")
    return model_inputs.input_ids, model_inputs.attention_mask


def build_training_example(model, tokenizer, num_tiles, question, report,
                           cutoff_len=1024):
    """Build (input_ids, attention_mask, labels) for VNF fine-tuning: the
    prompt tokens are masked (-100) so loss is computed only on the report."""
    from copy import deepcopy

    n_img_tok = model.num_image_token * num_tiles
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * n_img_tok + IMG_END_TOKEN

    template = deepcopy(model.conv_template)
    template.append_message(template.roles[0], f"<image>\n{question}")
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt().replace("<image>", image_tokens)

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    report_ids = tokenizer(report + tokenizer.eos_token, add_special_tokens=False).input_ids

    input_ids = (prompt_ids + report_ids)[:cutoff_len]
    labels = ([-100] * len(prompt_ids) + report_ids)[:cutoff_len]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
