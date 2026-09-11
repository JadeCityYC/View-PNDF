"""
Central configuration for View-PNDF.
====================================

All machine-specific paths, model identifiers, and API endpoints are collected
here as *placeholders* so the repository can be published without leaking local
directories or credentials.  Override any of them either by editing this file,
by passing the corresponding command-line flag, or by exporting the matching
environment variable (env vars take precedence over the defaults below).

Nothing here is secret: real API keys are read from the environment at call
time (see ``evaluate_llm.py`` / ``integrate.py``) and are never stored in code.
"""

import os

# --------------------------------------------------------------------------- #
# Backbone VLM
# --------------------------------------------------------------------------- #
# The public InternVL medical checkpoint used in the paper's ablations.  Swap
# for any of the other backbones (MedGemma / LLaVA-Med / Hulu-Med); see
# ``view_pndf.build_projection_spec`` for how a new decoder family is described.
DEFAULT_MODEL_PATH = os.environ.get(
    "VIEWPNDF_MODEL_PATH", "OpenGVLab/Mini-InternVL2-4B-DA-Medical"
)

# --------------------------------------------------------------------------- #
# Datasets  (placeholders -- point these at your local copies)
# --------------------------------------------------------------------------- #
# Each dataset needs a PromptMRG-style annotation JSON and an image root.
DATASETS = {
    "iu_xray": {
        # Flat list of {id, image_path:[...], report, view:[...], split}.
        "annotation_path": os.environ.get(
            "IU_XRAY_ANNOTATION", "./data/iu_xray/annotation.json"
        ),
        "image_root": os.environ.get(
            "IU_XRAY_IMAGE_ROOT", "./data/iu_xray/images"
        ),
    },
    "mimic_cxr": {
        # Split dict {"train":[...], "val":[...], "test":[...]} of
        # {id, image_path:[...], report, view/view_position, ...}.
        "annotation_path": os.environ.get(
            "MIMIC_CXR_ANNOTATION", "./data/mimic_cxr/annotation.json"
        ),
        "image_root": os.environ.get(
            "MIMIC_CXR_IMAGE_ROOT", "./data/mimic_cxr/images"
        ),
    },
}

# Where detected neuron JSON files and fine-tuned checkpoints are written.
NEURON_DIR = os.environ.get("VIEWPNDF_NEURON_DIR", "./neuron_deactivation")
CKPT_DIR = os.environ.get("VIEWPNDF_CKPT_DIR", "./ckpt")

# --------------------------------------------------------------------------- #
# LLM evaluator / report-integrator (OpenAI-compatible endpoint)
# --------------------------------------------------------------------------- #
# API key is ALWAYS read from the environment; never hard-code it here.
#   export OPENAI_API_KEY=sk-...
# ``base_url`` lets you point at Azure/OpenAI-compatible/self-hosted gateways
# (e.g. a local vLLM server serving Qwen / DeepSeek for LLM-as-judge).
LLM_API_KEY_ENV = "OPENAI_API_KEY"
LLM_BASE_URL = os.environ.get("VIEWPNDF_LLM_BASE_URL", None)  # None -> official OpenAI
LLM_JUDGE_MODEL = os.environ.get("VIEWPNDF_JUDGE_MODEL", "gpt-4o")
LLM_INTEGRATOR_MODEL = os.environ.get("VIEWPNDF_INTEGRATOR_MODEL", "gpt-4o")


def get_dataset_paths(name, annotation_path=None, image_root=None):
    """Resolve (annotation_path, image_root) for a dataset, letting explicit
    CLI args override the config placeholders."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Known: {list(DATASETS)}")
    cfg = DATASETS[name]
    return (
        annotation_path or cfg["annotation_path"],
        image_root or cfg["image_root"],
    )
