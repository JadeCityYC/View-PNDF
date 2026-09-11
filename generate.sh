#!/bin/bash
# Generate + integrate + evaluate view-specific reports.
# Compares the base backbone ("w/o") against a VNF checkpoint ("w/").
set -e

DATASET="${DATASET:-iu_xray}"
SPLIT="${SPLIT:-test}"
RATIO="${RATIO:-0.05}"
OUT="./outputs/${DATASET}"
mkdir -p "$OUT"

# --- w/o View-PNDF: base backbone -----------------------------------------
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --dataset "$DATASET" --split "$SPLIT" \
  --out "${OUT}/base_predictions.json"

# --- w/ View-PNDF: VNF-fine-tuned lateral checkpoint ----------------------
# (Swap in the frontal checkpoint or a merged model as needed.)
CUDA_VISIBLE_DEVICES=0 python generate.py \
  --model_path "./ckpt/lateral_vnf_${RATIO}" \
  --dataset "$DATASET" --split "$SPLIT" \
  --out "${OUT}/vnf_predictions.json"

# --- Integrate per-view reports into one report (LLM) ---------------------
# Requires OPENAI_API_KEY in the environment.
python integrate.py --predictions "${OUT}/vnf_predictions.json" \
  --out "${OUT}/vnf_integrated.json"

# --- NLG metrics on the integrated report (Tab. 1 / Tab. 2) ---------------
python evaluate_nlg.py --input "${OUT}/vnf_integrated.json"

# --- LLM clinical scoring on view-specific reports (Tab. 3 / Tab. 5) ------
python evaluate_llm.py --input "${OUT}/vnf_predictions.json" \
  --out "${OUT}/vnf_llm_scores.json"

echo "🎉 generation + evaluation complete. Artifacts in ${OUT}"
