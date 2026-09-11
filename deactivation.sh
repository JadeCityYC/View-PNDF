#!/bin/bash
# VNV: verify view-specific pattern neurons by deactivation.
set -e

MODEL="OpenGVLab/Mini-InternVL2-4B-DA-Medical"
NEURONS="./neuron_deactivation"
TAG="Mini-InternVL2-4B-DA-Medical_iu_xray"
RATIO="0.05"

for VIEW in frontal lateral; do
  echo "🔧 Verifying $VIEW neurons"
  CUDA_VISIBLE_DEVICES=0 python deactivation.py verify \
    --model_path "$MODEL" \
    --view "$VIEW" \
    --neuron_file "${NEURONS}/${TAG}_${VIEW}_neuron_${RATIO}.json" \
    --num_samples 5
done

echo "🎉 VNV complete."
