#!/bin/bash
# VND: detect view-specific pattern neurons for the medical VLM.
set -e

MODEL="OpenGVLab/Mini-InternVL2-4B-DA-Medical"
DATASET="iu_xray"
RATIO="0.05"          # top-ratio for both attention and FFN (paper sweeps 0.5%-2%)
SAMPLE_SIZE=100       # detection samples per view (paper detection sample size B)
OUTPUT="./neuron_deactivation/"

CUDA_VISIBLE_DEVICES=0 python detect.py \
  --model_path "$MODEL" \
  --dataset "$DATASET" \
  --views "frontal,lateral" \
  --split train \
  --atten_ratio "$RATIO" \
  --ffn_ratio "$RATIO" \
  --sample_size "$SAMPLE_SIZE" \
  --output_path "$OUTPUT"

echo "🎉 VND complete. Neuron files in $OUTPUT"
