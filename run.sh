#!/bin/bash
# VNF: fine-tune ONLY the detected view-specific pattern neurons (multi-GPU DDP).
set -e

MODEL="OpenGVLab/Mini-InternVL2-4B-DA-Medical"
NEURONS="./neuron_deactivation"
TAG="Mini-InternVL2-4B-DA-Medical_iu_xray"
RATIO="0.05"

for VIEW in frontal lateral; do
  echo "🚀 VNF fine-tuning on $VIEW view"
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --master_port 29500 \
    train.py \
    --model_path "$MODEL" \
    --neuron_file "${NEURONS}/${TAG}_${VIEW}_neuron_${RATIO}.json" \
    --view "$VIEW" \
    --output_dir "./ckpt/${VIEW}_vnf_${RATIO}" \
    --num_epochs 1 \
    --learning_rate 4e-6 \
    --micro_batch_size 8 \
    --cutoff_len 1024
done

echo "🎉 VNF complete."
