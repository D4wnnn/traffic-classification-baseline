#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
SEED="${SEED:-42}"
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

if [[ -z "${N_GPUS:-}" ]]; then
  IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
  N_GPUS="${#GPU_ARRAY[@]}"
fi

DATA_BASE_DIR="${DATA_BASE_DIR:-../YaTC/1-data_processing/outputs}"
SPLIT_ROOT="${SPLIT_ROOT:-./rq2_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/rq2_outputs/RQ2}"
RATIOS="${RATIOS:-0.10 0.15 0.20 0.30 0.50 1.00}"
DATASETS="${DATASETS:-cstnet:finetune_data_cstnet aes_128_gcm:finetune_data_aes_128_gcm aes_256_gcm:finetune_data_AES-256-GCM-Dataset chacha20_poly1305:finetune_data_chacha20-poly1305-Dataset mix:finetune_data_mix-Dataset datacon2021_part1:finetune_data_DataCon2021_part1 datacon2021_part2:finetune_data_DataCon2021_part2}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-./src/pre-train.pth}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-10}"
METHOD="${METHOD:-netmamba}"

for spec in $DATASETS; do
  dataset="${spec%%:*}"
  data_dir_name="${spec#*:}"
  input_dir="$DATA_BASE_DIR/$data_dir_name"
  train_dir="$input_dir/train"
  if [[ ! -d "$train_dir" ]]; then
    echo "[SKIP] missing $train_dir"
    continue
  fi

  "$PYTHON_BIN" -B rq2_label_efficiency.py \
    --input "$input_dir" \
    --output-root "$SPLIT_ROOT" \
    --dataset "$dataset" \
    --ratios $RATIOS \
    --seed "$SEED"

  nb_classes=$(find "$train_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
  for ratio in $RATIOS; do
    tag=$(printf "%03dpct" "$("$PYTHON_BIN" -c "print(round(float('$ratio') * 100))")")
    rq2_data_dir="$SPLIT_ROOT/$dataset/$tag"
    output_dir="$OUTPUT_ROOT/$dataset/$tag/$METHOD"
    log_dir="$output_dir/logs"

    rm -rf "$output_dir"
    mkdir -p "$log_dir"
    echo "[RQ2][NetMamba] dataset=$dataset ratio=$ratio classes=$nb_classes output=$output_dir"
    torchrun --nproc_per_node="$N_GPUS" src/fine-tune.py \
      --blr 2e-3 \
      --epochs "$EPOCHS" \
      --nb_classes "$nb_classes" \
      --finetune "$PRETRAINED_MODEL_PATH" \
      --data_path "$rq2_data_dir" \
      --output_dir "$output_dir" \
      --log_dir "$log_dir" \
      --model net_mamba_classifier \
      --batch_size "$BATCH_SIZE" \
      --num_workers "$NUM_WORKERS" \
      --no_amp \
      --seed "$SEED" 2>&1 | tee "$output_dir/train_eval.log"
  done
done
