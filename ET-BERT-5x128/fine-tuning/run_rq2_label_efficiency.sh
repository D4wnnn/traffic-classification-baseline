#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

DATA_BASE_DIR="${DATA_BASE_DIR:-../data_process/processed_data}"
SPLIT_ROOT="${SPLIT_ROOT:-./rq2_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/rq2_outputs/RQ2}"
RATIOS="${RATIOS:-0.10 0.15 0.20 0.30 0.50 1.00}"
DATASETS="${DATASETS:-CSTNET AES-128-GCM-Dataset AES-256-GCM-Dataset chacha20-poly1305-Dataset mix-Dataset DataCon2021_part1}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-../models/pre-trained_model.bin}"
VOCAB_PATH="${VOCAB_PATH:-../models/encryptd_vocab.txt}"
CONFIG_PATH="${CONFIG_PATH:-../models/bert/base_config.json}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-96}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
LR="${LR:-6e-5}"
SEED="${SEED:-42}"
METHOD="${METHOD:-et_bert}"

for dataset in $DATASETS; do
  input_dir="$DATA_BASE_DIR/$dataset"
  if [[ ! -f "$input_dir/train_dataset.tsv" ]]; then
    echo "[SKIP] missing $input_dir/train_dataset.tsv"
    continue
  fi

  "$PYTHON_BIN" -B rq2_label_efficiency.py \
    --input "$input_dir" \
    --output-root "$SPLIT_ROOT" \
    --dataset "$dataset" \
    --ratios $RATIOS \
    --seed "$SEED"

  for ratio in $RATIOS; do
    tag=$(printf "%03dpct" "$("$PYTHON_BIN" -c "print(round(float('$ratio') * 100))")")
    data_dir="$SPLIT_ROOT/$dataset/$tag"
    output_dir="$OUTPUT_ROOT/$dataset/$tag/$METHOD"
    output_model_path="$output_dir/finetuned_model.bin"

    rm -rf "$output_dir"
    mkdir -p "$output_dir"
    echo "[RQ2][ET-BERT] dataset=$dataset ratio=$ratio output=$output_dir"
    "$PYTHON_BIN" run_classifier.py \
      --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
      --vocab_path "$VOCAB_PATH" \
      --config_path "$CONFIG_PATH" \
      --train_path "$data_dir/train_dataset.tsv" \
      --dev_path "$data_dir/valid_dataset.tsv" \
      --test_path "$data_dir/test_dataset.tsv" \
      --output_model_path "$output_model_path" \
      --epochs_num "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --embedding word_pos_seg \
      --encoder transformer \
      --mask fully_visible \
      --seq_length "$SEQ_LENGTH" \
      --learning_rate "$LR" \
      --seed "$SEED" 2>&1 | tee "$output_dir/train_eval.log"
  done
done
