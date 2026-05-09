#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

PRETRAINED_MODEL="/mnt/8T/lc/datasets/netFoundHF" 
# 确保这里指向的是包含 labels 字段的 JSON 文件或 Arrow 文件夹
TRAIN_DATA="/mnt/8T/lc/netFound/scripts/output_folder/final/train"
TEST_DATA="/mnt/8T/lc/netFound/scripts/output_folder/final/test"
OUTPUT_DIR="./results_finetune2"

NUM_LABELS=41 
nproc_per_node=2

# torchrun --nproc_per_node=$nproc_per_node NetfoundFinetuning.py \
#     --model_name_or_path "$PRETRAINED_MODEL" \
#     --train-dir "$TRAIN_DATA" \
#     --test-dir "$TEST_DATA" \
#     --output_dir "$OUTPUT_DIR" \
#     --num_labels $NUM_LABELS \
#     --do_train \
#     --do_eval \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 10 \
#     --logging_steps 50 \
#     --save_strategy "epoch" \
#     --eval_strategy "epoch" \
#     --load_best_model_at_end True \
#     --metric_for_best_model "weighted_f1" \
#     --save_total_limit 2 \
#     --remove_unused_columns False \
#     --dataloader_num_workers 4 \
#     --flat False \
#     --no_meta False \
#     --netfound_large True \
#     --save_safetensors False \
#     --max_seq_length 500
torchrun --nproc_per_node=$nproc_per_node NetfoundFinetuning.py \
    --train_dir $TRAIN_DATA \
    --test-dir "$TEST_DATA" \
    --model_name_or_path "$PRETRAINED_MODEL" \
    --output_dir ./fwwinetuned_mo2del \
    --report_to tensorboard \
    --overwrite_output_dir \
    --save_safetensors false \
    --do_train \
    --do_eval \
    --eval_strategy epoch \
    --save_strategy epoch \
    --learning_rate 0.01 \
    --num_train_epochs 20 \
    --problem_type single_label_classification \
    --num_labels 41 \
    --load_best_model_at_end \
    --netfound_large True \
    --ddp_find_unused_parameters True \
    --per_device_train_batch_size 16


