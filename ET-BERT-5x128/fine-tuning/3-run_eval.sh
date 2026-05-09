#!/bin/bash
set -e 

# ================= 配置路径 =================
DATA_BASE_DIR="../data_process/processed_data"
FINETUNE_RESULT_DIR="./finetune_results"
EVAL_OUTPUT_DIR="./evaluation_results_robustness" # 修改输出目录名以区分

# 模型配置
VOCAB_PATH="../models/encryptd_vocab.txt"
CONFIG_PATH="../models/bert/base_config.json"
# ============================================

CUDA_VISIBLE_DEVICES=0

if [ ! -d "$EVAL_OUTPUT_DIR" ]; then
    mkdir -p "$EVAL_OUTPUT_DIR"
fi

run_eval() {
    local index=$1
    local name=$2
    local data_dir_name=$3
    local sub_dir_name=$4

    # 构造完整的数据路径
    local data_dir="$DATA_BASE_DIR/$data_dir_name"
    local train_path="$data_dir/train_dataset.tsv"
    local dev_path="$data_dir/valid_dataset.tsv"
    local test_path="$data_dir/test_dataset.tsv"

    local model_path="$FINETUNE_RESULT_DIR/$sub_dir_name/finetuned_model.bin"
    local output_dir="$EVAL_OUTPUT_DIR/$sub_dir_name"
    local json_result_path="$output_dir/robustness_stats.json"

    echo "========== [$index] 正在评估: $name =========="
    
    if [ ! -f "$model_path" ]; then
        echo "错误: 找不到模型 $model_path ，跳过。"
        return
    fi

    mkdir -p "$output_dir"
    
    # 清理旧的结果文件，避免追加混乱
    rm -f "$json_result_path"

    # --- 循环测试不同的噪声配置 ---
    
    # 1. Baseline (No noise)
    # echo "Running Baseline..."
    # python run_evaluation.py \
    #     --load_model_path "$model_path" --vocab_path "$VOCAB_PATH" --config_path "$CONFIG_PATH" \
    #     --train_path "$train_path" --dev_path "$dev_path" --test_path "$test_path" \
    #     --result_path "$json_result_path" \
    #     --batch_size 32 --seq_length 512 --seed 42 \
    #     --embedding word_pos_seg --encoder transformer --mask fully_visible \
    #     --perturb_type none --noise_ratio 0.0

    # 2. Byte Masking Tests
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5; do
        echo "Running Byte Masking @ $ratio..."
        python run_evaluation.py \
            --load_model_path "$model_path" --vocab_path "$VOCAB_PATH" --config_path "$CONFIG_PATH" \
            --train_path "$train_path" --dev_path "$dev_path" --test_path "$test_path" \
            --result_path "$json_result_path" \
            --batch_size 32 --seq_length 512 --seed 42 \
            --embedding word_pos_seg --encoder transformer --mask fully_visible \
            --perturb_type byte_mask --noise_ratio $ratio
    done

    # # 3. Packet Drop Tests
    # for ratio in 0.2 0.4 0.6; do
    #     echo "Running Packet Drop @ $ratio..."
    #     python run_evaluation.py \
    #         --load_model_path "$model_path" --vocab_path "$VOCAB_PATH" --config_path "$CONFIG_PATH" \
    #         --train_path "$train_path" --dev_path "$dev_path" --test_path "$test_path" \
    #         --result_path "$json_result_path" \
    #         --batch_size 32 --seq_length 512 --seed 42 \
    #         --embedding word_pos_seg --encoder transformer --mask fully_visible \
    #         --perturb_type packet_drop --noise_ratio $ratio
    # done

    echo "完成: $name. 结果已保存至 $json_result_path"
    echo ""
}

# --- 任务列表 ---

run_eval "8" "AES-128-GCM-Dataset" "AES-128-GCM-Dataset" "5x128_AES-128-GCM-Dataset"

echo "所有评估任务已完成！"