#!/bin/bash

set -e 
# ================= 配置路径 =================
# 预处理数据的根目录 (来自你的上一个脚本)
DATA_BASE_DIR="../data_process/processed_data"

# 结果输出根目录 (存放模型和JSON日志)
RESULT_BASE_DIR="./finetune_results"

# 预训练模型路径 (请确保此路径正确，相对于当前脚本位置)
PRETRAINED_MODEL_PATH="../models/pre-trained_model.bin"
VOCAB_PATH="../models/encryptd_vocab.txt"
CONFIG_PATH="../models/bert/base_config.json"

# ============================================
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 确保输出父目录存在
if [ ! -d "$RESULT_BASE_DIR" ]; then
    echo "创建结果输出根目录: $RESULT_BASE_DIR"
    mkdir -p "$RESULT_BASE_DIR"
fi

echo "开始批量运行 ET-BERT Fine-tuning..."
echo "数据源: $DATA_BASE_DIR"
echo "结果输出: $RESULT_BASE_DIR"
echo "---------------------------------------"

# 定义执行函数
run_finetune() {
    local index=$1
    local name=$2
    local data_dir_name=$3
    local sub_dir_name=$4

    # 构造数据路径
    local data_dir="$DATA_BASE_DIR/$data_dir_name"
    local train_path="$data_dir/train_dataset.tsv"
    local dev_path="$data_dir/valid_dataset.tsv"
    local test_path="$data_dir/test_dataset.tsv"

    # 构造输出目录
    local output_dir="$RESULT_BASE_DIR/$sub_dir_name"
    local output_model_path="$output_dir/finetuned_model.bin"

    echo "========== [$index/7] 正在微调数据集: $name =========="
    
    # 检查数据是否存在
    if [ ! -f "$train_path" ]; then
        echo "错误: 找不到训练数据 $train_path ，跳过。"
        return
    fi

    # 清理旧的输出目录
    if [ -d "$output_dir" ]; then
        echo "清理旧输出: $output_dir"
        rm -rf "$output_dir"
    fi
    mkdir -p "$output_dir"

    # 执行 Python 脚本
    python run_classifier_freeze.py \
        --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
        --vocab_path "$VOCAB_PATH" \
        --config_path "$CONFIG_PATH" \
        --train_path "$train_path" \
        --dev_path "$dev_path" \
        --test_path "$test_path" \
        --output_model_path "$output_model_path" \
        --epochs_num 20 \
        --batch_size 32 \
        --embedding word_pos_seg \
        --encoder transformer \
        --mask fully_visible \
        --seq_length 512 \
        --learning_rate 6e-5 \
        --seed 42 \
        --freeze_encoder

    echo "完成: $name -> 结果保存在 $output_dir"
    echo ""
}

# --- 开始循环执行 ---

# 1. VPN-service
# run_finetune "1" "VPN-service" "VPN-service" "5x128_VPN-service"

# run_finetune "2" "VPN-app" "VPN-app"

# 3. USTC-TFC
# run_finetune "3" "USTC-TFC" "USTC-TFC"

# 4. Cross-Platform-Android
# run_finetune "4" "Cross-Platform-Android" "Cross-Platform-Android"

# 5. Cross-Platform-IOS
# run_finetune "5" "Cross-Platform-IOS" "Cross-Platform-IOS"

# # 6. Browser-Dataset
# run_finetune "6" "Browser-Dataset" "Browser-Dataset"

# run_finetune "3" "USTC-TFC" "USTC-TFC"

# 7. CSTNET
# run_finetune "7" "CSTNET" "CSTNET" "5x128_CSTNET_freeze"
run_finetune "3" "DataCon2021_part1" "DataCon2021_part1" "5x128_DataCon2021_part1_freeze"
run_finetune "8" "AES-128-GCM-Dataset" "AES-128-GCM-Dataset" "5x128_AES-128-GCM-Dataset_freeze"
run_finetune "8" "AES-256-GCM-Dataset" "AES-256-GCM-Dataset" "5x128_AES-256-GCM-Dataset_freeze"
run_finetune "8" "chacha20-poly1305-Dataset" "chacha20-poly1305-Dataset" "5x128_chacha20-poly1305-Dataset_freeze"
run_finetune "8" "mix-Dataset" "mix-Dataset" "5x128_mix-Dataset_freeze"

# echo "---------------------------------------"
# echo "所有 ET-BERT 微调任务已完成！"