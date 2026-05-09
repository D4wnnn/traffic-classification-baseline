#!/bin/bash

# ================= 脚本说明 =================
# 仿照 ET-BERT 结构重构的 TrafficFormer 微调脚本
# 保留了 TrafficFormer 的特定参数：Seq=320, Epoch=50, LR=6e-5, Batch=128
# ============================================

set -e

# ================= 配置路径 =================
# 预处理数据的根目录 (假设你已经按照 ET-BERT 的格式整理好了数据)
# 如果你的数据还在 data_generation 目录，请修改此处为 "../data_generation/dataset"
DATA_BASE_DIR="../data_generation/dataset"

# 结果输出根目录
RESULT_BASE_DIR="./finetune_results"

# 模型相关路径
# 注意：这里使用了你提供的 checkpoint-120000 模型
PRETRAINED_MODEL_PATH="../models/nomoe_bertflow_pre-trained_model.bin-120000"
VOCAB_PATH="../models/encryptd_vocab.txt"
CONFIG_PATH="../models/bert/base_config.json"

# ================= 显卡与环境 =================
# TrafficFormer 原始脚本使用了 8 张卡，显存需求较大 (Seq 320)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 确保输出父目录存在
if [ ! -d "$RESULT_BASE_DIR" ]; then
    echo "创建结果输出根目录: $RESULT_BASE_DIR"
    mkdir -p "$RESULT_BASE_DIR"
fi

echo "开始批量运行 TrafficFormer Fine-tuning..."
echo "预训练模型: $PRETRAINED_MODEL_PATH"
echo "数据源: $DATA_BASE_DIR"
echo "---------------------------------------"

# ================= 定义执行函数 =================
run_finetune() {
    local index=$1
    local name=$2
    local sub_dir_name=$3

    # 构造数据路径
    local data_dir="$DATA_BASE_DIR/$sub_dir_name"
    local train_path="$data_dir/train_dataset.tsv"
    local dev_path="$data_dir/valid_dataset.tsv"
    local test_path="$data_dir/test_dataset.tsv"

    # 构造输出目录
    local output_dir="$RESULT_BASE_DIR/${sub_dir_name}_freeze"
    local output_model_path="$output_dir/finetuned_model.bin"

    echo "========== [$index] 正在微调数据集: $name =========="
    
    # 检查数据是否存在
    if [ ! -f "$train_path" ]; then
        echo "错误: 找不到训练数据 $train_path ，跳过。"
        return
    fi

    # 清理旧的输出目录 (可选，防止混淆)
    if [ -d "$output_dir" ]; then
        echo "清理旧输出: $output_dir"
        rm -rf "$output_dir"
    fi
    mkdir -p "$output_dir"

    # 执行 Python 脚本
    # 参数说明：
    python run_classifier_freeze.py \
        --vocab_path "$VOCAB_PATH" \
        --config_path "$CONFIG_PATH" \
        --pretrained_model_path "$PRETRAINED_MODEL_PATH" \
        --train_path "$train_path" \
        --dev_path "$dev_path" \
        --test_path "$test_path" \
        --output_model_path "$output_model_path" \
        --epochs_num 20 \
        --earlystop 20 \
        --batch_size 128 \
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

# ================= 开始循环执行 =================

# 1. VPN-service

# # 2. VPN-app
# run_finetune "2" "VPN-app" "VPN-app"

# # 3. USTC-TFC
# run_finetune "3" "USTC-TFC" "USTC-TFC"

# # 4. Cross-Platform-Android
# run_finetune "4" "Cross-Platform-Android" "Cross-Platform-Android"

# 5. Cross-Platform-IOS
# run_finetune "5" "Cross-Platform-IOS" "Cross-Platform-IOS"

# # 6. Browser-Dataset
# run_finetune "6" "Browser-Dataset" "Browser-Dataset"
# run_finetune "3" "USTC-TFC" "USTC-TFC"c
# 7. CSTNET
run_finetune "7" "CSTNET" "CSTNET"
run_finetune "8" "AES-128-GCM-Dataset" "AES-128-GCM-Dataset"
run_finetune "8" "AES-256-GCM-Dataset" "AES-256-GCM-Dataset"
run_finetune "8" "chacha20-poly1305-Dataset" "chacha20-poly1305-Dataset"
run_finetune "8" "mix-Dataset" "mix-Dataset"
run_finetune "8" "DataCon2021_part1" "DataCon2021_part1"



echo "---------------------------------------"
echo "所有 TrafficFormer 微调任务已完成！"