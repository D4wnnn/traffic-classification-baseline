#!/bin/bash

# ================= 脚本说明 =================
# TrafficFormer 批量评估脚本
# 作用：加载微调好的模型，在测试集上计算 F1, Precision, Recall, Confusion Matrix
# 原理：利用 run_classifier.py 中 "train_path is None" 的逻辑，
#       将 Test Set 传入 --dev_path 参数进行纯推理模式。
# ============================================

set -e

# ================= 配置路径 =================
# 数据集根目录 (与训练脚本保持一致)
DATA_BASE_DIR="../data_generation/dataset"

# 微调结果目录 (模型读取路径)
RESULT_BASE_DIR="./finetune_results"

# 评估日志输出目录
EVAL_LOG_DIR="./eval_logs"

# 词表与配置 (与训练保持一致)
VOCAB_PATH="../models/encryptd_vocab.txt"
CONFIG_PATH="../models/bert/base_config.json"

# ================= 显卡设置 =================
export CUDA_VISIBLE_DEVICES=0  # 评估通常只需单卡

# 确保日志目录存在
mkdir -p "$EVAL_LOG_DIR"

echo "开始批量评估 TrafficFormer 模型..."
echo "---------------------------------------"

# ================= 定义评估函数 =================
run_eval() {
    local dataset_name=$1
    local sub_dir_name=$2  # 某些数据集目录名可能与显示名不同，这里保持灵活

    # 1. 定义路径
    local data_dir="$DATA_BASE_DIR/$sub_dir_name"
    local test_path="$data_dir/test_dataset.tsv"
    
    # 模型路径：从微调结果目录读取 finetuned_model.bin
    # local model_path="$RESULT_BASE_DIR/$sub_dir_name/finetuned_model.bin"
    local model_path="$RESULT_BASE_DIR/${sub_dir_name}_freeze/finetuned_model.bin"

    
    local log_file="$EVAL_LOG_DIR/${dataset_name}_eval.log"

    echo "========== 正在评估数据集: $dataset_name =========="
    echo "模型路径: $model_path"
    echo "测试数据: $test_path"

    # 2. 检查文件是否存在
    if [ ! -f "$model_path" ]; then
        echo "错误: 找不到模型文件 $model_path ，跳过。" | tee -a "$log_file"
        echo ""
        return
    fi

    if [ ! -f "$test_path" ]; then
        echo "错误: 找不到测试数据 $test_path ，跳过。" | tee -a "$log_file"
        echo ""
        return
    fi

    # 3. 执行 Python 脚本 (Evaluation Mode)
    # 注意：
    # - 不传递 --train_path (触发纯评估模式)
    # - 将 test_path 传给 --dev_path (因为纯评估模式默认读取 dev_path)
    # - 使用 finetuned_model.bin 作为 pretrained_model_path 进行加载
    echo $model_path
    python run_classifier.py \
        --vocab_path "$VOCAB_PATH" \
        --config_path "$CONFIG_PATH" \
        --pretrained_model_path "$model_path" \
        --test_path "$test_path" \
        --batch_size 128 \
        --embedding word_pos_seg \
        --encoder transformer \
        --mask fully_visible \
        --seq_length 512 \
        --seed 42
}

# ================= 开始循环评估 =================
# 请根据实际微调过的数据集取消注释
# run_eval "CSTNET" "CSTNET"
# run_eval "AES-128-GCM-Dataset" "AES-128-GCM-Dataset"
# run_eval "AES-256-GCM-Dataset" "AES-256-GCM-Dataset"
# run_eval "chacha20-poly1305-Dataset" "chacha20-poly1305-Dataset"
# run_eval "mix-Dataset" "mix-Dataset"
run_eval "DataCon2021_part1" "DataCon2021_part1"

echo "---------------------------------------"
echo "所有评估任务已完成！日志位于 $EVAL_LOG_DIR"