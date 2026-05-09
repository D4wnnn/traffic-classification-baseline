#!/bin/bash

# 1. 初始化环境
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate yatc
set -e

# --- 自动检测当前机器可用的最大 GPU 数量 ---
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # 如果没指定，默认统计 nvidia-smi 看到的卡数
    MAX_AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
else
    # 如果指定了 CUDA_VISIBLE_DEVICES，则统计指定的卡数
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    MAX_AVAILABLE_GPUS=${#GPU_ARRAY[@]}
fi

echo "当前环境最大可用 GPU 数量: $MAX_AVAILABLE_GPUS"
echo "即将开始批量实验：遍历 1, 2, 4, 8 卡配置..."
echo "---------------------------------------"

# 2. 定义微调执行函数 (稍微修改，接受 GPU 数量作为参数)
run_finetune() {
    local use_gpus=$1      # 新增参数：使用的 GPU 数量
    local index=$2
    local name=$3
    local data_path=$4
    local base_output_dir=$5 # 基础输出目录

    # 构建带 GPU 数量标识的子目录，例如 ./output/4_gpus/vpn_service
    local final_output_dir="${base_output_dir}/${use_gpus}_gpus/${name}"

    echo "========== [GPU数量: $use_gpus] [$index/7] 处理数据集: $name =========="

    local train_dir="$data_path/train"
    
    if [ ! -d "$train_dir" ]; then
        echo "错误: 找不到训练目录 $train_dir ，跳过此任务。"
        return
    fi

    local nb_classes=$(ls -d "$train_dir"/*/ | wc -l)
    
    if [ "$nb_classes" -eq 0 ]; then
        echo "警告: 在 $train_dir 中未检测到类别目录，请检查数据格式！"
        return
    fi

    echo "检测到类别总数: $nb_classes"
    echo "最终输出目录: $final_output_dir"

    # 清理旧输出目录
    if [ -d "$final_output_dir" ]; then
        rm -rf "$final_output_dir"
    fi
    mkdir -p "$final_output_dir"

    # --- 执行训练 ---
    # 这里 --nproc_per_node 使用传入的 use_gpus
    torchrun --nproc_per_node=$use_gpus fine-tune.py \
        --blr 2e-3 \
        --epochs 20 \
        --finetune ./YaTC_pretrained_model.pth \
        --data_path "$data_path" \
        --output_dir "$final_output_dir" \
        --nb_classes "$nb_classes"
    
    echo "完成 $name (使用 $use_gpus 卡) 的训练。"
    echo ""
}

# --- 3. 外层循环：遍历卡数 1 -> 2 -> 4 -> 8 ---
for N_GPUS in 1 2 4 8; do
    
    # 检查硬件限制：如果要求的卡数超过了机器实际拥有的卡数，则停止后续循环
    if [ "$N_GPUS" -gt "$MAX_AVAILABLE_GPUS" ]; then
        echo "######################################################"
        echo "跳过 $N_GPUS 卡测试 (原因: 机器只有 $MAX_AVAILABLE_GPUS 张卡)"
        echo "######################################################"
        break # 或者 continue，取决于你想不想要强行停止
    fi

    echo "######################################################"
    echo ">>> 开始执行配置: 使用 $N_GPUS 张显卡 (Batch Size 将自动倍增)"
    echo "######################################################"

    # 定义统一的输出根目录
    OUTPUT_ROOT="./output"

    # 依次执行 7 个数据集
    # 参数顺序: run_finetune "卡数" "序号" "名称" "数据路径" "输出根路径"

    # 1. VPN-service
    run_finetune "$N_GPUS" "1" "VPN-service" \
        "./1-data_processing/finetune_data_vpn_service" "$OUTPUT_ROOT"

    # 2. VPN-app
    run_finetune "$N_GPUS" "2" "VPN-app" \
        "./1-data_processing/finetune_data_vpn_app" "$OUTPUT_ROOT"

    # 3. USTC-TFC
    run_finetune "$N_GPUS" "3" "USTC-TFC" \
        "./1-data_processing/finetune_data_ustc_tfc" "$OUTPUT_ROOT"

    # 4. Cross-Platform-Android
    run_finetune "$N_GPUS" "4" "Cross-Platform-Android" \
        "./1-data_processing/finetune_data_cross_android" "$OUTPUT_ROOT"

    # 5. Cross-Platform-IOS
    run_finetune "$N_GPUS" "5" "Cross-Platform-IOS" \
        "./1-data_processing/finetune_data_cross_ios" "$OUTPUT_ROOT"

    # 6. Browser-Dataset
    run_finetune "$N_GPUS" "6" "Browser-Dataset" \
        "./1-data_processing/finetune_data_browser" "$OUTPUT_ROOT"

    # 7. CSTNET
    run_finetune "$N_GPUS" "7" "CSTNET" \
        "./1-data_processing/finetune_data_cstnet" "$OUTPUT_ROOT"

done

echo "---------------------------------------"
echo "所有显卡配置的所有任务已处理完毕！"