#!/bin/bash

# 1. 初始化环境
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate yatc
set -e

# --- 环境检测 ---
# 注意：测试脚本通常不需要多卡并行（单卡推理即可），
# 但我们需要知道机器的上限，以便逻辑上与 finetune 脚本保持一致
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    MAX_AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
else
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    MAX_AVAILABLE_GPUS=${#GPU_ARRAY[@]}
fi

echo "当前环境最大可用 GPU 数量: $MAX_AVAILABLE_GPUS"
echo "即将开始批量测试：检查 1, 2, 4, 8 卡配置下的模型权重..."
echo "---------------------------------------"

# 2. 定义测试执行函数
# 参数: run_test_search "原始训练卡数" "序号" "名称" "数据路径" "输出根路径"
run_test_search() {
    local trained_gpus=$1  # 这个参数用于寻找文件夹，例如 '4' -> '4_gpus'
    local index=$2
    local name=$3
    local data_path=$4
    local base_output_dir=$5 

    # --- 关键：构建与 finetune_search 一致的路径 ---
    local model_dir="${base_output_dir}/${trained_gpus}_gpus/${name}"
    local checkpoint="$model_dir/checkpoint-best.pth"

    echo "========== [配置: ${trained_gpus}卡训练] [$index/7] 检查数据集: $name =========="

    # 1. 检查模型是否存在
    if [ ! -f "$checkpoint" ]; then
        echo "警告: 找不到模型文件 $checkpoint"
        echo "可能是该配置 (${trained_gpus}卡) 的训练尚未完成或被跳过。"
        echo ">>> 跳过测试。"
        echo ""
        return
    fi

    # 2. 自动计算类别数 (需存在 train 目录)
    local train_dir="$data_path/train"
    if [ ! -d "$train_dir" ]; then
        echo "错误: 找不到训练目录 $train_dir ，无法确定类别数，跳过。"
        return
    fi
    local nb_classes=$(ls -d "$train_dir"/*/ | wc -l)

    echo "检测到类别总数: $nb_classes"
    echo "加载权重路径: $checkpoint"
    echo "测试结果将保存至: $model_dir"

    # 3. 执行测试
    # 注意：这里使用 python 直接运行，通常使用单卡进行推理验证即可
    # final-test.py 会将结果保存到 checkpoint 所在的目录
    python final-test.py \
        --data_path "$data_path" \
        --checkpoint "$checkpoint" \
        --model TraFormer_YaTC \
        --nb_classes "$nb_classes" \
        --batch_size 64 \
        --device cuda \
        --num_workers 8
    
    echo "完成 $name (配置: ${trained_gpus}卡) 的测试。"
    echo ""
}

# --- 3. 外层循环：遍历卡数 1 -> 2 -> 4 -> 8 ---
# 这里遍历的是“训练时使用的卡数”，用于定位文件夹
for N_GPUS in 1 2 4 8; do
    
    # 如果机器根本没有这么多卡，说明对应的训练肯定没跑过（或者是在别的机器跑的）
    # 这里我们只做一个简单的提示，但依然尝试去检查文件夹，
    # 因为有可能你是把别人 8 卡跑完的权重拷贝到这台单卡机器上来测。
    if [ "$N_GPUS" -gt "$MAX_AVAILABLE_GPUS" ]; then
        echo "提示: 当前机器只有 $MAX_AVAILABLE_GPUS 卡，但我们将尝试寻找 $N_GPUS 卡的训练记录..."
    fi

    echo "######################################################"
    echo ">>> 正在搜索并测试: [由 $N_GPUS 张显卡训练出的模型]"
    echo "######################################################"

    # 定义统一的输出根目录 (必须与 finetune 脚本一致)
    OUTPUT_ROOT="./output"

    # 依次执行 7 个数据集
    # 路径必须与 4-finetune_search.sh 完全对应

    # 1. VPN-service
    run_test_search "$N_GPUS" "1" "VPN-service" \
        "./1-data_processing/finetune_data_vpn_service" "$OUTPUT_ROOT"

    # 2. VPN-app
    run_test_search "$N_GPUS" "2" "VPN-app" \
        "./1-data_processing/finetune_data_vpn_app" "$OUTPUT_ROOT"

    # 3. USTC-TFC
    run_test_search "$N_GPUS" "3" "USTC-TFC" \
        "./1-data_processing/finetune_data_ustc_tfc" "$OUTPUT_ROOT"

    # 4. Cross-Platform-Android
    run_test_search "$N_GPUS" "4" "Cross-Platform-Android" \
        "./1-data_processing/finetune_data_cross_android" "$OUTPUT_ROOT"

    # 5. Cross-Platform-IOS
    run_test_search "$N_GPUS" "5" "Cross-Platform-IOS" \
        "./1-data_processing/finetune_data_cross_ios" "$OUTPUT_ROOT"

    # 6. Browser-Dataset
    run_test_search "$N_GPUS" "6" "Browser-Dataset" \
        "./1-data_processing/finetune_data_browser" "$OUTPUT_ROOT"

    # 7. CSTNET
    run_test_search "$N_GPUS" "7" "CSTNET" \
        "./1-data_processing/finetune_data_cstnet" "$OUTPUT_ROOT"

done

echo "---------------------------------------"
echo "所有路径下的模型测试已结束！"
echo "请检查 output/X_gpus/数据集名称/ 下的 test_result.txt 和 confusion_matrix_labeled.csv"