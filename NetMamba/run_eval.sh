#!/bin/bash

# 遇到错误立即停止
set -e

# 1. GPU 自动检测与环境设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SEED=42
export PYTHONHASHSEED=$SEED
# 保持与训练一致的确定性设置
export CUBLAS_WORKSPACE_CONFIG=:4096:8 

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export N_GPUS=1
else
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    export N_GPUS=${#GPU_ARRAY[@]}
fi

echo "开始执行 NetMamba 批量评估脚本..."
echo "检测到可用 GPU 数量: $N_GPUS"
echo "---------------------------------------"

# 2. 定义评估执行函数
run_eval_netmamba() {
    local index=$1
    local name=$2
    local data_path=$3
    local output_base_dir=$4

    echo "========== [$index] 正在评估数据集: $name =========="

    # --- 自动计算类别数 (用于初始化模型结构) ---
    local train_dir="$data_path/train"
    if [ ! -d "$train_dir" ]; then
        echo "错误: 找不到训练目录 $train_dir (用于确定类别数)，跳过。"
        return
    fi
    local nb_classes=$(ls -d "$train_dir"/*/ | wc -l)
    if [ "$nb_classes" -eq 0 ]; then
        echo "警告: 未检测到类别，跳过！"
        return
    fi
    echo "检测到类别总数: $nb_classes"

    # --- 路径设置 ---
    # 假设训练结果保存在 output_base_dir/name 下
    local project_dir="${output_base_dir}/${name}"
    # 默认加载最佳权重
    local checkpoint_path="${project_dir}/checkpoint-best.pth"
    
    if [ ! -f "$checkpoint_path" ]; then
        echo "错误: 找不到权重文件: $checkpoint_path"
        echo "请检查该任务是否已训练完成。"
        return
    fi

    echo "加载权重: $checkpoint_path"

    # --- 执行评估 ---
    # 使用 torchrun 进行分布式评估 (eval.py 支持 dist_eval)
    # 注意：eval.py 中默认 model 是 flow_mamba_tiny，这里必须指定为 net_mamba_classifier 以匹配训练
    torchrun --nproc_per_node=$N_GPUS src/eval.py \
        --model net_mamba_classifier \
        --nb_classes "$nb_classes" \
        --data_path "$data_path" \
        --resume "$checkpoint_path" \
        --output_dir "$project_dir" \
        --batch_size 128 \
        --num_workers 8 \
        --seed "$SEED" \
        --pin_mem

    # 如果需要进行速度测试，可以取消下面这行的注释，并注释掉上面的 eval
    # torchrun --nproc_per_node=1 src/eval.py --model net_mamba_classifier --nb_classes "$nb_classes" --data_path "$data_path" --speed_test --batch_size 64

    echo "Successfully evaluated: $name"
    echo "结果已保存至: $project_dir/test_stats.json"
    echo "---------------------------------------"
}

# 3. 数据集与路径配置
# 请确保这里的 DATA_ROOT 和 OUTPUT_ROOT 与 run_finetune.sh 中保持一致
DATA_ROOT="../YaTC/1-data_processing/outputs"
OUTPUT_ROOT="./output"

# 4. 执行任务 (取消注释需要评估的任务)
run_eval_netmamba "7" "cstnet" "$DATA_ROOT/finetune_data_cstnet" "$OUTPUT_ROOT"
# run_eval_netmamba "8" "DataCon2021_part1" "$DATA_ROOT/finetune_data_DataCon2021_part1" "$OUTPUT_ROOT"
# run_eval_netmamba "6" "aes_128_gcm" "$DATA_ROOT/finetune_data_aes_128_gcm" "$OUTPUT_ROOT"
# run_eval_netmamba "6" "AES-256-GCM" "$DATA_ROOT/finetune_data_AES-256-GCM-Dataset" "$OUTPUT_ROOT"
# run_eval_netmamba "6" "chacha20-poly1305" "$DATA_ROOT/finetune_data_chacha20-poly1305-Dataset" "$OUTPUT_ROOT"
# run_eval_netmamba "6" "mix" "$DATA_ROOT/finetune_data_mix-Dataset" "$OUTPUT_ROOT"

echo "所有评估任务执行完毕。"