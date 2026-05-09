# 遇到错误立即停止
set -e

# 2. GPU 自动检测
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SEED=42  # 在这里统一设置你想要的种子
export PYTHONHASHSEED=$SEED
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# 如果没有手动指定 CUDA_VISIBLE_DEVICES，则默认使用 1 张卡
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export N_GPUS=1
else
    # 统计逗号分隔的 GPU 数量
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    export N_GPUS=${#GPU_ARRAY[@]}
fi

echo "开始执行 NetMamba 批量微调 (Freeze Encoder) 脚本..."
echo "检测到可用 GPU 数量: $N_GPUS"
echo "---------------------------------------"

# 3. 定义微调执行函数
run_finetune_netmamba_freeze() {
    local index=$1
    local name=$2
    local data_path=$3
    local output_base_dir=$4

    echo "========== [$index/7] (Freeze) 正在处理数据集: $name =========="

    # --- 自动计算类别数 ---
    # 假设 NetMamba 数据结构与 YaTC 一致: data_path/train/class_folders
    local train_dir="$data_path/train"
    
    if [ ! -d "$train_dir" ]; then
        echo "错误: 找不到训练目录 $train_dir ，跳过此任务。"
        return
    fi

    # 统计子目录数量作为类别数
    local nb_classes=$(ls -d "$train_dir"/*/ | wc -l)
    
    if [ "$nb_classes" -eq 0 ]; then
        echo "警告: 在 $train_dir 中未检测到类别，请检查路径！"
        return
    fi

    echo "检测到类别总数: $nb_classes"
    
    # 定义输出和日志路径，增加了 _freeze 后缀区分
    local output_dir="${output_base_dir}/${name}_freeze"
    local log_dir="${output_dir}/logs"

    # 清理并创建目录
    if [ -d "$output_dir" ]; then
        echo "清理旧的输出目录..."
        rm -rf "$output_dir"
    fi
    mkdir -p "$log_dir"

    # 执行训练 (使用 fine-tune-freeze.py)
    # 注意：Base LR 可能需要比全量微调大一点，这里设为 2e-3 或更高，视情况调整
    torchrun --nproc_per_node=$N_GPUS src/fine-tune-freeze.py \
        --blr 2e-3 \
        --epochs 20 \
        --nb_classes "$nb_classes" \
        --finetune ./pre-train.pth \
        --data_path "$data_path" \
        --output_dir "$output_dir" \
        --log_dir "$log_dir" \
        --model net_mamba_classifier \
        --no_amp \
        --seed "$SEED"
    
    echo "Successfully finished: $name (Freeze)"
    echo "---------------------------------------"
}

# 4. 循环执行任务
DATA_ROOT="../YaTC/1-data_processing/outputs"
OUTPUT_ROOT="./output_freeze"  # 输出到不同的目录

# run_finetune_netmamba_freeze "1" "vpn_service" "$DATA_ROOT/finetune_data_vpn_service" "$OUTPUT_ROOT"
# run_finetune_netmamba_freeze "7" "cstnet" "$DATA_ROOT/finetune_data_cstnet" "$OUTPUT_ROOT"
run_finetune_netmamba_freeze "8" "DataCon2021_part1" "$DATA_ROOT/finetune_data_DataCon2021_part1" "$OUTPUT_ROOT"
run_finetune_netmamba_freeze "6" "aes_128_gcm" "$DATA_ROOT/finetune_data_aes_128_gcm" "$OUTPUT_ROOT"
run_finetune_netmamba_freeze "6" "AES-256-GCM" "$DATA_ROOT/finetune_data_AES-256-GCM-Dataset" "$OUTPUT_ROOT"
run_finetune_netmamba_freeze "6" "chacha20-poly1305" "$DATA_ROOT/finetune_data_chacha20-poly1305-Dataset" "$OUTPUT_ROOT"
run_finetune_netmamba_freeze "6" "mix" "$DATA_ROOT/finetune_data_mix-Dataset" "$OUTPUT_ROOT"