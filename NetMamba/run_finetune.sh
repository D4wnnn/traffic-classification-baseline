
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

echo "开始执行 NetMamba 批量微调脚本..."
echo "检测到可用 GPU 数量: $N_GPUS"
echo "---------------------------------------"

# 3. 定义微调执行函数
run_finetune_netmamba() {
    local index=$1
    local name=$2
    local data_path=$3
    local output_base_dir=$4

    echo "========== [$index/7] 正在处理数据集: $name =========="

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
    
    # 定义输出和日志路径
    local output_dir="${output_base_dir}/${name}"
    local log_dir="${output_dir}/logs"

    # 清理并创建目录
    if [ -d "$output_dir" ]; then
        echo "清理旧的输出目录..."
        rm -rf "$output_dir"
    fi
    mkdir -p "$log_dir"

    # 执行训练 (保留 NetMamba 特有的参数: --model, --no_amp, --log_dir)
    # 注意：NetMamba 脚本在 src/ 目录下
    torchrun --nproc_per_node=$N_GPUS src/fine-tune.py \
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
    
    echo "Successfully finished: $name"
    echo "---------------------------------------"
}

# 4. 循环执行任务
# 基础数据路径（根据你的目录结构，NetMamba 脚本可能在 baselines/NetMamba/，
# 所以数据路径需要根据实际位置调整，这里建议使用绝对路径或相对路径的统一起点）

DATA_ROOT="../YaTC/1-data_processing/outputs"
OUTPUT_ROOT="./output"

# 1. VPN-service
# run_finetune_netmamba "1" "vpn_service" "$DATA_ROOT/finetune_data_vpn_service" "$OUTPUT_ROOT"

# run_finetune_netmamba "1" "vpn_app" "$DATA_ROOT/finetune_data_vpn_app" "$OUTPUT_ROOT"

# # 3. USTC-TFC
# run_finetune_netmamba "3" "ustc_tfc" "$DATA_ROOT/finetune_data_ustc_tfc" "$OUTPUT_ROOT"

# 4. Cross-Platform-Android
# run_finetune_netmamba "4" "cross_android" "$DATA_ROOT/finetune_data_cross_android" "$OUTPUT_ROOT"

# 5. Cross-Platform-IOS
# run_finetune_netmamba "5" "cross_ios" "$DATA_ROOT/finetune_data_cross_ios" "$OUTPUT_ROOT"

# # 6. Browser-Dataset
# run_finetune_netmamba "6" "browser" "$DATA_ROOT/finetune_data_browser" "$OUTPUT_ROOT"

# # 7. CSTNET
run_finetune_netmamba "7" "cstnet" "$DATA_ROOT/finetune_data_cstnet" "$OUTPUT_ROOT"

# run_finetune_netmamba "8" "DataCon2021_part1" "$DATA_ROOT/finetune_data_DataCon2021_part1" "$OUTPUT_ROOT"

# run_finetune_netmamba "6" "aes_128_gcm" "$DATA_ROOT/finetune_data_aes_128_gcm" "$OUTPUT_ROOT"
# run_finetune_netmamba "6" "AES-256-GCM" "$DATA_ROOT/finetune_data_AES-256-GCM-Dataset" "$OUTPUT_ROOT"
# run_finetune_netmamba "6" "chacha20-poly1305" "$DATA_ROOT/finetune_data_chacha20-poly1305-Dataset" "$OUTPUT_ROOT"
# run_finetune_netmamba "6" "mix" "$DATA_ROOT/finetune_data_mix-Dataset" "$OUTPUT_ROOT"

# echo "所有 NetMamba 微调任务已全部完成！"