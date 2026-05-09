#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=1
# 2. GPU 自动检测
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export N_GPUS=1
else
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    export N_GPUS=${#GPU_ARRAY[@]}
fi

echo "开始批量进行 YaTC 自动分类计数微调..."
echo "---------------------------------------"

# 3. 定义微调执行函数
run_finetune() {
    local index=$1
    local name=$2
    local data_path=$3
    local output_dir=$4

    echo "========== [$index/7] 处理数据集: $name =========="

    # --- 核心：自动计算类别数 ---
    # 假设结构为 data_path/train/类别1, data_path/train/类别2...
    local train_dir="$data_path/train"
    
    if [ ! -d "$train_dir" ]; then
        echo "错误: 找不到训练目录 $train_dir ，跳过此任务。"
        return
    fi

    # 统计 train 目录下的子目录数量
    local nb_classes=$(ls -d "$train_dir"/*/ | wc -l)
    
    if [ "$nb_classes" -eq 0 ]; then
        echo "警告: 在 $train_dir 中未检测到类别目录，请检查数据格式！"
        return
    fi

    echo "检测到类别总数 (nb_classes): $nb_classes"
    echo "输出目录: $output_dir"

    # 清理旧输出目录
    if [ -d "$output_dir" ]; then
        rm -rf "$output_dir"
    fi
    mkdir -p "$output_dir"

    # 执行训练
    torchrun --nproc_per_node=$N_GPUS fine-tune.py \
        --blr 2e-3 \
        --epochs 20 \
        --finetune ./YaTC_pretrained_model.pth \
        --data_path "$data_path" \
        --output_dir "$output_dir" \
        --nb_classes "$nb_classes" \
    
    echo "完成 $name 的训练。"
    echo ""
}

# --- 4. 开始执行各数据集微调 ---
# 现在只需要传入：序号、名称、数据路径、输出路径

# 1. VPN-service
# run_finetune "1" "VPN-service" \
#     "./1-data_processing/outputs/finetune_data_vpn_service" \
#     "./output/vpn_service"


# run_finetune "1" "VPN-app" \
#     "./1-data_processing/outputs/finetune_data_vpn_app" \
#     "./output/vpn_app"

# # 3. USTC-TFC
# run_finetune "3" "USTC-TFC" \
#     "./1-data_processing/outputs/finetune_data_ustc_tfc" \
#     "./output/ustc_tfc"

# run_finetune "3" "USTC-TFC" \
#     "./1-data_processing/outputs/finetune_data_ustc_tfc_sampled" \
#     "./output/ustc_tfc_sampled"


# # 4. Cross-Platform-Android
# run_finetune "4" "Cross-Platform-Android" \
#     "./1-data_processing/outputs/finetune_data_cross_android" \
#     "./output/cross_android"

# # 5. Cross-Platform-IOS
# run_finetune "5" "Cross-Platform-IOS" \
#     "./1-data_processing/outputs/finetune_data_cross_ios" \
#     "./output/cross_ios"

# # 6. Browser-Dataset
# run_finetune "6" "Browser-Dataset" \
#     "./1-data_processing/outputs/finetune_data_browser" \
#     "./output/browser"

# # 7. CSTNET
# run_finetune "7" "CSTNET" \
#     "./1-data_processing/outputs/finetune_data_cstnet" \
#     "./output/cstnet"

# # DataCon2020 数据集
# run_finetune "8" "DataCon2020" \
#     "./1-data_processing/outputs/finetune_data_DataCon2020" \
#     "./output/datacon2020"

# # # DataCon2021_part1 数据集
# run_finetune "9" "DataCon2021_part1" \
#     "./1-data_processing/outputs/finetune_data_DataCon2021_part1" \
#     "./output/datacon2021_part1"

# # # # DataCon2021_part2 数据集
# run_finetune "10" "DataCon2021_part2" \
#     "./1-data_processing/outputs/finetune_data_DataCon2021_part2" \
#     "./output/datacon2021_part2"


# run_finetune "7" "aes_128_gcm" \
#     "./1-data_processing/outputs/finetune_data_aes_128_gcm" \
#     "./output/aes_128_gcm"

# run_finetune "7" "aes_256_gcm" \
#     "./1-data_processing/outputs/finetune_data_AES-256-GCM-Dataset" \
#     "./output/aes_256_gcm"

# run_finetune "7" "chacha20_poly1305" \
#     "./1-data_processing/outputs/finetune_data_chacha20-poly1305-Dataset" \
#     "./output/chacha20_poly1305"
# run_finetune "7" "mix_Dataset" \
#     "./1-data_processing/outputs/finetune_data_mix-Dataset" \
#     "./output/mix_Dataset"

# run_finetune "7" "Version" \
#     "./1-data_processing/outputs/finetune_data_version" \
#     "./output/version"

run_finetune "7" "CICIDS2017" \
    "./1-data_processing/outputs/finetune_data_cicids2017" \
    "./output/cicids2017"


echo "---------------------------------------"
echo "所有任务已处理完毕！"