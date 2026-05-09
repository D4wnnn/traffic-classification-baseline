#!/bin/bash

echo "开始批量生成 Baseline 微调数据集..."
echo "---------------------------------------"

# 定义处理函数，减少重复代码
process_dataset() {
    local index=$1
    local name=$2
    local input_path=$3
    local output_path=$4

    echo "========== [$index/7] 处理 $name =========="
    
    # 清理旧目录
    if [ -d "$output_path" ]; then
        echo "清理旧目录: $output_path"
        rm -rf "$output_path"
    fi

    # 执行处理程序
    python get_finetune_dataset.py \
        --dataset_path "$input_path" \
        --output_path "$output_path"
    
    echo "完成: $name"
    echo ""
}

# --- 开始处理各个数据集 ---

# 1. VPN-service
# process_dataset "1" "VPN-service" \
#     "/raid/lc/datasets/processed_datasets/finetune/ISCX-VPN-NonVPN-2016/VPN-service-processed" \
#     "./outputs/finetune_data_vpn_service"
# process_dataset "1" "Version" \
#     "/raid/lc/datasets/processed_datasets/finetune/Version-processed" \
#     "./outputs/finetune_data_version"

process_dataset "1" "Version" \
    "/mnt/8T/lc/datasets/processed_datasets/finetune/CICIDS2017-processed" \
    "./outputs/finetune_data_cicids2017"

# process_dataset "1" "VPN-app" \
#     "/raid/lc/datasets/processed_datasets/finetune/ISCX-VPN-NonVPN-2016/VPN-app-processed" \
#     "./outputs/finetune_data_vpn_app"

# 3. USTC-TFC
# process_dataset "3" "USTC-TFC" \
#     "/raid/lc/datasets/processed_datasets/finetune/USTC-TFC2016-processed" \
#     "./outputs/finetune_data_ustc_tfc"

# 3. USTC-TFC
# process_dataset "3" "USTC-TFC" \
#     "/raid/lc/datasets/processed_datasets/finetune/USTC-TFC2016-processed_sampled" \
#     "./outputs/finetune_data_ustc_tfc_sampled"

# # 4. Cross-Platform-Android
# process_dataset "4" "Cross-Platform-Android" \
#     "/raid/lc/datasets/processed_datasets/finetune/Cross-Platform-android-processed" \
#     "./outputs/finetune_data_cross_android"

# # 5. Cross-Platform-IOS
# process_dataset "5" "Cross-Platform-IOS" \
#     "/raid/lc/datasets/processed_datasets/finetune/Cross-Platform-ios-processed" \
#     "./outputs/finetune_data_cross_ios"

# # 6. Browser-Dataset
# process_dataset "6" "Browser-Dataset" \
#     "/raid/lc/datasets/processed_datasets/finetune/Browser-processed" \
#     "./outputs/finetune_data_browser"

# # 7. CSTNET
# process_dataset "7" "CSTNET" \
#     "/raid/lc/datasets/initial_datasets/finetune/CSTNET" \
#     "./outputs/finetune_data_cstnet"

# # DataCon2020 数据集
# process_dataset "8" "DataCon2020" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2020-processed" \
#     "./outputs/finetune_data_DataCon2020"

# # DataCon2021_part1 数据集
# process_dataset "8" "DataCon2021_part1" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2021-processed/part1/" \
#     "./outputs/finetune_data_DataCon2021_part1"

# # # DataCon2021_part2 数据集
# process_dataset "8" "DataCon2021_part2" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2021-processed/part2/" \
#     "./outputs/finetune_data_DataCon2021_part2"

# process_dataset "6" "aes_128_gcm-Dataset" \
#     "/raid/lc/datasets/processed_datasets/finetune/AES-128-GCM-processed" \
#     "./outputs/finetune_data_aes_128_gcm"

# process_dataset "6" "AES-256-GCM-Dataset" \
#     "/raid/lc/datasets/processed_datasets/finetune/AES-256-GCM-processed" \
#     "./outputs/finetune_data_AES-256-GCM-Dataset"
# process_dataset "6" "chacha20-poly1305-Dataset" \
#     "/raid/lc/datasets/processed_datasets/finetune/chacha20-poly1305-processed" \
#     "./outputs/finetune_data_chacha20-poly1305-Dataset"

# process_dataset "6" "mix-Dataset" \
#     "/raid/lc/datasets/processed_datasets/finetune/mix-processed" \
#     "./outputs/finetune_data_mix-Dataset"









echo "---------------------------------------"
echo "恭喜！所有 Baseline 数据集预处理已完成！"