#!/bin/bash

# ================= 配置输出根目录 =================
# 将所有 ET-BERT 的数据放在指定位置的 ET-BERT 子目录下
BASE_OUTPUT_DIR="./processed_data"

# 确保父目录存在
if [ ! -d "$BASE_OUTPUT_DIR" ]; then
    echo "创建输出根目录: $BASE_OUTPUT_DIR"
    mkdir -p "$BASE_OUTPUT_DIR"
fi
# =================================================

echo "开始批量生成 ET-BERT Baseline 微调数据集..."
echo "输出位置: $BASE_OUTPUT_DIR"
echo "---------------------------------------"

# 定义处理函数
process_dataset() {
    local index=$1
    local name=$2
    local input_path=$3
    local sub_dir_name=$4  # 只需要传入子目录名

    # 拼接完整的输出路径
    local output_path="$BASE_OUTPUT_DIR/$sub_dir_name"

    echo "========== [$index/7] 处理 $name =========="
    
    # 清理旧目录 (如果存在)
    if [ -d "$output_path" ]; then
        echo "清理旧目录: $output_path"
        rm -rf "$output_path"
    fi
    
    # 创建目录
    mkdir -p "$output_path"

    # 执行处理程序
    # 确保你的 Python 脚本已经按上一步建议修改为支持 argparse
    python generate_tsv_from_split.py \
        --dataset_path "$input_path" \
        --output_path "$output_path" \
        --num_workers 40  
    
    echo "完成: $name -> $output_path"
    echo ""
}

# --- 开始处理各个数据集 ---

# # 1. VPN-service
# process_dataset "1" "VPN-service" \
#     "/raid/lc/datasets/processed_datasets/finetune/ISCX-VPN-NonVPN-2016/VPN-service-processed" \
#     "VPN-service"

# # 3. USTC-TFC
# process_dataset "3" "USTC-TFC" \
#     "/raid/lc/datasets/processed_datasets/finetune/USTC-TFC2016-processed" \
#     "USTC-TFC"

# # 4. Cross-Platform-Android
# process_dataset "4" "Cross-Platform-Android" \
#     "/raid/lc/datasets/processed_datasets/finetune/Cross-Platform-android-processed" \
#     "Cross-Platform-Android"

# # 5. Cross-Platform-IOS
process_dataset "5" "Cross-Platform-IOS" \
    "/raid/lc/datasets/processed_datasets/finetune/Cross-Platform-ios-processed" \
    "Cross-Platform-IOS"

# # 6. Browser-Dataset
# process_dataset "6" "Browser-Dataset" \
#     "/raid/lc/datasets/processed_datasets/finetune/Browser-processed" \
#     "Browser-Dataset"

# # 7. CSTNET
# process_dataset "7" "CSTNET" \
#     "/raid/lc/datasets/initial_datasets/finetune/CSTNET" \
#     "CSTNET"

# DataCon2020 数据集
# process_dataset "8" "DataCon2020" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2020-processed" \
#     "DataCon2020"

# # DataCon2021_part1 数据集
# process_dataset "8" "DataCon2021_part1" \
#     "/raid/lc/datasets/initial_datasets/finetune/DataCon2021/part1/organized_real_data_processed" \
#     "DataCon2021_part1"


# # DataCon2021_part2 数据集
# process_dataset "8" "DataCon2021_part2" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2021-processed/part2" \
#     "DataCon2021_part2"

echo ""


echo "---------------------------------------"
echo "恭喜！所有 ET-BERT Baseline 数据集预处理已完成！"
echo "数据已存储在: $BASE_OUTPUT_DIR"