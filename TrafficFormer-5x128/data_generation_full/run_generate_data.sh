#!/bin/bash

# ================= 配置输出根目录 =================
# 根据 TrafficFormer 的习惯，通常放在 ./dataset 目录下
BASE_OUTPUT_DIR="./dataset"

# 并行进程数 (根据你的服务器配置调整，TrafficFormer 示例中是 100，这里设为 60 以防 OOM)
NUM_WORKERS=60

# 确保父目录存在
if [ ! -d "$BASE_OUTPUT_DIR" ]; then
    echo "创建输出根目录: $BASE_OUTPUT_DIR"
    mkdir -p "$BASE_OUTPUT_DIR"
fi
# =================================================

echo "开始批量生成 TrafficFormer 微调数据集..."
echo "输出位置: $BASE_OUTPUT_DIR"
echo "进程数: $NUM_WORKERS"
echo "---------------------------------------"

# 定义处理函数
process_dataset() {
    local index=$1
    local name=$2
    local input_path=$3
    local sub_dir_name=$4  # 子目录名

    # 拼接完整的输出路径
    local output_path="$BASE_OUTPUT_DIR/$sub_dir_name"

    echo "========== [$index/9] 处理 $name =========="
    
    # 清理旧目录 (如果存在)
    if [ -d "$output_path" ]; then
        echo "清理旧目录: $output_path"
        rm -rf "$output_path"
    fi
    
    # 这里的 mkdir 不需要手动执行，因为 TrafficFormer 脚本通常会自动创建，
    # 但为了保险起见保留，或依赖 generate_finetuning_data.py 的逻辑。

    # 执行 TrafficFormer 数据生成脚本
    # 参数说明：
    # --payload_length 240: 截取长度
    # --payload_packet 5:   截取包数
    # --start_index 76:     跳过头部字节（根据你提供的示例保留为 76）
    python generate_finetuning_data.py \
        --data_dir "$input_path" \
        --output_dir "$output_path" \
        --payload_length 128 \
        --payload_packet 5 \
        --start_index 76 \
        --num_workers $NUM_WORKERS \
        --splits train val test
    
    echo "完成: $name -> $output_path"
    echo ""
}

# --- 开始处理各个数据集 (路径已对齐 ET-BERT 脚本) ---

# 1. VPN-service
# process_dataset "1" "VPN-service" \
#     "/raid/lc/datasets/processed_datasets/finetune/ISCX-VPN-NonVPN-2016/VPN-service-processed" \
#     "VPN-service"

# # 2. USTC-TFC
# process_dataset "2" "USTC-TFC" \
#     "/raid/lc/datasets/processed_datasets/finetune/USTC-TFC2016-processed" \
#     "USTC-TFC"

# # 3. Cross-Platform-Android
# process_dataset "3" "Cross-Platform-Android" \
#     "/raid/lc/datasets/processed_datasets/finetune/Cross-Platform-android-processed" \
#     "Cross-Platform-Android"

# # 4. Cross-Platform-IOS
process_dataset "4" "Cross-Platform-IOS" \
    "/raid/lc/datasets/processed_datasets/finetune/Cross-Platform-ios-processed" \
    "Cross-Platform-IOS"

# # 5. Browser-Dataset
# process_dataset "5" "Browser-Dataset" \
#     "/raid/lc/datasets/processed_datasets/finetune/Browser-processed" \
#     "Browser-Dataset"

# 6. CSTNET
# process_dataset "6" "CSTNET" \
#     "/raid/lc/datasets/initial_datasets/finetune/CSTNET" \
#     "CSTNET"

# # 7. DataCon2020
# process_dataset "7" "DataCon2020" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2020-processed" \
#     "DataCon2020"

# # 8. DataCon2021_part1
# process_dataset "8" "DataCon2021_part1" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2021-processed/part1" \
#     "DataCon2021_part1"

# # 9. DataCon2021_part2
# process_dataset "9" "DataCon2021_part2" \
#     "/raid/lc/datasets/processed_datasets/finetune/DataCon2021-processed/part2" \
#     "DataCon2021_part2"

echo ""
echo "---------------------------------------"
echo "恭喜！所有 TrafficFormer 数据集预处理已完成！"
echo "数据已存储在: $BASE_OUTPUT_DIR"


















# #!/bin/bash


# # 配置参数
# DATA_DIR="/storage/lc_data/processed_datasets/finetune/ISCX-VPN-NonVPN-2016/VPN-app-processed" 
# OUTPUT_DIR="./dataset/ISCX-VPN2016" 
# NUM_WORKERS=100  # 进程数
# rm -rf "$OUTPUT_DIR"
# # 运行数据生成
# python generate_finetuning_data.py \
#     --data_dir $DATA_DIR \
#     --output_dir $OUTPUT_DIR \
#     --payload_length 240 \
#     --payload_packet 5 \
#     --start_index 76 \
#     --num_workers $NUM_WORKERS \
#     --splits train val test