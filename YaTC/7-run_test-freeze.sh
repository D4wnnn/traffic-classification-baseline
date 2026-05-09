set -e
export CUDA_VISIBLE_DEVICES=5
# 2. GPU 自动检测 (虽然测试通常单卡即可，但保留环境一致性)
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export N_GPUS=1
else
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    export N_GPUS=${#GPU_ARRAY[@]}
fi

echo "开始批量进行 YaTC 模型性能测试..."
echo "---------------------------------------"

# 3. 定义测试执行函数
run_test() {
    local index=$1
    local name=$2
    local data_path=$3
    local output_dir=$4  # 微调时定义的输出目录

    echo "========== [$index/7] 测试数据集: $name =========="

    # --- 核心：检查模型权重 ---
    local checkpoint="$output_dir/checkpoint-best.pth"
    if [ ! -f "$checkpoint" ]; then
        echo "错误: 找不到模型权重文件 $checkpoint ，跳过此任务。"
        return
    fi

    # --- 核心：自动计算类别数 ---
    # 保持与 finetune 一致，从 train 目录统计，确保类别索引对应
    local train_dir="$data_path/train"
    if [ ! -d "$train_dir" ]; then
        echo "错误: 找不到目录 $train_dir ，无法确定类别数，跳过。"
        return
    fi
    local nb_classes=$(ls -d "$train_dir"/*/ | wc -l)

    echo "检测到类别总数 (nb_classes): $nb_classes"
    echo "使用模型权重: $checkpoint"

    # 执行推理测试
    # 注意：这里使用 python 而非 torchrun，因为测试脚本通常不需要分布式
    python final-test.py \
        --data_path "$data_path" \
        --checkpoint "$checkpoint" \
        --model TraFormer_YaTC \
        --nb_classes "$nb_classes" \
        --batch_size 64
    
    echo "完成 $name 的测试。"
    echo ""
}

run_test "7" "CSTNET" \
    "./1-data_processing/outputs/finetune_data_cstnet" \
    "./output/cstnet_freeze"

run_test "9" "DataCon2021_part1" \
    "./1-data_processing/outputs/finetune_data_DataCon2021_part1" \
    "./output/datacon2021_part1_freeze"

run_test "7" "aes_128_gcm" \
    "./1-data_processing/outputs/finetune_data_aes_128_gcm" \
    "./output/aes_128_gcm_freeze"

run_test "7" "aes_256_gcm" \
    "./1-data_processing/outputs/finetune_data_AES-256-GCM-Dataset" \
    "./output/aes_256_gcm_freeze"

run_test "7" "chacha20_poly1305" \
    "./1-data_processing/outputs/finetune_data_chacha20-poly1305-Dataset" \
    "./output/chacha20_poly1305_freeze"

run_test "7" "mix_Dataset" \
    "./1-data_processing/outputs/finetune_data_mix-Dataset" \
    "./output/mix_Dataset_freeze"




echo "---------------------------------------"
echo "所有测试任务已完成！请查看上方输出结果。"