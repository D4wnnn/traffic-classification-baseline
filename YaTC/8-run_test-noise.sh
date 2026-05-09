#!/bin/bash

# 设置显卡
export CUDA_VISIBLE_DEVICES=0

# 你的模型权重路径
CHECKPOINT="./output/aes_128_gcm/checkpoint-best.pth"
# 你的数据路径
DATA_PATH="./1-data_processing/outputs/finetune_data_aes_128_gcm"
# 类别数 (根据你的数据自动或手动设置)
NB_CLASSES=41

# 输出文件
RESULT_FILE="robustness_results_yatc.jsonl"

echo "开始 YaTC 鲁棒性测试..."

# 1. Byte Masking 测试
for ratio in 0.0 0.1 0.2 0.3 0.4 0.5; do
    python eval_robustness.py \
        --data_path "$DATA_PATH" \
        --checkpoint "$CHECKPOINT" \
        --nb_classes $NB_CLASSES \
        --batch_size 64 \
        --perturb_type byte_mask \
        --noise_ratio $ratio \
        --save_result "$RESULT_FILE"
done

# # 2. Packet Drop 测试
# for ratio in 0.1 0.3 0.5; do
#     python eval_robustness.py \
#         --data_path "$DATA_PATH" \
#         --checkpoint "$CHECKPOINT" \
#         --nb_classes $NB_CLASSES \
#         --batch_size 64 \
#         --perturb_type packet_drop \
#         --noise_ratio $ratio \
#         --save_result "$RESULT_FILE"
# done

echo "所有测试完成。"