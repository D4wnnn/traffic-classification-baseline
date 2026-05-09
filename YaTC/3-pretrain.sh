source ~/miniconda3/etc/profile.d/conda.sh
conda activate yatc

# python pre-train.py --batch_size 128 \
# --blr 1e-3 \
# --steps 20000 \
# --mask_ratio 0.9 \
# --data_path /home/lc/test-idea/baselines/YaTC/1-data_processing/pretrain_data \
# --save_steps_freq 100
# python pre-train.py --batch_size 1 \
# --blr 1e-3 \
# --steps 2 \
# --mask_ratio 0.9 \
# --data_path /home/lc/test-idea/baselines/YaTC/1-data_processing/pretrain_data \
# --save_steps_freq 100


export N_GPUS=2
CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=$N_GPUS pre-train.py \
--batch_size 512 \
--blr 1e-3 \
--steps 150000 \
--mask_ratio 0.9 \
--data_path /storage/lc_data/processed_datasets/pretrain/project_dataset/YaTC/pretrain_data2 \
--save_steps_freq 500 \
--output_dir /storage/lc_data/processed_datasets/pretrain/project_dataset/YaTC/pretraind_model \
--resume /storage/lc_data/processed_datasets/pretrain/project_dataset/YaTC/pretraind_model/checkpoint-step99500.pth