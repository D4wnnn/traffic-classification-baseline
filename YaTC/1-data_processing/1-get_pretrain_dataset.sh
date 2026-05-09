source ~/miniconda3/etc/profile.d/conda.sh
conda activate yatc
# python get_pretrain_dataset.py \
# --dataset_path /storage/lc_data/processed_datasets/ISCX-VPN-NonVPN-2016/VPN-processed /storage/lc_data/initial_datasets/CSTNET \
# --output_path ./pretrain_data

python get_pretrain_dataset.py \
--dataset_path /storage/lc_data/processed_datasets/pretrain/MAWI-processed /storage/lc_data/processed_datasets/pretrain/CIC_IOT_Dataset2022-processed \
--output_path /storage/lc_data/processed_datasets/pretrain/project_dataset/YaTC/pretrain_data2

