import os
import shutil
import json
from pathlib import Path
from tqdm import tqdm  # 如果没有安装 tqdm，可以去掉这行和相关的进度条代码，或者运行 pip install tqdm

# ================= 配置区域 =================
# 你的原始数据集根目录
# 结构应该是: SOURCE_DIR -> [train, val, test] -> [label_names] -> [*.pcap]
SOURCE_DATASET_DIR = "/raid/lc/datasets/processed_datasets/finetune/AES-128-GCM-processed" 

# 输出的目标目录
TARGET_DIR = "./netfound_ready_dataset"
# ===========================================

def get_all_labels(source_root):
    """扫描所有划分（train/val/test），获取所有唯一的标签名称并排序"""
    labels = set()
    splits = ['train', 'val', 'test']
    
    print("正在扫描标签...")
    for split in splits:
        split_path = Path(source_root) / split
        if not split_path.exists():
            print(f"警告: 目录 {split_path} 不存在，跳过。")
            continue
            
        # 获取该 split 下的所有子文件夹名作为 label
        for item in split_path.iterdir():
            if item.is_dir():
                labels.add(item.name)
    
    # 排序以保证每次运行映射关系一致
    return sorted(list(labels))

def copy_and_reorganize(source_root, target_root, label_map):
    splits = ['train', 'val', 'test']
    
    for split in splits:
        src_split_path = Path(source_root) / split
        if not src_split_path.exists():
            continue

        print(f"\n正在处理: {split} 集...")
        
        # netFound 要求每个独立任务文件夹下必须有 'raw' 文件夹
        # 例如: ./netfound_ready_dataset/train_data/raw/0/
        # 这样你可以对 train_data 文件夹运行 preprocess_data.py
        target_split_root = Path(target_root) / f"{split}_data" / "raw"
        
        # 遍历该 split 下的所有标签文件夹
        for label_name in os.listdir(src_split_path):
            label_dir = src_split_path / label_name
            if not label_dir.is_dir():
                continue
                
            # 获取对应的整数 ID
            if label_name not in label_map:
                continue # 理论上不会发生
            
            class_id = label_map[label_name]
            
            # 创建目标目录: target/train_data/raw/<class_id>
            target_class_dir = target_split_root / str(class_id)
            target_class_dir.mkdir(parents=True, exist_ok=True)
            
            # 获取所有 pcap 文件
            files = list(label_dir.glob("*.pcap"))
            
            if not files:
                continue

            # 复制文件
            # 使用 tqdm 显示进度条，如果未安装 tqdm 可直接用 for f in files:
            desc = f"Copying {label_name}(ID:{class_id})"
            for f in tqdm(files, desc=desc, unit="file"):
                shutil.copy2(f, target_class_dir / f.name)

def main():
    source_path = Path(SOURCE_DATASET_DIR)
    target_path = Path(TARGET_DIR)

    if not source_path.exists():
        print(f"错误: 源目录 {source_path} 不存在！")
        return

    # 1. 建立 Label 到 Integer 的映射
    sorted_labels = get_all_labels(source_path)
    if not sorted_labels:
        print("未找到任何标签文件夹，请检查目录结构。")
        return

    label_map = {name: idx for idx, name in enumerate(sorted_labels)}
    
    print("\n" + "="*40)
    print("生成的标签映射 (Label Map):")
    print(json.dumps(label_map, indent=4))
    print("="*40 + "\n")
    
    # 保存映射关系到文件，方便后续查看
    target_path.mkdir(parents=True, exist_ok=True)
    with open(target_path / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=4)
        
    # 2. 执行复制和重组
    copy_and_reorganize(source_path, target_path, label_map)
    
    print("\n" + "="*40)
    print("处理完成！")
    print(f"新数据集位于: {target_path.absolute()}")
    print("目录结构如下 (适配 netFound):")
    for split in ['train', 'val', 'test']:
        print(f"  ├── {split}_data/")
        print(f"  │   └── raw/")
        print(f"  │       ├── 0/  (对应 {sorted_labels[0]})")
        print(f"  │       └── ...")
    print("="*40)

if __name__ == "__main__":
    main()