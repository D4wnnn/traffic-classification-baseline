#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
Label Mapping Generator

独立脚本：从 PCAP 数据集目录结构中提取 label 到 id 的映射关系并保存。
映射规则与原数据预处理脚本一致：按文件夹名称的字母顺序排序，依次分配 id。

使用方法：
    python generate_label_mapping.py --dataset_path /path/to/dataset --output_path /path/to/output

输入目录结构要求：
    dataset_path/
    ├── train/
    │   ├── Label_A/
    │   ├── Label_B/
    │   └── Label_C/
    ├── val/
    └── test/

输出文件：
    - label_mapping.json  (包含双向映射的 JSON 文件)
    - label_mapping.csv   (方便查看的 CSV 文件)
"""

import os
import csv
import json
import argparse


def get_labels_from_directory(base_dir):
    """
    从数据集目录中获取所有 label（类别名称）
    
    Args:
        base_dir: str, 数据集根目录
        
    Returns:
        list: 按字母顺序排序的 label 列表
    """
    train_dir = os.path.join(base_dir, "train")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"未找到 train 目录: {train_dir}")
    
    # 获取 train 目录下的所有子文件夹（即类别）
    labels = sorted([
        d for d in os.listdir(train_dir) 
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    
    if not labels:
        raise ValueError(f"train 目录下未找到任何子文件夹: {train_dir}")
    
    return labels


def create_label_mapping(labels):
    """
    创建 label 到 id 的映射
    
    Args:
        labels: list, label 名称列表（已排序）
        
    Returns:
        tuple: (label_to_id, id_to_label)
    """
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for idx, label in enumerate(labels)}
    
    return label_to_id, id_to_label


def save_label_mapping(label_to_id, id_to_label, output_dir):
    """
    保存 label 映射到文件
    
    Args:
        label_to_id: dict, {label_name: label_id}
        id_to_label: dict, {label_id: label_name}
        output_dir: str, 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存 JSON 格式
    json_path = os.path.join(output_dir, "label_mapping.json")
    
    mapping_data = {
        "label_to_id": label_to_id,
        "id_to_label": {str(k): v for k, v in id_to_label.items()},
        "num_labels": len(label_to_id),
        "labels": list(label_to_id.keys())
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON 映射已保存: {json_path}")
    
    # 2. 保存 CSV 格式
    csv_path = os.path.join(output_dir, "label_mapping.csv")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label_id', 'label_name'])
        for label_id in sorted(id_to_label.keys()):
            writer.writerow([label_id, id_to_label[label_id]])
    
    print(f"✅ CSV 映射已保存: {csv_path}")
    
    return json_path, csv_path


def load_label_mapping(mapping_path):
    """
    从 JSON 文件加载 label 映射
    
    Args:
        mapping_path: str, 映射文件路径 (label_mapping.json)
        
    Returns:
        tuple: (label_to_id, id_to_label)
    """
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    
    label_to_id = mapping_data["label_to_id"]
    id_to_label = {int(k): v for k, v in mapping_data["id_to_label"].items()}
    
    return label_to_id, id_to_label


def get_label_names_list(mapping_path):
    """
    从映射文件获取按 id 排序的 label 名称列表
    用于混淆矩阵的行列标签
    
    Args:
        mapping_path: str, 映射文件路径
        
    Returns:
        list: 按 id 排序的 label 名称列表
    """
    _, id_to_label = load_label_mapping(mapping_path)
    return [id_to_label[i] for i in range(len(id_to_label))]


def main():
    parser = argparse.ArgumentParser(
        description='从 PCAP 数据集目录生成 Label 映射文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python generate_label_mapping.py --dataset_path ./raw_data --output_path ./datasets
    
输入目录结构:
    raw_data/
    ├── train/
    │   ├── BitTorrent/
    │   ├── FTP/
    │   └── HTTP/
    └── ...

输出文件:
    datasets/
    ├── label_mapping.json
    └── label_mapping.csv
        """
    )
    
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        required=True, 
        help='数据集根目录路径 (包含 train 子目录)'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help='输出目录路径'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Label Mapping Generator")
    print("=" * 60)
    print(f"数据集目录: {os.path.abspath(args.dataset_path)}")
    print(f"输出目录:   {os.path.abspath(args.output_path)}")
    print("=" * 60)
    
    # 1. 获取所有 label
    try:
        labels = get_labels_from_directory(args.dataset_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 错误: {e}")
        return 1
    
    # 2. 创建映射
    label_to_id, id_to_label = create_label_mapping(labels)
    
    # 3. 打印映射信息
    print(f"\n发现 {len(labels)} 个类别:\n")
    print(f"{'ID':<6} {'Label Name':<30}")
    print("-" * 40)
    for label_id in sorted(id_to_label.keys()):
        print(f"{label_id:<6} {id_to_label[label_id]:<30}")
    print("-" * 40)
    
    # 4. 保存映射
    print()
    save_label_mapping(label_to_id, id_to_label, args.output_path)
    
    print("\n" + "=" * 60)
    print("✅ Label 映射生成完成!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())