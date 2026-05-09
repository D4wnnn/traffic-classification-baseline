# File: TrafficFormer/data_generation/generate_finetuning_data.py

import os
import sys
from finetuning_data_gen import get_feature_flow
from utils import write_dataset_tsv, bigram_generation
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import json
import numpy as np

def process_single_pcap(args):
    """
    处理单个pcap文件
    """
    pcap_path, label_id, payload_length, payload_packet, start_index = args
    
    try:
        feature_data = get_feature_flow(
            pcap_path, 
            select_packet_len=payload_length,
            packets_num=payload_packet,
            start_index=start_index
        )
        
        if feature_data == -1:
            return None
        
        return (feature_data[0], label_id)
        
    except Exception as e:
        return None

def collect_pcap_files(data_dir, label_to_id):
    """
    收集指定目录下的所有pcap文件及其标签
    """
    if not os.path.exists(data_dir):
        return [], set()
    
    label_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    pcap_list = []
    found_labels = set()
    
    print(f"Scanning {data_dir}...")
    for label_name in tqdm(label_dirs, desc="  Collecting files"):
        if label_name not in label_to_id:
            print(f"  Warning: Label '{label_name}' not in global label mapping, skipping...")
            continue
            
        label_path = os.path.join(data_dir, label_name)
        label_id = label_to_id[label_name]
        found_labels.add(label_name)
        
        pcap_files = [f for f in os.listdir(label_path) if f.endswith('.pcap')]
        
        for pcap_file in pcap_files:
            pcap_path = os.path.join(label_path, pcap_file)
            pcap_list.append((pcap_path, label_id))
    
    return pcap_list, found_labels

def build_global_label_mapping(base_data_dir, splits):
    """
    扫描所有splits，构建全局统一的label映射
    """
    all_labels = set()
    all_labels_by_split = {}
    
    print("=" * 80)
    print("Building global label mapping...")
    print("=" * 80)
    
    for split in splits:
        split_dir = os.path.join(base_data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist")
            all_labels_by_split[split] = set()
            continue
        
        label_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))]
        
        split_labels = set(label_dirs)
        all_labels_by_split[split] = split_labels
        all_labels.update(split_labels)
        
        print(f"{split:10s}: {len(split_labels):3d} labels")
    
    sorted_labels = sorted(all_labels)
    label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
    
    print(f"\nTotal unique labels: {len(label_to_id)}")
    print(f"Label mapping: {label_to_id}")
    
    print("\n" + "=" * 80)
    print("Label consistency check:")
    print("=" * 80)
    
    for split in splits:
        if split not in all_labels_by_split:
            continue
        split_labels = all_labels_by_split[split]
        missing = all_labels - split_labels
        if missing:
            print(f"Warning: {split} is missing labels: {missing}")
    
    return label_to_id, all_labels_by_split

def process_split_multiprocess(data_dir, output_dir, output_filename,
                               label_to_id,
                               payload_length=64, 
                               payload_packet=5, 
                               start_index=76,
                               num_workers=None):
    """
    使用多进程处理单个数据集划分（train/val/test）- PCAP 模式
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 32)
    
    print(f"\nUsing {num_workers} worker processes")
    
    pcap_list, found_labels = collect_pcap_files(data_dir, label_to_id)
    
    if not pcap_list:
        print(f"Warning: No pcap files found in {data_dir}")
        return
    
    print(f"Found {len(found_labels)} labels: {sorted(found_labels)}")
    print(f"Total pcap files to process: {len(pcap_list)}")
    
    process_args = [
        (pcap_path, label_id, payload_length, payload_packet, start_index)
        for pcap_path, label_id in pcap_list
    ]
    
    data = []
    labels = []
    
    print(f"\nProcessing {len(process_args)} files with {num_workers} workers...")
    
    with mp.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_pcap, process_args, chunksize=10),
            total=len(process_args),
            desc="  Processing"
        ))
    
    for result in results:
        if result is not None:
            datagram, label_id = result
            data.append(datagram)
            labels.append(label_id)
    
    success_rate = len(data) / len(pcap_list) * 100 if pcap_list else 0
    print(f"\nSuccessfully processed: {len(data)}/{len(pcap_list)} files ({success_rate:.1f}%)")
    
    print(f"\nLabel distribution:")
    label_counts = {}
    for label_id in labels:
        label_counts[label_id] = label_counts.get(label_id, 0) + 1
    
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    for label_id in sorted(label_counts.keys()):
        label_name = id_to_label.get(label_id, "Unknown")
        count = label_counts[label_id]
        print(f"  {label_id:3d} ({label_name:30s}): {count:6d} samples")
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    prefix = output_filename.replace("_dataset.tsv", "")
    write_dataset_tsv(data, labels, output_dir + "/", prefix)
    
    print(f"\nSaved to {output_path}")


# ============================================================================
# CSTNET 支持函数
# ============================================================================

def is_cstnet_dataset(data_dir):
    """
    检测是否为 CSTNET 数据集格式
    """
    return os.path.exists(os.path.join(data_dir, "x_datagram_train.npy"))


def parse_ipv4_frames_from_hex(hex_string):
    """
    从拼接的 IPv4 Hex 字符串中解析出各个数据包
    
    CSTNET 的 datagram 可能是多个 IP 包拼接在一起的 hex 字符串
    需要根据 IP 头中的 Total Length 字段来切分
    """
    hex_string = hex_string.replace(" ", "").replace("\n", "").lower()
    if not hex_string:
        return []
    
    frames = []
    current_idx = 0
    total_len_chars = len(hex_string)
    
    while current_idx < total_len_chars:
        # 检查 IPv4 版本号 (第一个字符应该是 '4')
        if current_idx + 8 > total_len_chars:
            break
            
        if hex_string[current_idx] != '4':
            current_idx += 2  # 跳过一个字节
            continue
            
        try:
            # Total Length 在 IP 头的第 2-3 字节 (offset 4-7 hex chars)
            len_hex = hex_string[current_idx + 4 : current_idx + 8]
            packet_len = int(len_hex, 16)
            packet_len_chars = packet_len * 2
            
            if packet_len == 0 or packet_len < 20:  # 最小 IP 包长度
                current_idx += 2
                continue

            if current_idx + packet_len_chars <= total_len_chars:
                frames.append(hex_string[current_idx : current_idx + packet_len_chars])
                current_idx += packet_len_chars
            else:
                # 包不完整，添加剩余部分
                frames.append(hex_string[current_idx:])
                break
        except ValueError:
            current_idx += 2
            continue
            
    return frames


def process_cstnet_hex_to_trafficformer(hex_string, payload_length=64, payload_packet=5, start_index=76):
    """
    将 CSTNET 的 hex 字符串转换为 TrafficFormer 的输入格式
    
    模拟 get_feature_flow 中 No_ether 的处理逻辑
    """
    # 解析出各个 IP 数据包
    packets = parse_ipv4_frames_from_hex(hex_string)
    
    if not packets:
        return None
        
    packet_count = 0
    flow_data_string = ''
    
    # 从第一个包确定 "客户端" IP（用于判断方向）
    client_ip = None
    if len(packets[0]) >= 32:
        # IP Source Address 在 offset 12 字节 (24 hex chars)
        client_ip = packets[0][24:32]
        
    for packet_string in packets:
        # 判断方向
        is_forward = True
        if client_ip and len(packet_string) >= 32:
            src_ip = packet_string[24:32]
            if src_ip != client_ip:
                is_forward = False
        
        # 添加伪造的以太网头 (TrafficFormer 的逻辑)
        # Forward: c49a025996f8e46f13e2e3ae0800 (14 bytes = 28 hex chars)
        # Backward: e46f13e2e3aec49a025996f80800
        if is_forward:
            packet_string = "c49a025996f8e46f13e2e3ae0800" + packet_string
        else:
            packet_string = "e46f13e2e3aec49a025996f80800" + packet_string
            
        # 截取指定长度的 payload
        if len(packet_string) > start_index:
            packet_string = packet_string[start_index : start_index + 2 * payload_length]
        else:
            packet_string = ""
            
        if not packet_string:
            continue
            
        # 添加分隔符和 bigram
        flow_data_string += "[SEP] "
        flow_data_string += bigram_generation(packet_string.strip(), token_len=len(packet_string.strip()), flag=True)
        
        packet_count += 1
        if packet_count >= payload_packet:
            break
            
    return flow_data_string.strip() if flow_data_string.strip() else None


def process_cstnet_dataset(data_dir, output_dir, payload_length=64, payload_packet=5, start_index=76):
    """
    处理 CSTNET 数据集，生成 TrafficFormer 格式的 TSV 文件
    """
    print("\n" + "=" * 80)
    print("检测到 CSTNET 格式 (.npy)，进入 CSTNET 处理模式...")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # CSTNET 文件命名规则
    splits_config = [
        ("train", "x_datagram_train.npy", "y_train.npy", "train_dataset.tsv"),
        ("test", "x_datagram_test.npy", "y_test.npy", "test_dataset.tsv"),
        ("valid", "x_datagram_valid.npy", "y_valid.npy", "valid_dataset.tsv"),
    ]
    
    # 首先收集所有标签，构建统一映射
    all_labels = set()
    for split_name, x_name, y_name, _ in splits_config:
        y_path = os.path.join(data_dir, y_name)
        
        # 兼容 val/valid 命名
        if not os.path.exists(y_path) and split_name == "valid":
            y_path = os.path.join(data_dir, "y_val.npy")
            
        if os.path.exists(y_path):
            y_data = np.load(y_path, allow_pickle=True)
            all_labels.update(set(y_data))
    
    # 如果标签是整数，直接使用；如果是字符串，创建映射
    sample_label = list(all_labels)[0] if all_labels else None
    if isinstance(sample_label, (int, np.integer)):
        # 标签已经是整数，不需要映射
        label_to_id = None
        print(f"Labels are already integers. Unique labels: {sorted(all_labels)}")
    else:
        # 标签是字符串，创建映射
        sorted_labels = sorted([str(l) for l in all_labels])
        label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        print(f"Created label mapping for {len(label_to_id)} labels")
        print(f"Label mapping: {label_to_id}")
    
    # 处理每个 split
    for split_name, x_name, y_name, out_name in splits_config:
        x_path = os.path.join(data_dir, x_name)
        y_path = os.path.join(data_dir, y_name)
        
        # 兼容 val/valid 命名
        if not os.path.exists(x_path) and split_name == "valid":
            x_path = os.path.join(data_dir, "x_datagram_val.npy")
            y_path = os.path.join(data_dir, "y_val.npy")
            
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"\n跳过 {split_name}: 未找到 {x_name} 或 {y_name}")
            continue
            
        print(f"\n{'='*60}")
        print(f"处理 CSTNET {split_name.upper()} 数据集")
        print(f"加载: {x_path}")
        print(f"{'='*60}")
        
        try:
            x_data = np.load(x_path, allow_pickle=True)
            y_data = np.load(y_path, allow_pickle=True)
        except Exception as e:
            print(f"加载 .npy 文件失败: {e}")
            continue
            
        print(f"样本数量: {len(x_data)}")
        
        processed_data = []
        processed_labels = []
        
        for i in tqdm(range(len(x_data)), desc=f"生成特征 {split_name}"):
            hex_string = x_data[i]
            label = y_data[i]
            
            # 转换标签
            if label_to_id is not None:
                label = label_to_id[str(label)]
            else:
                label = int(label)
            
            # 生成 TrafficFormer 格式的特征
            feature = process_cstnet_hex_to_trafficformer(
                hex_string,
                payload_length=payload_length,
                payload_packet=payload_packet,
                start_index=start_index
            )
            
            if feature and len(feature) > 10:
                processed_data.append(feature)
                processed_labels.append(label)
        
        # 写入 TSV
        output_file = os.path.join(output_dir, out_name)
        prefix = out_name.replace("_dataset.tsv", "")
        write_dataset_tsv(processed_data, processed_labels, output_dir + "/", prefix)
        
        # 统计信息
        label_counts = {}
        for lbl in processed_labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        
        print(f"\n成功转换: {len(processed_data)} / {len(x_data)} 个样本")
        print(f"标签分布:")
        for lbl in sorted(label_counts.keys()):
            print(f"  Label {lbl}: {label_counts[lbl]} samples")
    
    # 保存标签映射
    if label_to_id is not None:
        mapping_path = os.path.join(output_dir, "label_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(label_to_id, f, indent=2, ensure_ascii=False)
        print(f"\nLabel mapping saved to: {mapping_path}")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fine-tuning dataset with multiprocessing')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Base data directory containing train/val/test folders OR CSTNET .npy files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for TSV files')
    parser.add_argument('--payload_length', type=int, default=64,
                       help='Number of bytes to use from each packet')
    parser.add_argument('--payload_packet', type=int, default=5,
                       help='Number of packets to use')
    parser.add_argument('--start_index', type=int, default=76,
                       help='Starting byte index (in hex chars)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count, only for PCAP mode)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Which splits to process (default: train val test, only for PCAP mode)')
    
    args = parser.parse_args()
    
    # 自动检测数据集格式
    if is_cstnet_dataset(args.data_dir):
        # CSTNET 模式
        print("=" * 80)
        print("TrafficFormer 数据集生成器 - CSTNET 模式")
        print("=" * 80)
        print(f"数据目录: {os.path.abspath(args.data_dir)}")
        print(f"输出目录: {os.path.abspath(args.output_dir)}")
        print(f"Payload 长度: {args.payload_length} bytes")
        print(f"使用包数: {args.payload_packet}")
        print(f"起始索引: {args.start_index}")
        
        process_cstnet_dataset(
            args.data_dir,
            args.output_dir,
            payload_length=args.payload_length,
            payload_packet=args.payload_packet,
            start_index=args.start_index
        )
    else:
        # PCAP 目录模式 (原有逻辑)
        print("=" * 80)
        print("TrafficFormer 数据集生成器 - PCAP 模式")
        print("=" * 80)
        
        # 第一步：构建全局label映射
        label_to_id, all_labels_by_split = build_global_label_mapping(
            args.data_dir, 
            args.splits
        )
        
        # 第二步：处理每个划分
        for split in args.splits:
            print("\n" + "=" * 80)
            print(f"Processing {split.upper()} set")
            print("=" * 80)
            
            split_dir = os.path.join(args.data_dir, split)
            if not os.path.exists(split_dir):
                print(f"Skipping {split} (directory does not exist)")
                continue
            
            if split == "val":
                output_filename = "valid_dataset.tsv"
            else:
                output_filename = f"{split}_dataset.tsv"
            
            process_split_multiprocess(
                split_dir,
                args.output_dir,
                output_filename,
                label_to_id,
                payload_length=args.payload_length,
                payload_packet=args.payload_packet,
                start_index=args.start_index,
                num_workers=args.num_workers
            )
        
        # 第三步：保存label映射
        mapping_path = os.path.join(args.output_dir, "label_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(label_to_id, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 80)
        print(f"Label mapping saved to: {mapping_path}")
    
    print("\n" + "=" * 80)
    print("✓ 数据集生成完成!")
    print(f"TSV files saved to: {args.output_dir}")
    print("=" * 80)