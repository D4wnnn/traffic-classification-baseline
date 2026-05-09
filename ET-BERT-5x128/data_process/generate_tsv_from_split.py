#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import csv
import glob
import shutil
import binascii
import argparse
import numpy as np  # 新增: 用于读取 CSTNET .npy 文件
import scapy.all as scapy
from flowcontainer.extractor import extract
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ==================== 特征生成工具 ====================

def bigram_generation(packet_datagram, packet_len=64):
    """生成 bigram 特征 (修正版: Byte-level)"""
    result = ''
    
    # 修正点：先将 Hex 字符串按 2 个字符切分为 Byte 列表
    # 输入 "1a2b3c" -> ['1a', '2b', '3c']
    bytes_list = [packet_datagram[i:i+2] for i in range(0, len(packet_datagram), 2)]
    
    generated_datagram = bytes_list
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            # 拼接相邻的两个 Byte，形成 4 字符的 Token (如 "1a2b")
            merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
            result += merge_word_bigram + ' '
    
    return result.strip()

# def bigram_generation(packet_datagram, packet_len=64):
#     """
#     正确的 Bigram 生成：以字节为单位滑动
#     输入: "1a2b3c4d"
#     输出: "1a2b 2b3c 3c4d"
#     """
#     # 1. 预处理：确保是偶数长度且全小写
#     if len(packet_datagram) % 2 != 0:
#         packet_datagram = packet_datagram[:-1]
    
#     # 2. 将 hex 字符串按 2 位切分为字节列表
#     # "1a2b3c" -> ['1a', '2b', '3c']
#     bytes_list = [packet_datagram[i:i+2] for i in range(0, len(packet_datagram), 2)]
    
#     res = []
#     # 3. 滑动窗口提取 Bigram (两个连续字节)
#     # 限制 packet_len 为 Token 的数量
#     for i in range(len(bytes_list) - 1):
#         if len(res) >= packet_len:
#             break
#         # 拼接相邻字节：'1a' + '2b' -> '1a2b'
#         bigram = bytes_list[i] + bytes_list[i+1]
#         res.append(bigram)
        
#     return " ".join(res)
# ==================== PCAP 处理逻辑 (原有) ====================

# def get_feature_flow(pcap_file, payload_len=128, payload_pac=5):
#     """从流级别的 PCAP 提取特征"""
#     try:
#         packets = scapy.rdpcap(pcap_file)
        
#         if len(packets) < 3:
#             return None
        
#         # 检查是否为 TCP/UDP 流
#         feature_result = extract(pcap_file, filter='tcp')
#         if len(feature_result) == 0:
#             feature_result = extract(pcap_file, filter='udp')
#             if len(feature_result) == 0:
#                 return None
        
#         flow_data_string = ''
#         packet_count = 0
        
#         for packet in packets:
#             packet_count += 1
#             if packet_count > payload_pac:
#                 break
            
#             # 提取 payload (跳过前 76 字节的头部)
#             packet_data = packet.copy()
#             data = binascii.hexlify(bytes(packet_data))
#             packet_string = data.decode()[76:]  # 去除以太网头、IP头、传输层头
            
#             # 生成 bigram
#             flow_data_string += bigram_generation(packet_string, packet_len=payload_len)
        
#         return flow_data_string
    
#     except Exception as e:
#         # 多进程中不打印错误，避免输出混乱
#         return None


def get_feature_flow(pcap_file, payload_len=128, payload_pac=5):
    """从流级别的 PCAP 提取特征"""
    try:
        packets = scapy.rdpcap(pcap_file)
        
        if len(packets) < 3:
            return None
        
        # 检查是否为 TCP/UDP 流
        feature_result = extract(pcap_file, filter='tcp')
        if len(feature_result) == 0:
            feature_result = extract(pcap_file, filter='udp')
            if len(feature_result) == 0:
                return None
        
        flow_data_string = ''
        packet_count = 0
        packet_features = [] # 修改点：使用列表存储每个包的 bigram 结果
        for packet in packets:
            packet_count += 1
            if packet_count > payload_pac:
                break
            
            # 提取 payload
            packet_data = packet.copy()
            data = binascii.hexlify(bytes(packet_data))
            packet_string = data.decode()[76:]  
            
            # 生成 bigram
            feature = bigram_generation(packet_string, packet_len=payload_len)
            
            if feature: # 确保提取到了内容
                packet_features.append(feature)
        
        return " ".join(packet_features)
    
    except Exception as e:
        # 多进程中不打印错误，避免输出混乱
        return None


def process_single_pcap(args):
    """处理单个PCAP文件（用于多进程）"""
    pcap_file, label_id, payload_len, payload_pac = args
    feature = get_feature_flow(pcap_file, payload_len, payload_pac)
    if feature and len(feature) > 10:
        return (feature, label_id)
    return None

# ==================== CSTNET 处理逻辑 (新增) ====================

def process_cstnet_dataset(base_dir, output_dir, payload_len=128):
    """
    直接处理 CSTNET 的 .npy 文件生成 ET-BERT 的 TSV 数据集
    """
    print(f"\n检测到 CSTNET 格式 (.npy)，进入 CSTNET 处理模式...")
    
    # 清空输出目录
    clear_directory(output_dir)
    
    # CSTNET 的文件命名习惯通常是 x_datagram_{split}.npy 和 y_{split}.npy
    # split 通常包含 train, test, valid (或 val)
    splits = [
        ("train", "x_datagram_train.npy", "y_train.npy"),
        ("test", "x_datagram_test.npy", "y_test.npy"),
        ("val", "x_datagram_valid.npy", "y_valid.npy") # 尝试匹配 valid
    ]

    for split_name, x_name, y_name in splits:
        x_path = os.path.join(base_dir, x_name)
        y_path = os.path.join(base_dir, y_name)
        
        # 兼容 val/valid 命名
        if split_name == "val" and not os.path.exists(x_path):
             x_path = os.path.join(base_dir, "x_datagram_val.npy")
             y_path = os.path.join(base_dir, "y_val.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f"跳过 {split_name}: 未找到 {x_name} 或 {y_name}")
            continue
            
        print(f"\n{'='*60}")
        print(f"处理 CSTNET {split_name.upper()} 数据集")
        print(f"加载: {x_path}")
        print(f"{'='*60}")

        # 加载数据
        try:
            x_data = np.load(x_path, allow_pickle=True)
            y_data = np.load(y_path, allow_pickle=True)
        except Exception as e:
            print(f"加载 .npy 文件失败: {e}")
            continue

        print(f"样本数量: {len(x_data)}")
        
        processed_data = []
        processed_labels = []

        # 处理进度条
        for i in tqdm(range(len(x_data)), desc=f"生成特征 {split_name}"):
            hex_string = x_data[i]
            label = y_data[i]
            
            # 注意：CSTNET的datagram通常已经是清洗过的Hex字符串
            # 我们直接对其进行bigram生成。
            # 这里 payload_len 控制生成的序列长度
            feature = bigram_generation(hex_string, packet_len=payload_len)
            
            if feature and len(feature) > 0:
                processed_data.append(feature)
                processed_labels.append(label)

        # 确定输出文件名
        if split_name == "val":
            out_name = "valid_dataset.tsv"
        else:
            out_name = f"{split_name}_dataset.tsv"
            
        output_file = os.path.join(output_dir, out_name)
        write_dataset_tsv(processed_data, processed_labels, output_file)
        
        print(f"成功转换: {len(processed_data)} 个样本")

# ==================== 通用工具 ====================

def write_dataset_tsv(data, labels, output_file):
    """写入 TSV 文件"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        
        # 写入表头
        tsv_writer.writerow(['label', 'text_a'])
        
        # 写入数据
        for label, text in zip(labels, data):
            tsv_writer.writerow([label, text])
    
    print(f"已保存到: {output_file}")


def clear_directory(directory):
    """清空目录内容"""
    if os.path.exists(directory):
        print(f"清空目录: {directory}")
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    print(f"已创建干净的目录: {directory}\n")


def generate_dataset(base_dir, output_dir="./datasets", payload_len=128, payload_pac=5, num_workers=None):
    """
    从 PCAP 目录生成数据集（原有多进程版本）
    """
    # 清空输出目录
    clear_directory(output_dir)
    
    # 确定进程数
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"使用 {num_workers} 个进程进行并行处理\n")
    
    # 获取所有标签（从 train 目录）
    train_dir = os.path.join(base_dir, "train")
    if not os.path.exists(train_dir):
        print("错误: 未找到 train 目录，且未检测到 CSTNET .npy 文件。")
        return

    labels = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    
    print(f"找到 {len(labels)} 个类别:")
    for label, label_id in label_to_id.items():
        print(f"  [{label_id}] {label}")
    
    # 处理每个数据集
    for dataset_type in ["train", "val", "test"]:
        dataset_dir = os.path.join(base_dir, dataset_type)
        
        if not os.path.exists(dataset_dir):
            print(f"\n警告: {dataset_dir} 不存在，跳过")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理 {dataset_type.upper()} 数据集")
        print(f"{'='*60}")
        
        # 收集所有待处理的文件
        all_tasks = []
        label_file_counts = {label_id: 0 for label_id in label_to_id.values()}
        
        for label_name in labels:
            label_dir = os.path.join(dataset_dir, label_name)
            
            if not os.path.exists(label_dir):
                continue
            
            label_id = label_to_id[label_name]
            
            # 获取所有 pcap 文件
            pcap_files = glob.glob(os.path.join(label_dir, "*.pcap"))
            pcap_files.extend(glob.glob(os.path.join(label_dir, "*.pcapng")))
            
            label_file_counts[label_id] = len(pcap_files)
            
            # 添加到任务列表
            for pcap_file in pcap_files:
                all_tasks.append((pcap_file, label_id, payload_len, payload_pac))
        
        print(f"\n待处理文件统计:")
        for label_name, label_id in sorted(label_to_id.items(), key=lambda x: x[1]):
            print(f"  [{label_id}] {label_name}: {label_file_counts[label_id]} 个文件")
        print(f"  总计: {len(all_tasks)} 个文件\n")
        
        # 多进程处理
        data_list = []
        label_list = []
        
        print(f"开始多进程处理...")
        with Pool(processes=num_workers) as pool:
            # 使用imap_unordered提高效率，tqdm显示进度
            results = list(tqdm(
                pool.imap_unordered(process_single_pcap, all_tasks),
                total=len(all_tasks),
                desc=f"处理 {dataset_type}",
                unit="文件"
            ))
        
        # 过滤None结果并整理数据
        for result in results:
            if result is not None:
                feature, label_id = result
                data_list.append(feature)
                label_list.append(label_id)
        
        # 统计信息
        label_stats = {label_id: 0 for label_id in label_to_id.values()}
        for label_id in label_list:
            label_stats[label_id] += 1
        
        print(f"\n{dataset_type.upper()} 数据集处理结果:")
        print(f"  成功处理: {len(data_list)} / {len(all_tasks)} 个样本")
        print(f"  成功率: {len(data_list)/len(all_tasks)*100:.2f}%")
        
        # 写入 TSV
        if dataset_type == "val":
            output_file = os.path.join(output_dir, "valid_dataset.tsv")
        else:
            output_file = os.path.join(output_dir, f"{dataset_type}_dataset.tsv")
        
        write_dataset_tsv(data_list, label_list, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ET-BERT Data Preprocessing')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input dataset (containing train/val/test OR .npy files)')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output TSV files')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker processes (Only for PCAP)')
    
    args = parser.parse_args()

    # ==================== 配置参数 ====================
    BASE_DIR = args.dataset_path
    OUTPUT_DIR = args.output_path
    NUM_WORKERS = args.num_workers
    
    # 注意：对于 CSTNET，这个长度用于控制生成的 bigram 序列的总长度
    PAYLOAD_LEN = 128   
    PAYLOAD_PAC = 5           
    # =================================================
    
    print("="*60)
    print("ET-BERT 数据集生成器 (支持 PCAP 目录结构 和 CSTNET .npy)")
    print("="*60)
    print(f"数据目录: {os.path.abspath(BASE_DIR)}")
    print(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    
    # 自动检测是否为 CSTNET 格式
    # 检测逻辑：目录下是否存在 x_datagram_train.npy
    is_cstnet = os.path.exists(os.path.join(BASE_DIR, "x_datagram_train.npy"))
    
    if is_cstnet:
        process_cstnet_dataset(
            BASE_DIR,
            OUTPUT_DIR,
            payload_len=PAYLOAD_LEN  # CSTNET数据直接使用hex串生成bigram，此处控制长度
        )
    else:
        print(f"Payload 长度: {PAYLOAD_LEN} 字节")
        print(f"使用包数: 前 {PAYLOAD_PAC} 个包")
        print(f"进程数: {'自动检测' if NUM_WORKERS is None else NUM_WORKERS}")
        print("="*60 + "\n")
        
        generate_dataset(
            BASE_DIR, 
            OUTPUT_DIR, 
            PAYLOAD_LEN, 
            PAYLOAD_PAC,
            NUM_WORKERS
        )
    
    print("\n" + "="*60)
    print("✓ 数据集生成完成!")
    print("="*60)