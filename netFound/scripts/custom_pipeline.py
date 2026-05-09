import os
import argparse
import subprocess
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm import tqdm  # 需要安装: pip install tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Custom Pipeline for Pre-split Dataset")
    parser.add_argument("--input_root", type=str, required=True, 
                        help="Root directory containing train/val/test folders (e.g., /data/my_dataset)")
    parser.add_argument("--output_root", type=str, required=True, 
                        help="Root directory for output (intermediate and final)")
    parser.add_argument("--bin_dir", type=str, default="../src/pre_process/packets_processing_src/3_field_extraction",
                        help="Path to the directory containing the '3_field_extraction' binary")
    parser.add_argument("--tokenizer_script", type=str, default="../src/pre_process/Tokenize.py",
                        help="Path to Tokenize.py")
    parser.add_argument("--tokenizer_config", type=str, required=True, help="Path to tokenizer config json")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2), 
                        help="Number of parallel worker processes")
    parser.add_argument("--tcp_options", action="store_true", help="Enable TCP options extraction")
    return parser.parse_args()

def run_command(cmd):
    """辅助函数：运行 Shell 命令"""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e.stderr.decode()

def worker_extract(task):
    """
    单个文件的特征提取任务，供多进程池调用
    task: (binary_path, input_pcap, output_bin, tcp_flag)
    """
    binary_path, input_pcap, output_bin, tcp_flag = task
    
    # 这里的 3_field_extraction 需要 output 是文件前缀或者文件名
    # 根据原 C++ 代码逻辑，它会自动加上后缀，所以我们给全路径即可
    
    cmd = [binary_path, str(input_pcap), str(output_bin), tcp_flag]
    success, err = run_command(cmd)
    return success, err, input_pcap

def step_1_extraction(args, splits=['train', 'val', 'test']):
    """
    阶段一：调用 C++ 程序提取特征
    """
    binary_path = os.path.join(args.bin_dir, "3_field_extraction")
    if not os.path.exists(binary_path):
        # 尝试构建默认路径 (假设脚本在 netFound/scripts/)
        base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        binary_path = os.path.join(base_dir, "src/pre_process/packets_processing_src/3_field_extraction/3_field_extraction")
        
    if not os.path.exists(binary_path):
        raise FileNotFoundError(f"Could not find C++ binary at: {binary_path}. Did you run 'make'?")

    logger.info(f"Using Extraction Binary: {binary_path}")
    
    tasks = []
    extraction_out_dir = Path(args.output_root) / "extracted"

    logger.info("Scanning input directory for .pcap files...")
    
    # 遍历 train, val, test
    input_root = Path(args.input_root)
    for split in splits:
        split_dir = input_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} does not exist, skipping.")
            continue
            
        # 遍历 Labels (子目录)
        for label_dir in split_dir.iterdir():
            if label_dir.is_dir():
                # 遍历 Pcap 文件
                for pcap_file in label_dir.glob("*.pcap"):
                    # 构建输出路径: output/extracted/train/label_0/flow_x.pcap
                    rel_path = pcap_file.relative_to(input_root)
                    target_dir = extraction_out_dir / rel_path.parent
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    target_file = target_dir / pcap_file.name
                    
                    # 只有当目标文件不存在时才加入任务 (支持断点续传)
                    # 注意：C++程序会生成 .tcp.6 等后缀，这里只简单判断主文件逻辑，
                    # 如果需要严谨重跑，请手动清空 output 目录
                    tasks.append((
                        binary_path, 
                        str(pcap_file), 
                        str(target_file), 
                        "1" if args.tcp_options else "0"
                    ))

    logger.info(f"Found {len(tasks)} files to extract.")
    
    # 并行执行
    if tasks:
        logger.info(f"Starting extraction with {args.workers} workers...")
        with Pool(args.workers) as pool:
            # 使用 tqdm 显示进度条
            for success, err, fname in tqdm(pool.imap_unordered(worker_extract, tasks), total=len(tasks)):
                if not success:
                    logger.error(f"Failed to extract {fname}: {err}")
    else:
        logger.info("No files found or all files already processed.")

def step_2_tokenization(args, splits=['train', 'val', 'test']):
    """
    阶段二：将提取后的二进制文件转换为 Arrow 格式
    注意：Tokenize.py 是针对一个文件夹（包含同一类别的流）进行处理的
    """
    extraction_out_dir = Path(args.output_root) / "extracted"
    final_out_dir = Path(args.output_root) / "final"
    
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    tokenize_script = os.path.join(base_dir, "src/pre_process/Tokenize.py")

    logger.info("Starting Tokenization...")

    for split in splits:
        split_extracted_dir = extraction_out_dir / split
        if not split_extracted_dir.exists():
            continue
        
        # 遍历 Label 目录
        label_dirs = [d for d in split_extracted_dir.iterdir() if d.is_dir()]
        
        for label_dir in tqdm(label_dirs, desc=f"Tokenizing {split}"):
            label_name = label_dir.name # e.g., "0", "1"
            
            # 输出路径: output/final/train/label_0
            target_shard_dir = final_out_dir / split / label_name
            target_shard_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "python3", tokenize_script,
                "--conf_file", args.tokenizer_config,
                "--input_dir", str(label_dir),
                "--output_dir", str(target_shard_dir),
                "--label", label_name,
                "--cores", str(min(args.workers, 4)) # Tokenize 内部也有多进程，限制一下避免过载
            ]
            
            # Tokenize.py 内部会处理该目录下的所有文件
            success, err = run_command(cmd)
            if not success:
                logger.error(f"Tokenization failed for {label_dir}: {err}")

def main():
    args = get_args()
    
    if not os.path.exists(args.input_root):
        logger.error(f"Input root {args.input_root} does not exist.")
        return

    # 1. 特征提取 (C++)
    print("="*40)
    print("Step 1: Feature Extraction")
    print("="*40)
    step_1_extraction(args)

    # 2. Token化 (Python)
    print("\n" + "="*40)
    print("Step 2: Tokenization")
    print("="*40)
    step_2_tokenization(args)
    
    print("\nDone! Processed data is in:", os.path.join(args.output_root, "final"))

if __name__ == "__main__":
    main()