import argparse
import os
import time
import json
import numpy as np
import torch
from torchvision import datasets, transforms
import pandas as pd
import util.misc as misc
import models_YaTC
from util.misc import setup_seed
from tqdm import tqdm

# ==========================================
# 1. 核心扰动类 (适配 YaTC 的 Image 输入)
# ==========================================
class Perturber:
    def __init__(self, perturb_type, noise_ratio, device):
        self.type = perturb_type
        self.ratio = noise_ratio
        self.device = device
        # YaTC 使用 Normalize(0.5, 0.5)，所以原始字节 0 对应 -1.0
        self.pad_value = -1.0 

    def apply(self, inputs):
        """
        inputs: Tensor [Batch, 1, 40, 40], Normalized range approx [-1, 1]
        """
        if self.ratio <= 0 or self.type == 'none':
            return inputs

        # --- 1. Byte Masking: 随机掩盖字节 ---
        if self.type == 'byte_mask':
            # 生成掩码 [Batch, 1, 40, 40]
            mask = torch.rand_like(inputs) < self.ratio
            
            # 生成噪声: 模拟归一化后的分布 (-1 到 1)
            # 原始代码是 0-255，这里直接用 uniform(-1, 1) 模拟随机像素
            noise = torch.rand_like(inputs) * 2 - 1 
            
            # 保护 Padding (值为 -1.0 的部分认为是 Pad/无内容)
            # 注意：浮点数比较最好用一定的 tolerance，这里简化处理
            is_content = inputs > (self.pad_value + 1e-4)
            
            final_mask = mask & is_content
            return torch.where(final_mask, noise, inputs)

        # --- 2. Packet Dropout: 随机丢弃整个包 ---
        elif self.type == 'packet_drop':
            # YaTC 的 40x40 图像由 5 个包组成，每个包 320 字节 (40*40 / 5 = 320)
            B, C, H, W = inputs.shape # [B, 1, 40, 40]
            
            # 展平并 reshape 成 [Batch, 5, 320]
            # 假设 pcap_processor 是按顺序拼接的，这样 view 是安全的
            flat_inputs = inputs.view(B, 5, -1) 
            
            # 生成包级别的掩码 [Batch, 5, 1]
            keep_prob = 1.0 - self.ratio
            mask = torch.bernoulli(torch.full((B, 5, 1), keep_prob)).to(self.device)
            
            # 扩展掩码到字节维度 [Batch, 5, 320]
            mask_expanded = mask.expand_as(flat_inputs)
            
            # 执行丢弃：保留的部分保持原样，丢弃的部分置为 pad_value (-1.0)
            # 公式: value * mask + pad_value * (1 - mask)
            perturbed_flat = flat_inputs * mask_expanded + self.pad_value * (1 - mask_expanded)
            
            # Reshape 回图像格式
            return perturbed_flat.view(B, C, H, W)

        return inputs

# ==========================================
# 2. 参数解析
# ==========================================
def get_args_parser():
    parser = argparse.ArgumentParser('YaTC Robustness Evaluation', add_help=False)
    
    # 基础参数
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--data_path', default='./data/ISCXVPN2016_MFR', type=str)
    parser.add_argument('--nb_classes', default=7, type=int)
    parser.add_argument('--device', default='cuda', help='device to use for testing')
    parser.add_argument('--seed', default=42, type=int)
    
    # 模型参数
    parser.add_argument('--model', default='TraFormer_YaTC', type=str)
    parser.add_argument('--input_size', default=40, type=int)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--checkpoint', required=True, type=str)
    
    # 鲁棒性参数
    parser.add_argument('--perturb_type', type=str, default='none', 
                        choices=['none', 'byte_mask', 'packet_drop'],
                        help='Type of perturbation attack')
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='Intensity of noise (0.0 - 1.0)')
    parser.add_argument('--save_result', type=str, default='robustness_results.jsonl',
                        help='Path to append JSONL results')

    # 运行相关
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)
    
    # 兼容性参数 (Distributed)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    return parser

# ==========================================
# 3. 数据集构建 (保持与 final-test.py 一致)
# ==========================================
def build_test_dataset(args):
    mean = [0.5]
    std = [0.5]
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    root = os.path.join(args.data_path, 'test') 
    if not os.path.exists(root):
        raise FileNotFoundError(f"Test dataset path does not exist: {root}")
        
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

# ==========================================
# 4. 主程序
# ==========================================
def main(args):
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    setup_seed(args.seed)

    # Load Data
    dataset_test = build_test_dataset(args)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    # Build Model
    print(f"Creating model: {args.model}")
    model = models_YaTC.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    # Load Weights
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded checkpoint. Missing: {msg.missing_keys}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model.to(device)
    model.eval()

    # Init Perturber
    perturber = Perturber(args.perturb_type, args.noise_ratio, device)

    print(f"\n{'='*60}")
    print(f"Robustness Eval: {args.perturb_type} @ {args.noise_ratio}")
    print(f"Model: {args.checkpoint}")
    print(f"{'='*60}")

    # Evaluation Loop
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, target in tqdm(data_loader_test, desc=f"Testing ({args.perturb_type})"):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # !!! APPLY PERTURBATION !!!
            images = perturber.apply(images)
            # --------------------------

            output = model(images)
            _, preds = output.max(1)

            total += target.size(0)
            correct += (preds == target).sum().item()

    accuracy = 100.0 * correct / total
    print(f"\nResult >> Noise: {args.noise_ratio} | Type: {args.perturb_type} | Accuracy: {accuracy:.4f}%")

    # Save Results
    result = {
        "model": "YaTC",
        "perturb_type": args.perturb_type,
        "noise_ratio": args.noise_ratio,
        "accuracy": accuracy,
        "dataset": os.path.basename(args.data_path.rstrip('/')),
        "checkpoint": os.path.basename(os.path.dirname(args.checkpoint))
    }
    
    output_dir = os.path.dirname(args.save_result)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.save_result, 'a') as f:
        f.write(json.dumps(result) + "\n")
    print(f"Result saved to {args.save_result}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)