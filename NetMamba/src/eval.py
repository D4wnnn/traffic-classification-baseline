import argparse
import datetime
import json
import numpy as np
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import os
from torchvision import datasets, transforms
import util.misc as misc
import models_net_mamba
from engine import evaluate, evaluate_speed_test


def get_args_parser():
    parser = argparse.ArgumentParser('NetMamba fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    # Model parameters
    parser.add_argument('--model', default='flow_mamba_tiny', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--data_path', default='./ddos_datasets/flow_image', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='./output/finetune',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/finetune',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    # spatio temporal features
    parser.add_argument('--if_stat', action='store_true',
                        help='whether to use statistical features')
    # evaluation settings
    parser.add_argument('--speed_test', action='store_true')
    parser.add_argument('--byte_length', default=1600, type=int)

    return parser

def build_dataset(is_train, args):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    root = os.path.join(args.data_path, 'train' if is_train else 'test')
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

def main(args):
    misc.init_distributed_mode(args)

    # print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_val = build_dataset(is_train=False, args=args)
    print("Test Dataset Classes:", dataset_val.classes)
    print("Class to Index:", dataset_val.class_to_idx)
    labels = dataset_val.classes
    # print(dataset_val.class_to_idx)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_net_mamba.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        byte_length=args.byte_length,
    )
    # try:
    #     checkpoint = torch.load(args.resume, map_location='cpu',weights_only=False)
    #     print("Load model checkpoint from: %s" % args.resume)
    #     checkpoint_model = checkpoint['model']
    #     msg = model.load_state_dict(checkpoint_model, strict=False)
    #     # print(msg)
    # except Exception as e:
    #     print(f"Failed to load model checkpoint: {e}")
    #     print("Evaluate the model from scratch.")
    # 替换 eval.py 中原来的加载逻辑
    try:
        checkpoint = torch.load(args.resume, map_location='cpu',weights_only=False)
        print("Load model checkpoint from: %s" % args.resume)
        
        checkpoint_model = checkpoint['model']
        
        # --- 核心修复：自动去除 'module.' 前缀 ---
        new_state_dict = {}
        for k, v in checkpoint_model.items():
            # 如果 key 以 'module.' 开头，去掉它
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        checkpoint_model = new_state_dict
        # -------------------------------------

        # 尝试加载，并打印详细的不匹配信息
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("加载结果: ", msg)
        
        # 强校验：如果核心层（如 head 或 backbone）丢失，必须报错
        if len(msg.missing_keys) > 0:
            print("[警告] 发现丢失的键 (Missing Keys):", msg.missing_keys[:5], "...")
            # 检查是否丢失了关键的 head.weight，如果是，说明完全没加载上
            if any("head" in k for k in msg.missing_keys):
                raise RuntimeError("错误：分类头(head)权重未加载！请检查 Key 是否匹配。")

    except Exception as e:
        print(f"Failed to load model checkpoint: {e}")
        # 这里建议直接退出，而不是继续评估，否则结果没有意义
        exit(1)
    model.to(device)

    if args.speed_test:
        dataset_train = build_dataset(is_train=True, args=args)
        sampler_train = torch.utils.data.SequentialSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        evaluate_speed_test(data_loader_train, model, device, args)
        return

    # test_stats = evaluate(data_loader_val, model, device)
    print(f"Batch size: {args.batch_size}")
    test_stats = evaluate(data_loader_val, model, device, if_stat=args.if_stat)
    # for k, v in test_stats.items():
    #     print(f"Test {k}: {v}")
    with open(os.path.join(args.output_dir, 'test_stats.json'), 'w') as f:
        new_stats = {k: v.tolist() if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray) else v for k, v in test_stats.items()}
        json.dump(new_stats, f, indent=2)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)