import argparse
import os
import time
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from pathlib import Path
import pandas as pd
import util.misc as misc
import models_YaTC
from engine import evaluate
from util.misc import setup_seed

def get_args_parser():
    parser = argparse.ArgumentParser('YaTC evaluation script', add_help=False)
    
    # 基础参数
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--data_path', default='./data/ISCXVPN2016_MFR', type=str,
                        help='dataset path (parent directory containing train/test folders)')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--device', default='cuda',
                        help='device to use for testing')
    parser.add_argument('--seed', default=0, type=int)
    
    # 模型参数 (必须与训练时一致)
    parser.add_argument('--model', default='TraFormer_YaTC', type=str, metavar='MODEL',
                        help='Name of model to test')
    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.0 for testing)')
    
    # 权重文件路径
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='path to the checkpoint model file (.pth)')
    
    # 运行相关
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # 分布式相关 (虽然测试通常单卡，但为了兼容 engine.py 可能需要的 misc 调用)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser

def build_test_dataset(args):
    """
    构建测试数据集
    假设目录结构为: args.data_path/test
    """
    mean = [0.5]
    std = [0.5]

    # 测试集使用确定性的变换，不进行随机增强
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # 强制读取 'test' 文件夹，或者根据你的实际验证集文件夹修改此处
    root = os.path.join(args.data_path, 'test') 
    
    if not os.path.exists(root):
        raise FileNotFoundError(f"Test dataset path does not exist: {root}")
        
    dataset = datasets.ImageFolder(root, transform=transform)
    print(f"Loaded test dataset from {root}. Total samples: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    
    return dataset

def main(args):
    misc.init_distributed_mode(args)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 固定随机种子
    seed = args.seed + misc.get_rank()
    setup_seed(seed)

    # 1. 准备数据
    dataset_test = build_test_dataset(args)
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # 2. 构建模型
    print(f"Creating model: {args.model}")
    model = models_YaTC.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    # 3. 加载权重
    if os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only = False)
        
        # 处理 checkpoint 字典结构
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint
        
        # 加载参数
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print("Missing keys:", msg.missing_keys)
        print("Unexpected keys:", msg.unexpected_keys)
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # 4. 执行验证
    print(f"Start testing on {len(dataset_test)} images...")
    start_time = time.time()
    
    test_stats = evaluate(data_loader_test, model, device)

    # 5. 输出最终结果
    print(f"Testing finished.")
    print(f"Accuracy: {test_stats['acc1']:.4f}%")
    print(f"Macro F1: {test_stats['macro_f1']:.4f}")
    print(f"Recall: {test_stats['macro_rec']:.4f}")
    print(f"Precision: {test_stats['macro_pre']:.4f}")
    print("-" * 30)
    # print("Confusion Matrix:")
    # print("\n",test_stats['cm'])
    
    
    
    # ------------------ 修改开始 ------------------
    print("-" * 30)
    print("Detailed Confusion Matrix Analysis:")
    
    # 获取类别名称列表
    class_names = dataset_test.classes
    
    # 获取原始混淆矩阵 (假设是 numpy array 或 list of lists)
    cm_array = test_stats['cm']
    
    # 转换为 numpy 确保兼容性
    if isinstance(cm_array, torch.Tensor):
        cm_array = cm_array.cpu().numpy()
    cm_array = np.array(cm_array)

    # 创建 Pandas DataFrame，行是真实标签，列是预测标签
    df_cm = pd.DataFrame(cm_array, index=class_names, columns=class_names)
    
    # 为了方便大模型阅读，我们在打印前加上明确的说明
    print("\n[Confusion Matrix Table]")
    print("Rows = True Labels (Ground Truth)")
    print("Columns = Predicted Labels")
    print("-" * 20)
    
    # 使用 to_markdown() (需要 tabulate 库) 或者 to_string() 打印
    # 建议使用 to_string() 或 to_markdown()，大模型读这个非常清晰
    # try:
    #     print(df_cm.to_markdown()) 
    # except ImportError:
    #     print(df_cm.to_string())
        
    # print("-" * 30)
    
    # 同时也保存一份带标签的 CSV 到本地，方便你写论文画图
    csv_path = os.path.join(os.path.dirname(args.checkpoint), "confusion_matrix_labeled.csv")
    df_cm.to_csv(csv_path)
    print(f"Labeled confusion matrix saved to {csv_path}")
    # ------------------ 修改结束 ------------------
    
    
    
    
    # 可选：保存结果到文件
    output_dir = os.path.dirname(args.checkpoint)
    result_path = os.path.join(output_dir, "test_result.txt")
    with open(result_path, "w") as f:
        f.write(json.dumps(test_stats, cls=NumpyEncoder))
    print(f"Results saved to {result_path}")

class NumpyEncoder(json.JSONEncoder):
    """ 处理 JSON 序列化中的 Numpy 类型 """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)