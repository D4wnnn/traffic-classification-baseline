"""
File: ET-BERT-5x128/fine-tuning/run_evaluation.py
"""
import argparse
import torch
import torch.nn as nn
import os
import sys
import json
import pandas as pd
import numpy as np

# 设置 uer 目录以便导入模块
uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.opts import finetune_opts

# ==========================================
# 1. 新增：扰动类 (Perturber)
# ==========================================
class Perturber:
    def __init__(self, perturb_type, noise_ratio, vocab_size, device):
        self.type = perturb_type
        self.ratio = noise_ratio
        self.vocab_size = vocab_size
        self.device = device
        self.pad_id = 0  # 假设 PAD ID 为 0，通常 UER/BERT 也是 0

    def apply(self, src):
        """
        src: [Batch, Seq_Len] (LongTensor)
        """
        if self.ratio <= 0 or self.type == 'none':
            return src

        # --- Byte Mask (Token Noise) ---
        # 模拟字节掩码：随机将 Token 替换为词表内的随机 ID
        if self.type == 'byte_mask':
            # 生成掩码 [Batch, Seq_Len]
            mask = torch.rand_like(src.float()) < self.ratio
            
            # 生成噪声 ID (1 到 vocab_size-1, 避开 PAD)
            noise = torch.randint(1, self.vocab_size, src.shape).to(self.device)
            
            # 保护 PAD (不扰动填充部分)
            is_content = src != self.pad_id
            # 保护 CLS (通常在 index 0) - 可选，这里简单保护非 PAD 内容
            
            final_mask = mask & is_content
            return torch.where(final_mask, noise, src)

        # --- Packet Drop (Simulated) ---
        # 模拟丢包：ET-BERT 输入通常由 5 个包拼接而成。
        # 我们将序列 Reshape 为 [Batch, 5, Chunk] 然后进行 Block Mask
        elif self.type == 'packet_drop':
            B, L = src.shape
            num_packets = 5 # 假设逻辑包数为 5
            
            # 确保可以整除，如果不能整除则截断多余部分处理 (为了 Reshape)
            chunk_size = L // num_packets
            effective_len = chunk_size * num_packets
            
            # 截取前 effective_len 部分进行处理
            src_core = src[:, :effective_len]
            src_rest = src[:, effective_len:] # 剩余部分保持不变
            
            # Reshape: [Batch, 5, Chunk]
            src_view = src_core.view(B, num_packets, chunk_size)
            
            # 生成包级掩码 [Batch, 5, 1]
            keep_prob = 1.0 - self.ratio
            mask = torch.bernoulli(torch.full((B, num_packets, 1), keep_prob)).to(self.device)
            
            # 扩展掩码 [Batch, 5, Chunk]
            mask_expanded = mask.expand_as(src_view).long()
            
            # 应用丢弃 (置为 PAD ID 0)
            src_perturbed = src_view * mask_expanded
            
            # 还原形状
            src_core_perturbed = src_perturbed.reshape(B, effective_len)
            
            # 拼回剩余部分
            if src_rest.shape[1] > 0:
                return torch.cat([src_core_perturbed, src_rest], dim=1)
            else:
                return src_core_perturbed

        return src

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
            
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        return None, logits

def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            if columns["label"] < len(line):
                label = int(line[columns["label"]])
                labels_set.add(label)
    if not labels_set:
        return 0
    return max(labels_set) + 1

def batch_loader(batch_size, src, tgt, seg):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        yield src_batch, tgt_batch, seg_batch

    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        yield src_batch, tgt_batch, seg_batch

def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            
            if "text_b" not in columns:
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            dataset.append((src, tgt, seg))
    return dataset

def evaluate(args, dataset, print_confusion_matrix=False, csv_save_path=None):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()
    
    # 初始化 Perturber
    vocab_size = len(args.tokenizer.vocab)
    perturber = Perturber(args.perturb_type, args.noise_ratio, vocab_size, args.device)

    import tqdm
    # 添加描述信息
    desc_text = f"Eval ({args.perturb_type}={args.noise_ratio})"
    
    # 转换为 DataLoader 以便使用 Tqdm 进度条 (这里手动 BatchLoader)
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    for src_batch, tgt_batch, seg_batch in tqdm.tqdm(batch_loader(batch_size, src, tgt, seg), total=total_batches, desc=desc_text):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        
        # ============================
        # !!! 核心修改：应用扰动 !!!
        # ============================
        src_batch = perturber.apply(src_batch)
        # ============================
        
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()
    
    # 指标计算
    confusion = confusion.cpu()
    eps = 1e-9
    precisions, recalls, f1s = [], [], []
    
    for i in range(confusion.size()[0]):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    
    macro_f1 = sum(f1s) / len(f1s)
    
    total_tp = confusion.diag().sum().item()
    total_samples = confusion.sum().item()
    accuracy = correct / len(dataset)
    
    label_names = [f"Label_{i}" for i in range(args.labels_num)]

    print(f"\n[Result] Type: {args.perturb_type} | Ratio: {args.noise_ratio} | Acc: {accuracy:.4f} | Macro F1: {macro_f1:.4f}")

    if print_confusion_matrix and csv_save_path:
        # 为了防止文件名冲突，将扰动参数加入文件名
        base, ext = os.path.splitext(csv_save_path)
        new_csv_path = f"{base}_{args.perturb_type}_{args.noise_ratio}{ext}"
        
        confusion_df = pd.DataFrame(confusion.numpy(), index=[f"Pred_{name}" for name in label_names], columns=[f"True_{name}" for name in label_names])
        confusion_df.to_csv(new_csv_path)

    return {
        'perturb_type': args.perturb_type,
        'noise_ratio': args.noise_ratio,
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    finetune_opts(parser)
    
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first", help="Pooling type.")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert", help="Tokenizer.")
    parser.add_argument("--load_model_path", type=str, required=True, help="Path to the fine-tuned model checkpoint.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to save the evaluation JSON results.")
    
    # --- 新增鲁棒性参数 ---
    parser.add_argument('--perturb_type', type=str, default='none', 
                        choices=['none', 'byte_mask', 'packet_drop'],
                        help='Type of perturbation attack')
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='Intensity of noise (0.0 - 1.0)')
    # ----------------------
    
    args = parser.parse_args()
    args = load_hyperparam(args)
    set_seed(args.seed)

    args.labels_num = count_labels_num(args.train_path)
    print(f"Detected {args.labels_num} labels from training set: {args.train_path}")

    args.tokenizer = str2tokenizer[args.tokenizer](args)
    model = Classifier(args)

    print(f"Loading model from: {args.load_model_path}")
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(args.load_model_path), strict=False)
    else:
        model.load_state_dict(torch.load(args.load_model_path, map_location="cpu"), strict=False)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    args.model = model

    output_dir = os.path.dirname(args.result_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 执行评估
    result_stats = evaluate(args, read_dataset(args, args.test_path), True, os.path.join(output_dir, "eval_confusion_matrix.csv"))

    # 将新结果追加到 json 文件 (支持多次写入)
    # 如果文件已存在，先读取旧内容，变成列表，再追加
    existing_data = []
    if os.path.exists(args.result_path):
        try:
            with open(args.result_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content:
                    # 尝试读取，兼容单对象或列表
                    try:
                        data = json.loads(content)
                        if isinstance(data, list):
                            existing_data = data
                        else:
                            existing_data = [data]
                    except:
                         existing_data = [] # 文件损坏或格式不对
        except:
            pass
            
    existing_data.append(result_stats)

    with open(args.result_path, mode="w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)
        
    print(f"Stats saved to {args.result_path}")

if __name__ == "__main__":
    main()