import random
import os
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# def set_seed(seed=42):
#     # 1. Python built-in random
#     random.seed(seed)
    
#     # 2. Numpy
#     np.random.seed(seed)
    
#     # 3. PyTorch CPU & GPU
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # 如果是多卡训练，这一步很重要
    
#     # 4. 强制 CUDA 确定性算法 (关键!)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False  # 必须改为 False，牺牲一点速度换取复现性
    
#     # 5. 避免 Hash 随机化
#     os.environ['PYTHONHASHSEED'] = str(seed)
    
#     print(f"[Info] Random seed set to {seed}, deterministic mode enabled.")