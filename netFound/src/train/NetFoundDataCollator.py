import torch
from transformers import DataCollatorForLanguageModeling, BatchEncoding
from typing import Any, Dict, List, Optional, Union
import numpy as np
import random
from transformers.utils import requires_backends, is_torch_device
from utils import get_logger

logger = get_logger(name=__name__)


class DataCollatorWithMeta(DataCollatorForLanguageModeling):
    def __init__(self, values_clip: Optional[int] = None, swap_rate=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values_clip = values_clip
        self.swap_rate = swap_rate

    def torch_call(
            self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = {}
        burstsInEachFlow = [example["total_bursts"] for example in examples]
        maxBursts = max(burstsInEachFlow)
        for i in range(len(examples)):
            inputs = dict((k, v) for k, v in examples[i].items())
            for key in inputs.keys():
                if key == "labels" or key == "total_bursts" or key == "replacedAfter":
                    continue
                if key not in batch:
                    if key != "replacedAfter":
                        batch[key] = []
                if key == "ports":
                    batch[key].append(inputs[key] + 1)
                elif key in ("protocol", "flow_duration"):
                    batch[key].append(inputs[key])
                else:
                    batch[key].append(
                        inputs[key][: maxBursts * self.tokenizer.max_burst_length]
                    )
        for key in batch.keys():
            batch[key] = torch.Tensor(np.array(batch[key]))
            if (
                    key == "input_ids"
                    or key == "attention_masks"
                    or key == "ports"
                    or key == "protocol"
            ):
                batch[key] = torch.Tensor(batch[key]).to(torch.long)

        if self.mlm:
            batch["input_ids"], batch["labels"], batch["swappedLabels"], batch[
                "burstMetasToBeMasked"] = self.torch_mask_tokens(
                batch["input_ids"], burstsInEachFlow, self.tokenizer.max_burst_length, self.swap_rate,
                batch["protocol"], special_tokens_mask=None
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return BatchEncoding(batch)

    def swap_bursts_adjust_prob_matrix(self, input_ids, burstsInEachFlow, max_burst_length, swap_rate):
        labels = torch.from_numpy(np.array(np.random.rand(len(burstsInEachFlow)) < swap_rate, dtype=int))
        swappedIds = []
        for i in range(input_ids.shape[0]):
            if labels[i] == 1:
                burstToRep = random.randint(0, burstsInEachFlow[i] - 1)
                flowChoice = random.randint(0, input_ids.shape[0] - 1)
                if flowChoice == i:
                    flowChoice = (flowChoice + 1) % input_ids.shape[0]
                burstChoice = random.randint(0, burstsInEachFlow[flowChoice] - 1)
                swappedIds.append([i, burstToRep])
                input_ids[i][burstToRep * max_burst_length:(burstToRep + 1) * max_burst_length] = input_ids[flowChoice][burstChoice * max_burst_length:(burstChoice + 1) * max_burst_length]
        return input_ids, swappedIds, labels

    def maskMetaData(self, input_ids, burstsInEachFlow, swapped_bursts):
        maskedMetaBursts = np.full((input_ids.shape[0], max(burstsInEachFlow)), 0.3)
        for ids in swapped_bursts:
            maskedMetaBursts[ids[0]][ids[1]] = 0
        candidateFlows = np.array(
            [np.array(np.array(burstsInEachFlow) > 3, dtype=int)]).transpose()  # converting to nX1 matrix
        return torch.bernoulli(torch.from_numpy(candidateFlows * maskedMetaBursts)).bool()

    def torch_mask_tokens(self, input_ids, burstsInEachFlow, max_burst_length, swap_rate, protos, **kwargs):
        labels = input_ids.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        new_ip_ids, swappedIds, swappedLabels = self.swap_bursts_adjust_prob_matrix(input_ids, burstsInEachFlow,
                                                                                    max_burst_length, swap_rate)
        maskMetaData = self.maskMetaData(input_ids, burstsInEachFlow, swappedIds)
        for ids in swappedIds:
            probability_matrix[ids[0]][ids[1] * max_burst_length:(ids[1]) * max_burst_length] = 0
        input_ids = new_ip_ids

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
                torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
                & masked_indices
                & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return input_ids, labels, swappedLabels, maskMetaData


class DataCollatorForFlowClassification:
    def __init__(self, max_burst_length):
        self.max_burst_length = max_burst_length

    def __call__(self, examples):
        first = examples[0]
        
        # 1. 计算当前 Batch 的最大目标长度
        # total_bursts 可能是 string 或 int，安全转换
        burst_counts = []
        for e in examples:
            tb = e.get("total_bursts", 0)
            burst_counts.append(int(tb) if tb is not None else 0)
            
        maxBursts = max(burst_counts)
        target_len = maxBursts * self.max_burst_length
        
        # 2. 预处理 stats 字段 (str -> list[float])
        for i in range(len(examples)):
            if "stats" in examples[i] and isinstance(examples[i]["stats"], str):
                try:
                    examples[i]["stats"] = [
                        float(t) for t in examples[i]["stats"].strip().split()
                    ]
                except:
                    examples[i]["stats"] = [0.0] * 10 # Fallback

        batch = {}
        
        # 3. 处理 Labels, Protocol, Duration (标量)
        for key in ["labels", "protocol", "flow_duration"]:
            if key in first and first[key] is not None:
                # 统一转为 tensor
                values = [f[key] for f in examples]
                # 尝试推断类型
                if isinstance(first[key], int) or "labels" in key:
                    dtype = torch.long
                else:
                    dtype = torch.float
                batch[key] = torch.tensor(values, dtype=dtype)

        # 4. 处理序列特征 (需要 Padding 的部分)
        # 排除已处理的标量 key
        # exclude_keys = {"labels", "label_ids", "total_bursts", "protocol", "flow_duration"}
        exclude_keys = {"labels", "label_ids", "total_bursts", "protocol", "flow_duration", "burst_tokens", "directions", "counts", "stats_original"}
        
        for k, v in first.items():
            if k not in exclude_keys and v is not None and not isinstance(v, str):
                seqs = []
                for idx, f in enumerate(examples):
                    data = f[k]
                    
                    # A. 强制转换为 Python List (如果是 Tensor 或 Numpy)
                    if hasattr(data, "tolist"):
                        data = data.tolist()
                    elif isinstance(data, (torch.Tensor, np.ndarray)):
                        data = data.tolist()
                    
                    # B. 计算截断与填充
                    # 如果 data 是 stats (比如长度固定为 20)，我们不应该用 target_len (比如 90) 去截断它
                    # 只有流序列特征 (input_ids, bytes, etc.) 需要对齐到 target_len
                    # 这里做一个简单的判断：如果由 tokenizer 生成的流特征，长度通常是 max_burst_length 的倍数
                    
                    is_flow_feature = k in ["input_ids", "attention_mask", "direction", "bytes", "pkt_count", "iats"]
                    
                    if is_flow_feature:
                        # 截断
                        curr_seq = data[:target_len]
                        # 填充
                        padding_len = target_len - len(curr_seq)
                        if padding_len > 0:
                            # 补 0
                            curr_seq = list(curr_seq) + [0] * padding_len
                    else:
                        # 对于 stats 等其他特征，保持原样 (假设它们在 batch 内长度一致)
                        curr_seq = data
                        
                    seqs.append(curr_seq)

                # C. 堆叠为 Tensor
                try:
                    # input_ids 等应该是 long，其他 float
                    if k in ["input_ids", "attention_mask", "direction", "counts", "pkt_count", "bytes"]:
                        batch[k] = torch.tensor(seqs, dtype=torch.long)
                    else:
                        batch[k] = torch.tensor(seqs, dtype=torch.float)
                except ValueError as e:
                    # 打印详细错误方便调试
                    print(f"!!! Error creating tensor for key '{k}' !!!")
                    print(f"Target length: {target_len}")
                    print(f"Sequence lengths in batch: {[len(s) for s in seqs]}")
                    print(f"Example sequence 0: {seqs[0]}")
                    raise e
                
        return batch