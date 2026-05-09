import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import torch
import torch.distributed
import numpy as np
import utils
import random
from dataclasses import field, dataclass
from datasets.distributed import split_dataset_by_node
from typing import Optional
from copy import deepcopy
from torchinfo import summary
from torch.distributed.elastic.multiprocessing.errors import record

from transformers import (
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    EarlyStoppingCallback,
)

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
    classification_report, confusion_matrix
)

from NetFoundDataCollator import DataCollatorForFlowClassification
from NetFoundModels import NetfoundFinetuningModel, NetfoundNoPTM
from NetFoundTrainer import NetfoundTrainer
from NetfoundConfig import NetfoundConfig, NetFoundTCPOptionsConfig, NetFoundLarge
from NetfoundTokenizer import NetFoundTokenizer
from utils import ModelArguments, CommonDataTrainingArguments, freeze, verify_checkpoint, \
    load_train_test_datasets, get_90_percent_cpu_count, get_logger, init_tbwriter, update_deepspeed_config, \
    LearningRateLogCallback

random.seed(42)
logger = get_logger(name=__name__)


@dataclass
class FineTuningDataTrainingArguments(CommonDataTrainingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    num_labels: int = field(metadata={"help": "number of classes in the datasets"}, default=None)
    problem_type: Optional[str] = field(
        default=None,
        metadata={"help": "Override regression or classification task"},
    )
    p_val: float = field(
        default=0,
        metadata={
            "help": "noise rate"
        },
    )
    netfound_large: bool = field(
        default=False,
        metadata={
            "help": "Use the large configuration for netFound model"
        },
    )


def regression_metrics(p: EvalPrediction):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    return {"loss": np.mean(np.absolute((logits - label_ids)))}


# def classif_metrics(p: EvalPrediction, num_classes):
#     logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#     label_ids = p.label_ids.astype(int)
#     weighted_f1 = f1_score(
#         y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
#     )
#     weighted_prec = precision_score(
#         y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
#     )
#     weighted_recall = recall_score(
#         y_true=label_ids, y_pred=logits.argmax(axis=1), average="weighted", zero_division=0
#     )
#     accuracy = accuracy_score(y_true=label_ids, y_pred=logits.argmax(axis=1))
#     logger.warning(classification_report(label_ids, logits.argmax(axis=1), digits=5))
#     logger.warning(confusion_matrix(label_ids, logits.argmax(axis=1)))
#     if num_classes > 3:
#         logger.warning(f"top3:{top_k_accuracy_score(label_ids, logits, k=3, labels=np.arange(num_classes))}")
#     if num_classes > 5:
#         logger.warning(f"top5:{top_k_accuracy_score(label_ids, logits, k=5, labels=np.arange(num_classes))}")
#     if num_classes > 10:
#         logger.warning(f"top10:{top_k_accuracy_score(label_ids, logits, k=10, labels=np.arange(num_classes))}")
#     return {
#         "weighted_f1": weighted_f1,
#         "accuracy": accuracy,
#         "weighted_prec: ": weighted_prec,
#         "weighted_recall": weighted_recall,
#     }

def classif_metrics(p: EvalPrediction, num_classes):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    label_ids = p.label_ids.astype(int)
    
    # 计算预测结果
    preds = logits.argmax(axis=1)

    # 基础指标计算 (非常快)
    weighted_f1 = f1_score(y_true=label_ids, y_pred=preds, average="macro", zero_division=0)
    weighted_prec = precision_score(y_true=label_ids, y_pred=preds, average="macro", zero_division=0)
    weighted_recall = recall_score(y_true=label_ids, y_pred=preds, average="macro", zero_division=0)
    accuracy = accuracy_score(y_true=label_ids, y_pred=preds)

    # ================= 关键修改 =================
    # 1. 获取当前进程的 rank，防止多进程重复打印
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # 2. 只有主进程 (Rank 0) 或者是单卡模式才打印，且不要打印那个巨大的 report
    if local_rank <= 0:
        # 只打印简短的 Summary，绝对不要打印 confusion_matrix (这东西一旦大了会直接卡死 IO)
        logger.warning(f"Accuracy: {accuracy:.4f}, F1: {weighted_f1:.4f}")
        
        # 如果你非要看 Top-K，请保留，但也只在 Rank 0 打印
        if num_classes > 3:
             logger.warning(f"top3: {top_k_accuracy_score(label_ids, logits, k=3, labels=np.arange(num_classes))}")
    # ===========================================

    return {
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "weighted_prec": weighted_prec,  # 注意修正了原本代码里 key 的冒号 typo
        "weighted_recall": weighted_recall,
    }

@record
def main():
    parser = HfArgumentParser(
        (ModelArguments, FineTuningDataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    utils.LOGGING_LEVEL = training_args.get_process_log_level()

    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"training_args: {training_args}")

    train_dataset, test_dataset = load_train_test_datasets(logger, data_args)
    # ================= 插入开始: 自动修复字符串标签 =================
    # 检查第一个样本的 label 是否为字符串
    sample_label = train_dataset[0]['labels'] if len(train_dataset) > 0 else None
    if isinstance(sample_label, str):
        logger.warning(f"Detected string labels (e.g., '{sample_label}'). Converting to integer IDs...")
        
        # 1. 收集所有唯一的标签（合并训练集和测试集以保证一致性）
        unique_labels = set(train_dataset['labels'])
        if test_dataset is not None:
            unique_labels |= set(test_dataset['labels'])
        
        label_list = sorted(list(unique_labels))
        label2id = {l: i for i, l in enumerate(label_list)}
        id2label = {i: l for i, l in enumerate(label_list)}
        
        logger.info(f"Automatically detected {len(label_list)} classes.")
        logger.info(f"Label mapping: {label2id}")
        
        # 2. 更新 data_args.num_labels，防止 bash 脚本里写的数量不对导致报错
        data_args.num_labels = len(label_list)
        
        # 3. 定义映射函数
        def encode_labels(example):
            example['labels'] = label2id[example['labels']]
            return example

        # 4. 应用映射
        logger.info("Applying label encoding to datasets...")
        train_dataset = train_dataset.map(encode_labels)
        if test_dataset is not None:
            test_dataset = test_dataset.map(encode_labels)
    # ================= 插入结束 =================
    # if "WORLD_SIZE" in os.environ:
    #     train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    #     test_dataset = split_dataset_by_node(test_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))

    config = NetFoundTCPOptionsConfig if data_args.tcpoptions else NetfoundConfig
    config = config(
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        hidden_size=model_args.hidden_size,
        no_meta=data_args.no_meta,
        flat=data_args.flat,
    )
    if data_args.netfound_large:
        config.hidden_size = NetFoundLarge().hidden_size
        config.num_hidden_layers = NetFoundLarge().num_hidden_layers
        config.num_attention_heads = NetFoundLarge().num_attention_heads

    config.pretraining = False
    config.num_labels = data_args.num_labels
    config.problem_type = data_args.problem_type
    testingTokenizer = NetFoundTokenizer(config=config)

    training_config = deepcopy(config)
    training_config.p = data_args.p_val
    training_config.limit_bursts = data_args.limit_bursts
    trainingTokenizer = NetFoundTokenizer(config=training_config)
    additionalFields = None

    if "WORLD_SIZE" in os.environ and training_args.local_rank > 0 and not data_args.streaming:
        logger.warning("Waiting for main process to perform the mapping")
        torch.distributed.barrier()

    params = {
        "batched": True
    }
    if not data_args.streaming:
        params['num_proc'] = data_args.preprocessing_num_workers or get_90_percent_cpu_count()
    train_dataset = train_dataset.map(function=trainingTokenizer, **params)
    test_dataset = test_dataset.map(function=testingTokenizer, **params)

    if "WORLD_SIZE" in os.environ and training_args.local_rank == 0 and not data_args.streaming:
        logger.warning("Loading results from main process")
        torch.distributed.barrier()

    data_collator = DataCollatorForFlowClassification(config.max_burst_length)
    if model_args.model_name_or_path is not None and os.path.exists(
            model_args.model_name_or_path
    ):
        logger.warning(f"Using weights from {model_args.model_name_or_path}")
        model = freeze(NetfoundFinetuningModel.from_pretrained(
            model_args.model_name_or_path, config=config
        ), model_args)
    elif model_args.no_ptm:
        model = NetfoundNoPTM(config=config)
    else:
        model = freeze(NetfoundFinetuningModel(config=config), model_args)
    if training_args.local_rank == 0:
        summary(model)

    # metrics
    problem_type = data_args.problem_type
    if problem_type == "regression":
        compute_metrics = regression_metrics
    else:
        compute_metrics = lambda p: classif_metrics(p, data_args.num_labels)

    trainer = NetfoundTrainer(
        model=model,
        extraFields=additionalFields,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=testingTokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
        data_collator=data_collator,
    )
    init_tbwriter(training_args.output_dir)
    trainer.add_callback(LearningRateLogCallback(utils.TB_WRITER))
    utils.start_gpu_logging(training_args.output_dir)
    utils.start_cpu_logging(training_args.output_dir)

    verify_checkpoint(logger, training_args)

    if training_args.do_train:
        logger.warning("*** Train ***")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()  # doesn't store tokenizer
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.warning("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
