#!/bin/bash
TRAIN_PATH="../data_process/datasets/train_dataset.tsv"
DEV_PATH="../data_process/datasets/valid_dataset.tsv"
TEST_PATH="../data_process/datasets/test_dataset.tsv"
OUTPUT_MODEL_PATH="../models/finetuned_model.bin"

# export CUDA_VISIBLE_DEVICES=0,2,3,6




python run_classifier.py --pretrained_model_path ../models/pre-trained_model.bin \
                                   --vocab_path ../models/encryptd_vocab.txt \
                                    --train_path $TRAIN_PATH \
                                    --dev_path $DEV_PATH \
                                    --test_path $TEST_PATH \
                                   --epochs_num 20 --batch_size 32 --embedding word_pos_seg \
                                   --encoder transformer --mask fully_visible \
                                   --seq_length 128 --learning_rate 2e-5 \
                                   --config_path ../models/bert/base_config.json \
                                   --output_model_path $OUTPUT_MODEL_PATH