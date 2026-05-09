python3 preprocess_data.py \
    --input_folder /raid/lc/netFound/scripts/netfound_ready_dataset/train_data \
    --action finetune \
    --tokenizer_config /raid/lc/netFound/configs/TestFinetuningConfig.json \
    --combined
python3 preprocess_data.py \
    --input_folder /raid/lc/netFound/scripts/netfound_ready_dataset/val_data \
    --action finetune \
    --tokenizer_config /raid/lc/netFound/configs/TestFinetuningConfig.json \
    --combined

python3 preprocess_data.py \
    --input_folder /raid/lc/netFound/scripts/netfound_ready_dataset/test_data \
    --action finetune \
    --tokenizer_config /raid/lc/netFound/configs/TestFinetuningConfig.json \
    --combined