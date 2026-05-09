python custom_pipeline.py \
  --input_root /raid/lc/datasets/processed_datasets/finetune/AES-128-GCM-processed \
  --output_root ./output_folder \
  --tokenizer_config /raid/lc/netFound/configs/TestFinetuningConfig.json \
  --workers 16  \
  --bin_dir /raid/lc/netFound/src/pre_process/packets_processing_src/build