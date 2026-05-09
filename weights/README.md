# Centralized Weights

This directory stores local model weights that are intentionally not committed
to GitHub.

The baseline code still uses its original paths. Those paths are lightweight
relative symlinks pointing into this directory, so no script path changes are
needed after cloning.

Expected layout:

```text
weights/
  ET-BERT-5x128/pre-trained_model.bin
  NetMamba/pre-train.pth
  TrafficFormer-5x128/nomoe_bertflow_pre-trained_model.bin-120000
  YaTC/YaTC_pretrained_model.pth
```

After cloning on a new machine, copy this `weights/` directory to the repository
root. Then check links with:

```bash
find TrafficFormer-5x128 ET-BERT-5x128 YaTC NetMamba -xtype l -print
```

No output means the links are valid.
