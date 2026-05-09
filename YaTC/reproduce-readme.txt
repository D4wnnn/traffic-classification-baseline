pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install timm==0.4.5
pip install numpy==1.19.5
pip install scikit-learn==0.24.2
pip install tensorboard
pip uninstall setuptools
pip install setuptools==59.5.0
pip install sixfrom torch._six import container_abc
pip install scikit-image
pip install matplotlib

pre-train.py：
- dataset_train = datasets.ImageFolder(os.path.join(args.data_path), transform=transform_train)
- # assert timm.__version__ == "0.3.2"  # version check


除去所有的qk_scale

poe_embed.py: omega = np.arange(embed_dim // 2, dtype=np.float)--->omega = np.arange(embed_dim // 2, dtype=float)