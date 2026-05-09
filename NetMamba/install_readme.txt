# 执行 pip install -e .的 时候可能报错
检查CUDA版本
ls -ld /usr/local/cuda*
临时设置环境变量
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 安装causal-conv1d的时候可能报错
使用https://github.com/SmartData-Polito/Debunk_Traffic_Representation/tree/master/code/NetMamba的教程
另外注意安装的时候使用 pip install . --no-build-isolation，这样不会创建隔离环境