import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- 配置部分 ----------------
# 结果根目录
OUTPUT_ROOT = "./output"
# 需要遍历的 GPU 配置
GPU_CONFIGS = [1, 2, 4, 8]
# 需要提取的指标 (JSON key -> 显示名称)
METRICS_MAP = {
    "acc1": "Accuracy",
    "macro_pre": "Precision",
    "macro_rec": "Recall",
    "macro_f1": "F1_Score"
}
# ----------------------------------------

def load_data():
    data = []
    
    # 遍历所有定义的 GPU 数量文件夹
    for n_gpu in GPU_CONFIGS:
        gpu_dir = os.path.join(OUTPUT_ROOT, f"{n_gpu}_gpus")
        
        if not os.path.exists(gpu_dir):
            print(f"Warning: 目录不存在 {gpu_dir}，跳过。")
            continue
            
        # 遍历该 GPU 配置下的所有数据集文件夹
        # 假设结构: output/4_gpus/VPN-service/test_result.txt
        dataset_paths = glob.glob(os.path.join(gpu_dir, "*"))
        
        for ds_path in dataset_paths:
            if not os.path.isdir(ds_path):
                continue
                
            dataset_name = os.path.basename(ds_path)
            result_file = os.path.join(ds_path, "test_result.txt")
            
            if not os.path.exists(result_file):
                print(f"Checking: {dataset_name} ({n_gpu} GPUs) -> 结果文件缺失")
                continue
            
            # 读取并解析 JSON
            try:
                with open(result_file, 'r') as f:
                    content = f.read()
                    # 某些情况下文件可能包含非 JSON 内容，简单清理
                    if not content.strip(): continue
                    stats = json.loads(content)
                    
                # 提取需要的指标
                row = {
                    "Dataset": dataset_name,
                    "GPU_Count": n_gpu,
                }
                
                for key, alias in METRICS_MAP.items():
                    val = stats.get(key, 0.0)
                    # 转换为百分比，保留4位小数 (根据你的样例 acc1 是 0.97 这种小数)
                    row[alias] = round(val * 100, 2)
                
                data.append(row)
                
            except json.JSONDecodeError:
                print(f"Error: 无法解析 {result_file}")
    
    return pd.DataFrame(data)

def plot_metrics(df):
    """绘制折线图：展示不同 GPU 数量下各指标的变化趋势"""
    if df.empty:
        print("没有数据，无法绘图。")
        return

    # 获取所有数据集名称
    datasets = df["Dataset"].unique()
    metrics = list(METRICS_MAP.values())
    
    # 设置绘图风格
    sns.set_theme(style="whitegrid")
    
    # 创建画布：每个指标一个子图
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # 绘制折线图: X轴=GPU数量, Y轴=指标数值, Hue=数据集
        sns.lineplot(
            data=df, 
            x="GPU_Count", 
            y=metric, 
            hue="Dataset", 
            marker="o", 
            ax=ax,
            palette="tab10"
        )
        
        ax.set_title(f"{metric} vs GPU Count", fontsize=14)
        ax.set_xlabel("Number of GPUs (Batch Size Scale)", fontsize=12)
        ax.set_ylabel(f"{metric} (%)", fontsize=12)
        ax.set_xticks(GPU_CONFIGS) # 强制显示 1, 2, 4, 8
        
        # 只在第一个图显示图例，避免拥挤
        if idx == 0:
            ax.legend(title="Dataset", loc='lower right', fontsize='small')
        else:
            ax.get_legend().remove()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_ROOT, "analysis_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\n[Success] 可视化图表已保存至: {plot_path}")

def main():
    print("开始分析实验结果...")
    df = load_data()
    
    if df.empty:
        print("未找到任何有效的 test_result.txt 文件。请检查路径。")
        return

    # 1. 保存汇总 CSV
    # 按照数据集和 GPU 数量排序
    df = df.sort_values(by=["Dataset", "GPU_Count"])
    csv_path = os.path.join(OUTPUT_ROOT, "analysis_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Success] 汇总表格已保存至: {csv_path}")

    # 2. 打印 F1 Score 的透视表 (Pivot Table) 供终端快速查看
    print("\n====== F1 Score Summary (Unit: %) ======")
    pivot_f1 = df.pivot(index="Dataset", columns="GPU_Count", values="F1_Score")
    print(pivot_f1)
    
    print("\n====== Accuracy Summary (Unit: %) ======")
    pivot_acc = df.pivot(index="Dataset", columns="GPU_Count", values="Accuracy")
    print(pivot_acc)

    # 3. 绘图
    plot_metrics(df)

if __name__ == "__main__":
    main()