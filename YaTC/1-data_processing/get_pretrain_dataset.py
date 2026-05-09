import os
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm

# 从共享模块导入核心处理函数
from pcap_processor import save_image
import numpy as np
from pcap_processor import save_cstnet_datagram_image

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


def process_and_save_pcap(pcap_path, output_root):
    """
    处理单个pcap文件并将其保存为图像。
    类别名称由pcap文件的直接父目录决定。
    """
    try:
        # 提取类别名称 (pcap文件所在的文件夹名)
        class_name = os.path.basename(os.path.dirname(pcap_path))

        # 提取原始文件名（不含扩展名），用于生成新的png文件名
        pcap_filename_base = os.path.splitext(os.path.basename(pcap_path))[0]

        # 创建输出目录，例如: output_path/Chat/
        output_class_dir = os.path.join(output_root, class_name)
        os.makedirs(output_class_dir, exist_ok=True)

        # 定义最终的图片保存路径
        output_image_path = os.path.join(output_class_dir, f"{pcap_filename_base}.png")

        # 调用核心函数进行处理和保存
        save_image(pcap_path, output_image_path)
    except Exception as e:
        print(f"Failed to process {pcap_path}: {e}")


def process_cstnet_dataset(dataset_path, output_path):
    """
    处理CSTNET数据集（.npy格式）

    Args:
        dataset_path: CSTNET数据集路径（包含.npy文件的目录）
        output_path: 输出图像的根目录
    """
    print(f"Processing CSTNET dataset from: {dataset_path}")

    # 处理三个split
    for split in ["train", "test", "valid"]:
        print(f"\nProcessing {split} split...")

        try:
            # 加载数据
            datagram_file = os.path.join(dataset_path, f"x_datagram_{split}.npy")
            label_file = os.path.join(dataset_path, f"y_{split}.npy")

            if not os.path.exists(datagram_file) or not os.path.exists(label_file):
                print(f"Skipping {split}: files not found")
                continue

            x_datagram = np.load(datagram_file, allow_pickle=True)
            y_labels = np.load(label_file)

            print(f"  Loaded {len(x_datagram)} samples")

            # 创建标签到索引的映射
            unique_labels = np.unique(y_labels)
            label_to_idx = {
                label: idx for idx, label in enumerate(sorted(unique_labels))
            }

            # 创建任务列表
            tasks = []
            for idx, (datagram_str, label) in enumerate(zip(x_datagram, y_labels)):
                # 获取类别名称（使用标签索引）
                class_idx = label_to_idx[label]
                class_name = f"class_{class_idx}"

                # 创建输出目录
                output_class_dir = os.path.join(output_path, class_name)
                os.makedirs(output_class_dir, exist_ok=True)

                # 定义输出文件路径
                output_image_path = os.path.join(output_class_dir, f"{split}_{idx}.png")

                tasks.append((datagram_str, output_image_path))

            # 使用多进程处理
            print(f"  Processing {len(tasks)} samples with {cpu_count()} cores...")
            with Pool(cpu_count()) as pool:
                list(
                    tqdm(
                        pool.starmap(save_cstnet_datagram_image, tasks),
                        total=len(tasks),
                    )
                )

            print(f"  Completed {split} split")

        except Exception as e:
            print(f"Error processing {split} split: {e}")

    print(f"\nCSTNET dataset processing complete. Images saved in '{output_path}'.")


def main(dataset_paths, output_path):
    """
    主函数，负责查找所有pcap文件并使用多进程进行处理。
    如果检测到CSTNET数据集，则使用特殊处理。
    """
    # 检测是否是CSTNET数据集（检查第一个路径）
    first_path = dataset_paths[0] if isinstance(dataset_paths, list) else dataset_paths

    # 检查是否包含CSTNET特征文件
    is_cstnet = False
    if os.path.isdir(first_path):
        cstnet_files = ["x_datagram_train.npy", "x_datagram_test.npy", "y_train.npy"]
        if all(os.path.exists(os.path.join(first_path, f)) for f in cstnet_files):
            is_cstnet = True

    if is_cstnet:
        print("Detected CSTNET dataset format")
        process_cstnet_dataset(first_path, output_path)
        return

    # 原有的pcap文件处理逻辑
    pcap_files_to_process = []
    print("Scanning for .pcap files...")
    for path in dataset_paths:
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".pcap"):
                    pcap_files_to_process.append(os.path.join(root, file))

    if not pcap_files_to_process:
        print("No .pcap files found in the specified directories.")
        return

    print(f"Found {len(pcap_files_to_process)} .pcap files to process.")

    tasks = [(pcap_file, output_path) for pcap_file in pcap_files_to_process]

    print("Processing files using multiple cores...")
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.starmap(process_and_save_pcap, tasks), total=len(tasks)))

    print(f"Processing complete. Images saved in '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively process pcap files from multiple directories and save them as images."
    )

    # 允许输入一个或多个路径
    parser.add_argument(
        "--dataset_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to the dataset directories.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory where images will be saved.",
    )

    args = parser.parse_args()

    # 直接使用解析后的参数列表
    main(args.dataset_paths, args.output_path)
