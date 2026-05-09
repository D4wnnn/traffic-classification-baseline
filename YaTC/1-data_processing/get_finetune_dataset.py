import os
from multiprocessing import Pool, cpu_count
import argparse
from tqdm import tqdm

# 从共享模块导入核心处理函数
from pcap_processor import save_image
import numpy as np
from pcap_processor import save_cstnet_datagram_image


def process_and_save_pcap_finetune(pcap_path, input_root, output_root):
    """
    处理单个pcap文件，并根据其在输入目录中的相对路径（train/label或test/label）
    在输出目录中创建并保存为图像。
    """
    try:
        # 1. 计算pcap文件相对于输入根目录的路径
        # e.g., input_root = '/path/to/data', pcap_path = '/path/to/data/train/Chat/1.pcap'
        #       -> relative_path = 'train/Chat/1.pcap'
        relative_path = os.path.relpath(pcap_path, input_root)

        # 2. 替换文件扩展名为 .png
        # e.g., -> 'train/Chat/1.png'
        relative_path_png = os.path.splitext(relative_path)[0] + ".png"

        # 3. 构建完整的输出文件路径
        # e.g., -> output_image_path = '/path/to/output/train/Chat/1.png'
        output_image_path = os.path.join(output_root, relative_path_png)

        # 4. 创建输出文件所在的目录
        # e.g., os.makedirs('/path/to/output/train/Chat/', exist_ok=True)
        output_dir = os.path.dirname(output_image_path)
        os.makedirs(output_dir, exist_ok=True)

        # 5. 调用核心函数进行处理和保存
        save_image(pcap_path, output_image_path)

    except Exception as e:
        print(f"Failed to process {pcap_path}: {e}")


def process_cstnet_finetune_dataset(dataset_path, output_path):
    """
    处理CSTNET数据集用于微调（.npy格式）
    保持train/test结构

    Args:
        dataset_path: CSTNET数据集路径（包含.npy文件的目录）
        output_path: 输出图像的根目录
    """
    print(f"Processing CSTNET finetune dataset from: {dataset_path}")

    # 只处理train和test（微调通常不使用valid）
    for split in ["train", "valid","test"]:
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
                # 获取类别名称
                class_idx = label_to_idx[label]
                class_name = f"class_{class_idx}"

                # 创建输出目录：output_path/train/class_X/ 或 output_path/test/class_X/
                output_class_dir = os.path.join(output_path, split, class_name)
                # 定义输出文件路径
                if split == "valid":
                    output_class_dir = os.path.join(output_path,'val', class_name)
                os.makedirs(output_class_dir, exist_ok=True)
                
                output_image_path = os.path.join(output_class_dir, f"{idx}.png")

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

    print(
        f"\nCSTNET finetune dataset processing complete. Images saved in '{output_path}'."
    )


def main(dataset_path, output_path):
    """
    主函数，负责查找所有pcap文件并使用多进程进行处理。
    如果检测到CSTNET数据集，则使用特殊处理。
    """
    # 检测是否是CSTNET数据集
    is_cstnet = False
    if os.path.isdir(dataset_path):
        cstnet_files = ["x_datagram_train.npy", "x_datagram_test.npy", "y_train.npy"]
        if all(os.path.exists(os.path.join(dataset_path, f)) for f in cstnet_files):
            is_cstnet = True

    if is_cstnet:
        print("Detected CSTNET dataset format")
        process_cstnet_finetune_dataset(dataset_path, output_path)
        return

    # 原有的pcap文件处理逻辑
    pcap_files_to_process = []
    print(f"Scanning for .pcap files in '{dataset_path}'...")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".pcap"):
                pcap_files_to_process.append(os.path.join(root, file))

    if not pcap_files_to_process:
        print("No .pcap files found in the specified directory.")
        return

    print(f"Found {len(pcap_files_to_process)} .pcap files to process.")

    tasks = [
        (pcap_file, dataset_path, output_path) for pcap_file in pcap_files_to_process
    ]

    print("Processing files using multiple cores...")
    with Pool(cpu_count()) as pool:
        list(
            tqdm(pool.starmap(process_and_save_pcap_finetune, tasks), total=len(tasks))
        )

    print(f"Processing complete. Images saved in '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a fine-tuning pcap dataset and save as images, preserving the train/test/label structure."
    )

    # 微调数据集只接受一个路径
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the root dataset directory (containing train/test subfolders).",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to the output directory where structured images will be saved.",
    )

    args = parser.parse_args()

    main(args.dataset_path, args.output_path)
