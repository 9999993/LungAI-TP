"""
真实病理图像数据准备
划分：训练集(70%) / 验证集(15%) / 测试集(15%)
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def prepare_real_data_with_test():
    """使用真实数据准备训练/验证/测试集"""

    raw_dir = os.path.join(config.DATA_DIR, "raw")

    # 类别映射
    class_mapping = {
        "lung_aca": {"subtype": 0, "name": "LUAD"},
        "lung_scc": {"subtype": 1, "name": "LUSC"},
        "lung_n": {"subtype": 2, "name": "Normal"},
    }

    # 数据划分比例
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    # 创建目录
    for split in ["train", "val", "test"]:
        for name in ["LUAD", "LUSC", "Normal"]:
            dir_path = os.path.join(config.DATA_DIR, split, name)
            os.makedirs(dir_path, exist_ok=True)
            # 清空目录
            for f in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, f))

    # 收集所有真实图像
    all_data = {}
    for raw_class, info in class_mapping.items():
        raw_class_dir = os.path.join(raw_dir, raw_class)
        images = [
            f
            for f in os.listdir(raw_class_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        np.random.shuffle(images)
        all_data[info["name"]] = {
            "images": images,
            "raw_dir": raw_class_dir,
            "subtype": info["subtype"],
        }
        print(f"{info['name']}: {len(images)} images")

    # 分配数据
    train_records = []
    val_records = []
    test_records = []

    for class_name, data in all_data.items():
        images = data["images"]
        raw_dir_path = data["raw_dir"]
        subtype = data["subtype"]
        n = len(images)

        # 计算分割点
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        print(
            f"\n{class_name}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}"
        )

        # 复制训练图像
        for i, img_name in enumerate(tqdm(train_imgs, desc=f"{class_name} Train")):
            src = os.path.join(raw_dir_path, img_name)
            dst_name = f"{class_name.lower()}_train_{i:04d}.jpg"
            dst = os.path.join(config.DATA_DIR, "train", class_name, dst_name)
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize((224, 224))
                img.save(dst)
                train_records.append(
                    {
                        "image_path": os.path.join(class_name, dst_name),
                        "subtype": subtype,
                        "response": 0 if subtype == 2 else 1,
                    }
                )
            except Exception as e:
                print(f"Error: {e}")

        # 复制验证图像
        for i, img_name in enumerate(tqdm(val_imgs, desc=f"{class_name} Val")):
            src = os.path.join(raw_dir_path, img_name)
            dst_name = f"{class_name.lower()}_val_{i:04d}.jpg"
            dst = os.path.join(config.DATA_DIR, "val", class_name, dst_name)
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize((224, 224))
                img.save(dst)
                val_records.append(
                    {
                        "image_path": os.path.join(class_name, dst_name),
                        "subtype": subtype,
                        "response": 0 if subtype == 2 else 1,
                    }
                )
            except Exception as e:
                print(f"Error: {e}")

        # 复制测试图像
        for i, img_name in enumerate(tqdm(test_imgs, desc=f"{class_name} Test")):
            src = os.path.join(raw_dir_path, img_name)
            dst_name = f"{class_name.lower()}_test_{i:04d}.jpg"
            dst = os.path.join(config.DATA_DIR, "test", class_name, dst_name)
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize((224, 224))
                img.save(dst)
                test_records.append(
                    {
                        "image_path": os.path.join(class_name, dst_name),
                        "subtype": subtype,
                        "response": 0 if subtype == 2 else 1,
                    }
                )
            except Exception as e:
                print(f"Error: {e}")

    # 保存标签文件
    train_df = (
        pd.DataFrame(train_records)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    val_df = (
        pd.DataFrame(val_records).sample(frac=1, random_state=42).reset_index(drop=True)
    )
    test_df = (
        pd.DataFrame(test_records)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    train_df.to_csv(os.path.join(config.DATA_DIR, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(config.DATA_DIR, "val_labels.csv"), index=False)
    test_df.to_csv(os.path.join(config.DATA_DIR, "test_labels.csv"), index=False)

    print("\n" + "=" * 60)
    print("数据准备完成!")
    print(f"训练集: {len(train_df)} 张")
    print(f"验证集: {len(val_df)} 张")
    print(f"测试集: {len(test_df)} 张")
    print("=" * 60)

    return train_df, val_df, test_df


if __name__ == "__main__":
    np.random.seed(42)
    prepare_real_data_with_test()
