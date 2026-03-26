"""
使用真实病理图像数据准备训练集
原始数据来源：data/raw/
- lung_aca: 肺腺癌 (5000张)
- lung_scc: 肺鳞癌 (5000张)
- lung_n: 正常 (5000张)
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


def prepare_real_data(num_train_per_class=1200, num_val_per_class=300):
    """使用真实数据准备训练集"""

    raw_dir = os.path.join(config.DATA_DIR, "raw")

    # 类别映射
    class_mapping = {
        "lung_aca": {"subtype": 0, "name": "LUAD"},
        "lung_scc": {"subtype": 1, "name": "LUSC"},
        "lung_n": {"subtype": 2, "name": "Normal"},
    }

    # 清空并重建目录
    for split in ["train", "val"]:
        for name in ["LUAD", "LUSC", "Normal"]:
            dir_path = os.path.join(config.DATA_DIR, split, name)
            os.makedirs(dir_path, exist_ok=True)

    # 收集所有真实图像
    all_images = {}
    for raw_class, info in class_mapping.items():
        raw_class_dir = os.path.join(raw_dir, raw_class)
        images = [
            f
            for f in os.listdir(raw_class_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        np.random.shuffle(images)
        all_images[info["name"]] = {
            "images": images,
            "raw_dir": raw_class_dir,
            "subtype": info["subtype"],
        }
        print(f"{info['name']}: {len(images)} images")

    # 分配训练/验证集
    train_records = []
    val_records = []

    for class_name, data in all_images.items():
        images = data["images"]
        raw_dir_path = data["raw_dir"]
        subtype = data["subtype"]

        # 分割数据
        train_images = images[:num_train_per_class]
        val_images = images[
            num_train_per_class : num_train_per_class + num_val_per_class
        ]

        # 复制训练图像
        print(f"\nProcessing {class_name} training images...")
        for img_name in tqdm(train_images):
            src = os.path.join(raw_dir_path, img_name)
            dst_name = f"{class_name.lower()}_train_{len(train_images):04d}.jpg"
            dst = os.path.join(config.TRAIN_DIR, class_name, dst_name)

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
                print(f"Error processing {src}: {e}")

        # 复制验证图像
        print(f"Processing {class_name} validation images...")
        for img_name in tqdm(val_images):
            src = os.path.join(raw_dir_path, img_name)
            dst_name = f"{class_name.lower()}_val_{len(val_images):04d}.jpg"
            dst = os.path.join(config.VAL_DIR, class_name, dst_name)

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
                print(f"Error processing {src}: {e}")

    # 保存标签文件
    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    # 打乱顺序
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df.to_csv(os.path.join(config.DATA_DIR, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(config.DATA_DIR, "val_labels.csv"), index=False)

    print("\n" + "=" * 60)
    print("Real data preparation complete!")
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    print("=" * 60)

    return train_df, val_df


if __name__ == "__main__":
    np.random.seed(42)
    prepare_real_data()
