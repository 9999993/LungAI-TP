"""LungAI-TP 数据准备模块
生成更真实的合成病理图像（增加难度）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_synthetic_pathology_image(subtype, size=224):
    """生成更真实的合成病理图像

    增加难度：
    1. 更相似的颜色分布
    2. 更复杂的纹理
    3. 更多的变异
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # 所有类型使用相似的基础颜色（更难区分）
    base_hue = np.random.randint(170, 220)  # 粉紫色范围
    base_saturation = np.random.randint(20, 40)
    base_color = np.array(
        [
            base_hue + np.random.randint(-15, 15),
            base_hue - 30 + np.random.randint(-10, 10),
            base_hue + 10 + np.random.randint(-10, 10),
        ]
    )
    base_color = np.clip(base_color, 150, 230)

    # 细胞数量范围重叠（增加难度）
    if subtype == 0:  # 腺癌
        num_cells = np.random.randint(25, 45)
        cell_density_var = 0.3
        structure_type = "glandular"
    elif subtype == 1:  # 鳞癌
        num_cells = np.random.randint(20, 40)
        cell_density_var = 0.4
        structure_type = "squamous"
    else:  # 正常
        num_cells = np.random.randint(15, 35)
        cell_density_var = 0.2
        structure_type = "normal"

    # 生成复杂的背景纹理
    for i in range(size):
        for j in range(size):
            # 更复杂的噪声模式
            noise = np.random.randint(-40, 40, 3)

            # 添加空间相关性
            spatial_noise = np.sin(i * 0.1) * np.cos(j * 0.1) * 10

            if np.random.random() < 0.25:
                # 细胞核（颜色变异更大）
                nucleus_hue = np.random.randint(60, 120)
                cell_nucleus = np.array(
                    [nucleus_hue, nucleus_hue - 20, nucleus_hue + 10]
                )
                img[i, j] = cell_nucleus + noise // 2
            else:
                img[i, j] = base_color + noise + spatial_noise

    # 添加细胞结构（更复杂的变化）
    for _ in range(num_cells):
        x, y = np.random.randint(15, size - 15, 2)

        if structure_type == "glandular":
            # 腺癌：形成腺腔结构
            s = np.random.randint(4, 10)
            # 不规则形状
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    if dx * dx + dy * dy <= s * s:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if np.random.random() < 0.7:  # 70%填充
                                img[nx, ny] = [
                                    60 + np.random.randint(-15, 15),
                                    40 + np.random.randint(-10, 10),
                                    80 + np.random.randint(-15, 15),
                                ]

        elif structure_type == "squamous":
            # 鳞癌：鳞状排列
            s = np.random.randint(3, 8)
            # 椭圆形
            for dx in range(-s, s + 1):
                for dy in range(-s // 2, s // 2 + 1):
                    if dx * dx / (s * s) + dy * dy / ((s // 2) ** 2 + 1) <= 1:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            if np.random.random() < 0.6:
                                img[nx, ny] = [
                                    70 + np.random.randint(-15, 15),
                                    50 + np.random.randint(-10, 10),
                                    90 + np.random.randint(-15, 15),
                                ]

        else:  # normal
            # 正常：规则排列
            s = np.random.randint(2, 5)
            for dx in range(-s, s + 1):
                for dy in range(-s, s + 1):
                    if dx * dx + dy * dy <= s * s:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < size and 0 <= ny < size:
                            img[nx, ny] = [
                                90 + np.random.randint(-20, 20),
                                70 + np.random.randint(-15, 15),
                                100 + np.random.randint(-20, 20),
                            ]

    # 添加更多噪声（增加难度）
    noise = np.random.normal(0, 20, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 添加模糊（模拟聚焦问题）
    if np.random.random() < 0.3:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.filter(
            ImageFilter.GaussianBlur(radius=np.random.uniform(0.5, 1.5))
        )
        img = np.array(pil_img)

    return Image.fromarray(img)


def generate_molecular_labels(subtype):
    """根据亚型生成分子标记标签"""
    if subtype == 2:  # 正常组织
        return {"egfr": 0, "alk": 0, "kras": 0, "pdl1": 0}

    if subtype == 0:  # 腺癌
        egfr = 1 if np.random.random() < 0.40 else 0
        alk = 1 if np.random.random() < 0.08 else 0
        if np.random.random() < 0.15:
            kras = 1
        elif np.random.random() < 0.10:
            kras = 2
        else:
            kras = 0
        pdl1_rand = np.random.random()
        if pdl1_rand < 0.40:
            pdl1 = 0
        elif pdl1_rand < 0.75:
            pdl1 = 1
        else:
            pdl1 = 2
    else:  # 鳞癌
        egfr = 1 if np.random.random() < 0.05 else 0
        alk = 1 if np.random.random() < 0.02 else 0
        if np.random.random() < 0.08:
            kras = 1
        elif np.random.random() < 0.05:
            kras = 2
        else:
            kras = 0
        pdl1_rand = np.random.random()
        if pdl1_rand < 0.30:
            pdl1 = 0
        elif pdl1_rand < 0.55:
            pdl1 = 1
        else:
            pdl1 = 2

    return {"egfr": egfr, "alk": alk, "kras": kras, "pdl1": pdl1}


def generate_treatment_labels(subtype, molecular):
    """根据亚型和分子标记生成治疗响应标签"""

    def get_response_weights(base_weights, molecular_boost=0.0):
        weights = base_weights.copy()
        weights[0] += molecular_boost * 0.5
        weights[1] += molecular_boost * 0.3
        weights[3] -= molecular_boost * 0.8
        weights = [max(0, w) for w in weights]
        total = sum(weights)
        return [w / total for w in weights]

    if subtype == 2:
        return {"targeted": 0, "immunotherapy": 0, "chemotherapy": 0, "combined": 0}

    egfr = molecular["egfr"]
    alk = molecular["alk"]
    kras = molecular["kras"]
    pdl1 = molecular["pdl1"]

    if egfr == 1:
        targeted_weights = get_response_weights([0.15, 0.55, 0.20, 0.10], 0.2)
    elif alk == 1:
        targeted_weights = get_response_weights([0.12, 0.58, 0.20, 0.10], 0.15)
    elif kras == 1:
        targeted_weights = get_response_weights([0.05, 0.35, 0.35, 0.25], 0.1)
    else:
        targeted_weights = get_response_weights([0.02, 0.08, 0.30, 0.60], 0)
    targeted = np.random.choice(4, p=targeted_weights)

    if pdl1 == 2:
        immuno_weights = get_response_weights([0.08, 0.37, 0.30, 0.25], 0.1)
    elif pdl1 == 1:
        immuno_weights = get_response_weights([0.05, 0.25, 0.35, 0.35], 0)
    else:
        immuno_weights = get_response_weights([0.03, 0.12, 0.35, 0.50], -0.05)
    immunotherapy = np.random.choice(4, p=immuno_weights)

    if subtype == 0:
        chemo_weights = [0.03, 0.25, 0.40, 0.32]
    else:
        chemo_weights = [0.02, 0.22, 0.38, 0.38]
    chemotherapy = np.random.choice(4, p=chemo_weights)

    combined_weights = [0.10, 0.40, 0.30, 0.20]
    combined = np.random.choice(4, p=combined_weights)

    return {
        "targeted": int(targeted),
        "immunotherapy": int(immunotherapy),
        "chemotherapy": int(chemotherapy),
        "combined": int(combined),
    }


def generate_prognosis_labels(subtype, molecular):
    """生成预后标签"""
    if subtype == 2:
        return {
            "survival_1yr": 0.98,
            "survival_3yr": 0.95,
            "survival_5yr": 0.92,
            "recurrence_risk": 0,
        }

    has_targetable = (
        molecular["egfr"] == 1 or molecular["alk"] == 1 or molecular["kras"] == 1
    )

    if has_targetable:
        survival_1yr = np.random.uniform(0.80, 0.95)
        survival_3yr = np.random.uniform(0.45, 0.70)
        survival_5yr = np.random.uniform(0.20, 0.45)
        recurrence_risk = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
    else:
        survival_1yr = np.random.uniform(0.55, 0.80)
        survival_3yr = np.random.uniform(0.20, 0.45)
        survival_5yr = np.random.uniform(0.05, 0.25)
        recurrence_risk = np.random.choice([0, 1, 2], p=[0.1, 0.4, 0.5])

    return {
        "survival_1yr": round(survival_1yr, 3),
        "survival_3yr": round(survival_3yr, 3),
        "survival_5yr": round(survival_5yr, 3),
        "recurrence_risk": int(recurrence_risk),
    }


def generate_dataset(num_samples, output_dir, dataset_type="train"):
    """生成完整数据集"""
    records = []

    num_luad = int(num_samples * 0.45)
    num_lusc = int(num_samples * 0.35)
    num_normal = num_samples - num_luad - num_lusc

    print(f"Generating {dataset_type} dataset...")
    print(f"  LUAD: {num_luad}, LUSC: {num_lusc}, Normal: {num_normal}")

    luad_dir = os.path.join(output_dir, "LUAD")
    lusc_dir = os.path.join(output_dir, "LUSC")
    normal_dir = os.path.join(output_dir, "Normal")
    os.makedirs(luad_dir, exist_ok=True)
    os.makedirs(lusc_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    def process_samples(num, subtype, subtype_name, output_subdir, desc):
        for i in tqdm(range(num), desc=desc):
            img = generate_synthetic_pathology_image(
                subtype=subtype, size=config.IMAGE_SIZE
            )
            img_filename = f"{subtype_name.lower()}_{dataset_type}_{i:04d}.jpg"
            img_path = os.path.join(output_subdir, img_filename)
            img.save(img_path)

            molecular = generate_molecular_labels(subtype)
            treatment = generate_treatment_labels(subtype, molecular)
            prognosis = generate_prognosis_labels(subtype, molecular)

            response = 0 if subtype == 2 else 1

            record = {
                "image_path": os.path.join(subtype_name, img_filename),
                "subtype": subtype,
                "response": response,
                "egfr": molecular["egfr"],
                "alk": molecular["alk"],
                "kras": molecular["kras"],
                "pdl1": molecular["pdl1"],
                "targeted_response": treatment["targeted"],
                "immunotherapy_response": treatment["immunotherapy"],
                "chemotherapy_response": treatment["chemotherapy"],
                "combined_response": treatment["combined"],
                "survival_1yr": prognosis["survival_1yr"],
                "survival_3yr": prognosis["survival_3yr"],
                "survival_5yr": prognosis["survival_5yr"],
                "recurrence_risk": prognosis["recurrence_risk"],
            }
            records.append(record)

    process_samples(num_luad, 0, "LUAD", luad_dir, "LUAD")
    process_samples(num_lusc, 1, "LUSC", lusc_dir, "LUSC")
    process_samples(num_normal, 2, "Normal", normal_dir, "Normal")

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="LungAI-TP Data Preparation (Harder)")
    parser.add_argument("--num_samples", type=int, default=3000, help="Total samples")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    print("=" * 60)
    print("LungAI-TP Data Preparation (Harder Version)")
    print("=" * 60)

    num_train = int(args.num_samples * 0.8)
    num_val = args.num_samples - num_train

    print(f"Train: {num_train}, Val: {num_val}")

    train_df = generate_dataset(num_train, config.TRAIN_DIR, "train")
    val_df = generate_dataset(num_val, config.VAL_DIR, "val")

    train_df.to_csv(os.path.join(config.DATA_DIR, "train_labels.csv"), index=False)
    val_df.to_csv(os.path.join(config.DATA_DIR, "val_labels.csv"), index=False)

    print("\n" + "=" * 60)
    print("Data generation complete!")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")


if __name__ == "__main__":
    main()
