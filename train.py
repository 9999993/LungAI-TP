"""LungAI-TP 完整训练模块
支持多任务学习: 诊断分型 + 分子标记 + 治疗响应 + 预后预测
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from model import LungAICompleteModel, MultiTaskLoss


class LungPathologyDataset(Dataset):
    """肺癌病理数据集 (完整标签版本)"""

    def __init__(self, data_dir, labels_file, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.labels_df = pd.read_csv(labels_file)

        # 数据增强
        self.augment_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
                transforms.RandomRotation(config.ROTATION_DEGREES),
                transforms.ColorJitter(
                    brightness=config.COLOR_JITTER_BRIGHTNESS,
                    contrast=config.COLOR_JITTER_CONTRAST,
                ),
            ]
        )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]

        # 加载图像
        img_path = os.path.join(self.data_dir, row["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.fromarray(
                np.random.randint(
                    0, 255, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8
                )
            )

        # 数据增强
        if self.augment:
            image = self.augment_transform(image)
        if self.transform:
            image = self.transform(image)

        # 构建标签字典
        labels = {
            # 基础诊断
            "subtype": torch.tensor(int(row["subtype"]), dtype=torch.long),
            "response": torch.tensor(int(row["response"]), dtype=torch.long),
            # 分子标记
            "molecular": {
                "egfr": torch.tensor(int(row["egfr"]), dtype=torch.long),
                "alk": torch.tensor(int(row["alk"]), dtype=torch.long),
                "kras": torch.tensor(int(row["kras"]), dtype=torch.long),
                "pdl1": torch.tensor(int(row["pdl1"]), dtype=torch.long),
            },
            # 治疗响应
            "treatment": {
                "targeted": torch.tensor(
                    int(row["targeted_response"]), dtype=torch.long
                ),
                "immunotherapy": torch.tensor(
                    int(row["immunotherapy_response"]), dtype=torch.long
                ),
                "chemotherapy": torch.tensor(
                    int(row["chemotherapy_response"]), dtype=torch.long
                ),
                "combined": torch.tensor(
                    int(row["combined_response"]), dtype=torch.long
                ),
            },
            # 预后
            "prognosis": {
                "survival_prob": torch.tensor(
                    [
                        float(row["survival_1yr"]),
                        float(row["survival_3yr"]),
                        float(row["survival_5yr"]),
                    ],
                    dtype=torch.float,
                ),
                "recurrence_risk": torch.tensor(
                    int(row["recurrence_risk"]), dtype=torch.long
                ),
            },
        }

        return image, labels


def get_transforms():
    """获取图像变换"""
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def create_dataloaders():
    """创建数据加载器"""
    train_dataset = LungPathologyDataset(
        config.TRAIN_DIR,
        os.path.join(config.DATA_DIR, "train_labels.csv"),
        get_transforms(),
        augment=True,
    )
    val_dataset = LungPathologyDataset(
        config.VAL_DIR,
        os.path.join(config.DATA_DIR, "val_labels.csv"),
        get_transforms(),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    return train_loader, val_loader


def move_labels_to_device(labels, device):
    """将标签移动到设备"""
    moved = {
        "subtype": labels["subtype"].to(device),
        "response": labels["response"].to(device),
        "molecular": {k: v.to(device) for k, v in labels["molecular"].items()},
        "treatment": {k: v.to(device) for k, v in labels["treatment"].items()},
        "prognosis": {
            "survival_prob": labels["prognosis"]["survival_prob"].to(device),
            "recurrence_risk": labels["prognosis"]["recurrence_risk"].to(device),
        },
    }
    return moved


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0

    # 收集预测结果用于计算指标
    all_subtype_preds, all_subtype_targets = [], []
    all_response_preds, all_response_targets = [], []
    all_mol_preds = {"egfr": [], "alk": [], "kras": [], "pdl1": []}
    all_mol_targets = {"egfr": [], "alk": [], "kras": [], "pdl1": []}

    for images, labels in tqdm(train_loader, desc="训练"):
        images = images.to(device)
        labels = move_labels_to_device(labels, device)

        # 前向传播
        predictions = model(images)

        # 计算损失
        loss, losses_dict = criterion(predictions, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 收集预测结果
        all_subtype_preds.extend(
            torch.argmax(predictions["subtype_pred"], 1).cpu().numpy()
        )
        all_subtype_targets.extend(labels["subtype"].cpu().numpy())
        all_response_preds.extend(
            torch.argmax(predictions["response_pred"], 1).cpu().numpy()
        )
        all_response_targets.extend(labels["response"].cpu().numpy())

        for marker in ["egfr", "alk", "kras", "pdl1"]:
            all_mol_preds[marker].extend(
                torch.argmax(predictions["molecular_pred"][marker], 1).cpu().numpy()
            )
            all_mol_targets[marker].extend(labels["molecular"][marker].cpu().numpy())

    # 计算指标
    metrics = {
        "loss": total_loss / len(train_loader),
        "subtype_acc": accuracy_score(all_subtype_targets, all_subtype_preds),
        "response_acc": accuracy_score(all_response_targets, all_response_preds),
    }

    # 分子标记准确率
    for marker in ["egfr", "alk", "kras", "pdl1"]:
        metrics[f"{marker}_acc"] = accuracy_score(
            all_mol_targets[marker], all_mol_preds[marker]
        )

    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0

    all_subtype_preds, all_subtype_targets = [], []
    all_response_preds, all_response_targets = [], []
    all_mol_preds = {"egfr": [], "alk": [], "kras": [], "pdl1": []}
    all_mol_targets = {"egfr": [], "alk": [], "kras": [], "pdl1": []}
    all_treatment_preds = {
        "targeted": [],
        "immunotherapy": [],
        "chemotherapy": [],
        "combined": [],
    }
    all_treatment_targets = {
        "targeted": [],
        "immunotherapy": [],
        "chemotherapy": [],
        "combined": [],
    }
    all_survival_preds, all_survival_targets = [], []

    for images, labels in tqdm(val_loader, desc="验证"):
        images = images.to(device)
        labels = move_labels_to_device(labels, device)

        predictions = model(images)
        loss, losses_dict = criterion(predictions, labels)
        total_loss += loss.item()

        # 收集基础诊断预测
        all_subtype_preds.extend(
            torch.argmax(predictions["subtype_pred"], 1).cpu().numpy()
        )
        all_subtype_targets.extend(labels["subtype"].cpu().numpy())
        all_response_preds.extend(
            torch.argmax(predictions["response_pred"], 1).cpu().numpy()
        )
        all_response_targets.extend(labels["response"].cpu().numpy())

        # 收集分子标记预测
        for marker in ["egfr", "alk", "kras", "pdl1"]:
            all_mol_preds[marker].extend(
                torch.argmax(predictions["molecular_pred"][marker], 1).cpu().numpy()
            )
            all_mol_targets[marker].extend(labels["molecular"][marker].cpu().numpy())

        # 收集治疗响应预测
        for treatment in ["targeted", "immunotherapy", "chemotherapy", "combined"]:
            all_treatment_preds[treatment].extend(
                torch.argmax(predictions["treatment_pred"][treatment], 1).cpu().numpy()
            )
            all_treatment_targets[treatment].extend(
                labels["treatment"][treatment].cpu().numpy()
            )

        # 收集预后预测
        all_survival_preds.extend(
            predictions["prognosis_pred"]["survival_prob"].cpu().numpy()
        )
        all_survival_targets.extend(labels["prognosis"]["survival_prob"].cpu().numpy())

    # 计算指标
    metrics = {
        "loss": total_loss / len(val_loader),
        "subtype_acc": accuracy_score(all_subtype_targets, all_subtype_preds),
        "subtype_f1": f1_score(
            all_subtype_targets, all_subtype_preds, average="weighted"
        ),
        "response_acc": accuracy_score(all_response_targets, all_response_preds),
        "response_f1": f1_score(
            all_response_targets, all_response_preds, average="weighted"
        ),
    }

    # 分子标记指标
    for marker in ["egfr", "alk", "kras", "pdl1"]:
        metrics[f"{marker}_acc"] = accuracy_score(
            all_mol_targets[marker], all_mol_preds[marker]
        )

    # 治疗响应指标
    for treatment in ["targeted", "immunotherapy", "chemotherapy", "combined"]:
        metrics[f"{treatment}_acc"] = accuracy_score(
            all_treatment_targets[treatment], all_treatment_preds[treatment]
        )

    # 预后指标 (MSE)
    metrics["survival_mse"] = mean_squared_error(
        all_survival_targets, all_survival_preds
    )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LungAI-TP 完整训练")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size

    print("=" * 60)
    print("LungAI-TP 完整模型训练")
    print("=" * 60)
    print(f"设备: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}, Batch: {config.BATCH_SIZE}")

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders()

    # 创建模型
    model = LungAICompleteModel().to(config.DEVICE)
    criterion = MultiTaskLoss().to(config.DEVICE)

    # 优化器和学习率调度器
    optimizer = optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_val_score = 0.0
    history = []
    start_time = time.time()

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.NUM_EPOCHS}")
        print("-" * 40)

        # 训练
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE
        )

        # 验证
        val_metrics = validate(model, val_loader, criterion, config.DEVICE)

        # 更新学习率
        scheduler.step()

        # 打印指标
        print(f"训练 - 损失: {train_metrics['loss']:.4f}")
        print(f"  亚型准确率: {train_metrics['subtype_acc']:.4f}")
        print(f"  响应准确率: {train_metrics['response_acc']:.4f}")

        print(f"验证 - 损失: {val_metrics['loss']:.4f}")
        print(
            f"  亚型: 准确率={val_metrics['subtype_acc']:.4f}, F1={val_metrics['subtype_f1']:.4f}"
        )
        print(
            f"  响应: 准确率={val_metrics['response_acc']:.4f}, F1={val_metrics['response_f1']:.4f}"
        )
        print(
            f"  分子标记: EGFR={val_metrics['egfr_acc']:.3f}, ALK={val_metrics['alk_acc']:.3f}, "
            f"KRAS={val_metrics['kras_acc']:.3f}, PD-L1={val_metrics['pdl1_acc']:.3f}"
        )
        print(
            f"  治疗: 靶向={val_metrics['targeted_acc']:.3f}, 免疫={val_metrics['immunotherapy_acc']:.3f}, "
            f"化疗={val_metrics['chemotherapy_acc']:.3f}"
        )
        print(f"  预后MSE: {val_metrics['survival_mse']:.4f}")

        # 综合评分 (用于选择最佳模型)
        val_score = (
            val_metrics["subtype_acc"] * 0.3
            + val_metrics["response_acc"] * 0.2
            + np.mean(
                [val_metrics[f"{m}_acc"] for m in ["egfr", "alk", "kras", "pdl1"]]
            )
            * 0.2
            + np.mean(
                [
                    val_metrics[f"{t}_acc"]
                    for t in ["targeted", "immunotherapy", "chemotherapy"]
                ]
            )
            * 0.2
            + (1 - min(val_metrics["survival_mse"], 1)) * 0.1
        )

        # 保存最佳模型
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  *** 保存最佳模型 (综合评分: {best_val_score:.4f}) ***")

        # 记录历史
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "subtype_acc": val_metrics["subtype_acc"],
                "response_acc": val_metrics["response_acc"],
                "val_score": val_score,
            }
        )

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(config.RESULTS_DIR, "training_history.csv"), index=False
    )

    total_time = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print(f"训练完成!")
    print(f"总时间: {total_time:.1f} 分钟")
    print(f"最佳综合评分: {best_val_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
