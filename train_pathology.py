"""
PathologyResNet训练脚本（GPU优化版）
使用混合精度训练加速
"""

import os
import sys
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from models import create_pathology_resnet


class PathologyDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None, augment=False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.labels_df = pd.read_csv(labels_file)

        self.augment_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )

        if self.augment:
            image = self.augment_transform(image)
        if self.transform:
            image = self.transform(image)

        return image, int(row["subtype"])


def get_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # 混合精度训练
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(train_loader), accuracy_score(all_labels, all_preds)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    for images, labels in tqdm(val_loader, desc="Validating"):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(val_loader), accuracy_score(all_labels, all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--no_pathology_module", action="store_true", help="不使用病理感知模块"
    )
    parser.add_argument("--light", action="store_true", help="使用轻量版模型")
    args = parser.parse_args()

    print("=" * 50)
    print("PathologyResNet GPU训练 (混合精度)")
    print("=" * 50)
    print(f"设备: {config.DEVICE}")
    print(
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
    )
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"使用病理感知模块: {not args.no_pathology_module}")
    print(f"使用轻量模型: {args.light}")
    print("=" * 50)

    # 数据加载器
    train_dataset = PathologyDataset(
        config.TRAIN_DIR,
        os.path.join(config.DATA_DIR, "train_labels.csv"),
        get_transforms(),
        augment=True,
    )
    val_dataset = PathologyDataset(
        config.VAL_DIR,
        os.path.join(config.DATA_DIR, "val_labels.csv"),
        get_transforms(),
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")

    # 创建模型
    model = create_pathology_resnet(
        num_classes=config.NUM_SUBTYPES,
        pretrained=True,
        use_pathology_module=not args.no_pathology_module,
        light=args.light,
    ).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 混合精度训练
    scaler = GradScaler()

    # 训练循环
    best_acc = 0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, config.DEVICE
        )
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)
        scheduler.step()

        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"保存最佳模型 (准确率: {best_acc:.4f})")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

    pd.DataFrame(history).to_csv(
        os.path.join(config.RESULTS_DIR, "history.csv"), index=False
    )
    print(f"\n训练完成! 最佳准确率: {best_acc:.4f}")


if __name__ == "__main__":
    main()
