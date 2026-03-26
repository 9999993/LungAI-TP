"""
可视化SE通道注意力机制
演示通道是怎么来的，以及如何加权
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


# 字体配置
font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
plt.rcParams["axes.unicode_minus"] = False


class SEBlock(nn.Module):
    """SE注意力模块"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.squeeze(x).view(B, C)
        weights = self.excitation(y)
        y = weights.view(B, C, 1, 1)
        return x * y, weights


def visualize_se_process():
    """可视化SE注意力的完整过程"""

    # 加载一张真实图像
    image_path = "data/train/LUAD/luad_train_2853.jpg"
    image = Image.open(image_path).convert("RGB")

    # 预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    # 使用ResNet-18的前几层来提取特征
    resnet = models.resnet18(pretrained=False)

    # 提取Conv1+BN+ReLU的输出 (64个通道)
    x = resnet.conv1(img_tensor)
    x = resnet.bn1(x)
    x = resnet.relu(x)  # (1, 64, 112, 112)

    features_before = x[0].detach().numpy()  # (64, 112, 112)

    # 应用SE注意力
    se = SEBlock(channels=64, reduction=16)
    se.eval()

    with torch.no_grad():
        features_after, channel_weights = se(x)

    features_after = features_after[0].detach().numpy()  # (64, 112, 112)
    weights = channel_weights[0].numpy()  # (64,)

    # 创建可视化
    fig = plt.figure(figsize=(18, 14))

    # 第1部分：原始图像
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.imshow(image)
    ax1.set_title("原始病理图像", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # 第2部分：说明通道来源
    ax2 = fig.add_subplot(4, 4, 2)
    ax2.text(
        0.5,
        0.5,
        "Conv1 (3→64)\n3×3卷积\n64个卷积核",
        ha="center",
        va="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightblue"),
    )
    ax2.set_title("通道来源", fontsize=12, fontweight="bold")
    ax2.axis("off")

    # 第3部分：展示几个通道
    ax3 = fig.add_subplot(4, 4, 3)
    ax3.imshow(features_before[0], cmap="viridis")
    ax3.set_title(f"通道0 (权重:{weights[0]:.3f})", fontsize=10)
    ax3.axis("off")

    ax4 = fig.add_subplot(4, 4, 4)
    ax4.imshow(features_before[1], cmap="viridis")
    ax4.set_title(f"通道1 (权重:{weights[1]:.3f})", fontsize=10)
    ax4.axis("off")

    # 第2行：展示更多通道
    for i in range(4):
        ax = fig.add_subplot(4, 4, 5 + i)
        ch_idx = [2, 10, 30, 50][i]
        ax.imshow(features_before[ch_idx], cmap="viridis")
        ax.set_title(f"通道{ch_idx} (权重:{weights[ch_idx]:.3f})", fontsize=10)
        ax.axis("off")

    # 第3行：SE处理后的通道
    for i in range(4):
        ax = fig.add_subplot(4, 4, 9 + i)
        ch_idx = [0, 1, 2, 50][i]
        ax.imshow(features_after[ch_idx], cmap="viridis")
        ax.set_title(f"SE后 通道{ch_idx}", fontsize=10)
        ax.axis("off")

    # 第4行左：通道权重分布
    ax_weights = fig.add_subplot(4, 2, 7)
    colors = ["red" if w > 0.6 else ("orange" if w > 0.4 else "blue") for w in weights]
    bars = ax_weights.bar(range(64), weights, color=colors)
    ax_weights.set_xlabel("通道索引", fontsize=12)
    ax_weights.set_ylabel("注意力权重", fontsize=12)
    ax_weights.set_title("SE通道注意力权重分布", fontsize=14, fontweight="bold")
    ax_weights.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # 标注最高和最低权重
    max_idx = np.argmax(weights)
    min_idx = np.argmin(weights)
    ax_weights.annotate(
        f"最高: {weights[max_idx]:.3f}",
        xy=(max_idx, weights[max_idx]),
        xytext=(max_idx + 5, weights[max_idx]),
        arrowprops=dict(arrowstyle="->", color="red"),
    )
    ax_weights.annotate(
        f"最低: {weights[min_idx]:.3f}",
        xy=(min_idx, weights[min_idx]),
        xytext=(min_idx + 5, weights[min_idx]),
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    # 第4行右：加权前后对比
    ax_compare = fig.add_subplot(4, 2, 8)

    # 选择几个通道对比
    compare_indices = [0, 1, 2, 50, 60]
    x_pos = np.arange(len(compare_indices))
    width = 0.35

    before_vals = [features_before[i].mean() for i in compare_indices]
    after_vals = [features_after[i].mean() for i in compare_indices]

    bars1 = ax_compare.bar(
        x_pos - width / 2, before_vals, width, label="SE前", color="steelblue"
    )
    bars2 = ax_compare.bar(
        x_pos + width / 2, after_vals, width, label="SE后", color="coral"
    )

    ax_compare.set_xlabel("通道索引", fontsize=12)
    ax_compare.set_ylabel("特征均值", fontsize=12)
    ax_compare.set_title("SE加权前后对比", fontsize=14, fontweight="bold")
    ax_compare.set_xticks(x_pos)
    ax_compare.set_xticklabels([f"Ch{i}" for i in compare_indices])
    ax_compare.legend()

    plt.suptitle("SE通道注意力机制详解", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("se_attention_detail.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("可视化已保存: se_attention_detail.png")

    # 打印详细信息
    print("\n" + "=" * 60)
    print("SE通道注意力详细信息")
    print("=" * 60)
    print(f"\n输入特征形状: (1, 64, 112, 112)")
    print(f"  - Batch: 1")
    print(f"  - 通道数: 64 (来自Conv1的64个卷积核)")
    print(f"  - 空间尺寸: 112×112")

    print(f"\n通道权重统计:")
    print(f"  - 最高权重: 通道{max_idx} = {weights[max_idx]:.4f}")
    print(f"  - 最低权重: 通道{min_idx} = {weights[min_idx]:.4f}")
    print(f"  - 平均权重: {weights.mean():.4f}")
    print(f"  - 权重>0.5的通道数: {(weights > 0.5).sum()}/64")


if __name__ == "__main__":
    visualize_se_process()
