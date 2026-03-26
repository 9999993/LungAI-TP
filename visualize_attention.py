"""
可视化创新点效果
展示注意力图、特征图等
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.pathology_resnet import PathologyAwareModule, SEBlock, SpatialAttention


def visualize_pathology_module(image_path=None):
    """可视化病理感知模块的效果"""

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载图像或创建随机图像
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        # 创建模拟的病理图像
        image = create_synthetic_pathology_image()

    # 预处理
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    img_tensor = transform(image).unsqueeze(0)

    # 创建病理感知模块
    pathology_module = PathologyAwareModule(in_channels=3)
    pathology_module.eval()

    # 前向传播
    with torch.no_grad():
        # 获取中间结果
        color_features = pathology_module.color_conv(img_tensor)
        nucleus_map = pathology_module.nucleus_detector(
            torch.cat([img_tensor, color_features], dim=1)
        )
        texture_features = pathology_module.texture_conv(img_tensor)
        output = pathology_module(img_tensor)

    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 原图
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("原始图像", fontsize=12)
    axes[0, 0].axis("off")

    # 颜色分离
    for i, name in enumerate(["细胞核", "细胞质", "基质", "空白"]):
        axes[0, i].imshow(color_features[0, i].numpy(), cmap="hot")
        axes[0, i].set_title(f"颜色: {name}", fontsize=12)
        axes[0, i].axis("off")

    # 核浆比图
    axes[1, 0].imshow(nucleus_map[0, 0].numpy(), cmap="hot")
    axes[1, 0].set_title("核浆比图", fontsize=12)
    axes[1, 0].axis("off")

    # 纹理特征
    for i in range(3):
        axes[1, i + 1].imshow(texture_features[0, i].numpy(), cmap="viridis")
        axes[1, i + 1].set_title(f"纹理特征 {i + 1}", fontsize=12)
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.savefig("pathology_module_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("可视化已保存到 pathology_module_visualization.png")


def create_synthetic_pathology_image():
    """创建模拟的病理图像"""
    img = np.zeros((224, 224, 3), dtype=np.uint8)

    # 背景（粉色细胞质）
    img[:, :] = [220, 180, 200]

    # 添加细胞核（深紫色）
    np.random.seed(42)
    for _ in range(30):
        x, y = np.random.randint(20, 204, 2)
        r = np.random.randint(5, 15)
        y_grid, x_grid = np.ogrid[-r : r + 1, -r : r + 1]
        mask = x_grid**2 + y_grid**2 <= r**2
        x_start, x_end = max(0, x - r), min(224, x + r + 1)
        y_start, y_end = max(0, y - r), min(224, y + r + 1)
        img[x_start:x_end, y_start:y_end][
            mask[: x_end - x_start, : y_end - y_start]
        ] = [80, 40, 100]

    # 添加腺腔结构（白色区域）
    for _ in range(5):
        x, y = np.random.randint(40, 184, 2)
        r = np.random.randint(15, 30)
        y_grid, x_grid = np.ogrid[-r : r + 1, -r : r + 1]
        mask = x_grid**2 + y_grid**2 <= r**2
        x_start, x_end = max(0, x - r), min(224, x + r + 1)
        y_start, y_end = max(0, y - r), min(224, y + r + 1)
        img[x_start:x_end, y_start:y_end][
            mask[: x_end - x_start, : y_end - y_start]
        ] = [240, 230, 235]

    return Image.fromarray(img)


def visualize_se_attention():
    """可视化SE注意力效果"""

    # 创建SE模块
    se = SEBlock(channels=64, reduction=16)
    se.eval()

    # 创建随机特征图
    x = torch.randn(1, 64, 28, 28)

    # 获取注意力权重
    with torch.no_grad():
        B, C, H, W = x.shape
        y = se.squeeze(x).view(B, C)
        weights = se.excitation(y)

    # 可视化
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.bar(range(64), weights[0].numpy())
    plt.xlabel("通道索引")
    plt.ylabel("注意力权重")
    plt.title("SE通道注意力权重")

    plt.subplot(1, 3, 2)
    plt.imshow(x[0, 0].numpy(), cmap="viridis")
    plt.title("输入特征 (通道0)")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    output = se(x)
    plt.imshow(output[0, 0].numpy(), cmap="viridis")
    plt.title("SE加权后 (通道0)")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("se_attention_visualization.png", dpi=150)
    plt.show()

    print("SE注意力可视化已保存到 se_attention_visualization.png")


if __name__ == "__main__":
    print("=" * 50)
    print("创新点可视化")
    print("=" * 50)

    print("\n1. 可视化病理感知模块...")
    visualize_pathology_module()

    print("\n2. 可视化SE注意力...")
    visualize_se_attention()

    print("\n可视化完成!")
