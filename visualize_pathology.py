"""
可视化病理感知模块处理结果
使用中文标签，三个类别对比
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm

import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.pathology_resnet import PathologyAwareModule


# 查找中文字体
def find_chinese_font():
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None


# 配置字体
font_path = find_chinese_font()
if font_path:
    print(f"使用字体: {font_path}")
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    print("未找到中文字体，尝试安装...")
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


def process_single_image(image_path, pathology_module):
    """处理单张图像"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        color_features = pathology_module.color_conv(img_tensor)
        nucleus_map = pathology_module.nucleus_detector(
            torch.cat([img_tensor, color_features], dim=1)
        )
        texture_features = pathology_module.texture_conv(img_tensor)
        output = pathology_module(img_tensor)

    return {
        "original": np.array(image),
        "color": color_features[0].numpy(),
        "nucleus": nucleus_map[0, 0].numpy(),
        "texture": texture_features[0].numpy(),
        "output": output[0].permute(1, 2, 0).numpy(),
    }


def visualize_comparison():
    """可视化三个类别对比"""

    pathology_module = PathologyAwareModule(in_channels=3)
    pathology_module.eval()

    categories = [
        ("data/train/LUAD", "LUAD", "肺腺癌"),
        ("data/train/LUSC", "LUSC", "肺鳞癌"),
        ("data/train/Normal", "Normal", "正常组织"),
    ]

    results = []
    for dir_path, code, name in categories:
        images = os.listdir(dir_path)
        if images:
            image_path = os.path.join(dir_path, images[0])
            print(f"处理 {name}: {image_path}")
            result = process_single_image(image_path, pathology_module)
            result["name"] = name
            result["code"] = code
            results.append(result)

    # 中文标签
    row_labels = ["肺腺癌 (LUAD)", "肺鳞癌 (LUSC)", "正常组织 (Normal)"]
    col_titles = ["原始图像", "细胞核", "核浆比图", "PAAM输出"]
    component_names = ["细胞核", "细胞质", "基质", "背景"]

    # 3x4 对比图
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    for row, result in enumerate(results):
        axes[row, 0].imshow(result["original"])
        axes[row, 0].set_title(col_titles[0], fontsize=12, fontweight="bold")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=11, fontweight="bold")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(result["color"][0], cmap="Blues")
        axes[row, 1].set_title(
            f"{col_titles[1]}\n(均值: {result['color'][0].mean():.3f})", fontsize=10
        )
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        axes[row, 2].imshow(result["nucleus"], cmap="hot")
        axes[row, 2].set_title(
            f"{col_titles[2]}\n(均值: {result['nucleus'].mean():.3f})", fontsize=10
        )
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

        axes[row, 3].imshow(np.clip(result["output"], 0, 1))
        axes[row, 3].set_title(col_titles[3], fontsize=10)
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])

    plt.suptitle("病理感知模块 - 三类对比", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("pathology_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n对比图已保存: pathology_comparison.png")

    # 每个类别的详细分析
    for result in results:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        axes[0, 0].imshow(result["original"])
        axes[0, 0].set_title("原始图像", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        for i in range(3):
            axes[0, i + 1].imshow(result["color"][i], cmap="hot")
            axes[0, i + 1].set_title(component_names[i], fontsize=10)
            axes[0, i + 1].axis("off")

        axes[1, 0].imshow(result["nucleus"], cmap="hot")
        axes[1, 0].set_title("核浆比图", fontsize=11, fontweight="bold")
        axes[1, 0].axis("off")

        for i in range(3):
            axes[1, i + 1].imshow(result["texture"][i], cmap="viridis")
            axes[1, i + 1].set_title(f"纹理特征 {i + 1}", fontsize=10)
            axes[1, i + 1].axis("off")

        plt.suptitle(
            f"病理感知模块分析: {result['name']}", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(f"pathology_{result['code']}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"详细分析已保存: pathology_{result['code']}.png")

    # 统计对比
    print("\n" + "=" * 60)
    print("各类别统计对比")
    print("=" * 60)
    print(f"{'类别':<12} {'细胞核均值':<12} {'核浆比均值':<12} {'高核浆比占比':<12}")
    print("-" * 48)
    for result in results:
        nucleus_mean = result["color"][0].mean()
        nc_ratio = result["nucleus"].mean()
        high_ratio = (result["nucleus"] > 0.5).mean() * 100
        print(
            f"{result['code']:<12} {nucleus_mean:<12.4f} {nc_ratio:<12.4f} {high_ratio:<12.1f}%"
        )


if __name__ == "__main__":
    visualize_comparison()
