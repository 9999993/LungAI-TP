"""
LungAI-TP Models Package
病理图像深度学习模型

创新点：
1. SE通道注意力 - 学习哪些特征通道更重要
2. 空间注意力 - 学习关注哪些图像区域
3. 多尺度特征融合 - 融合不同尺度的特征
4. 病理感知模块 - 结合HE染色先验知识
"""

from .pathology_resnet import PathologyResNet, create_pathology_resnet

__all__ = [
    "PathologyResNet",
    "create_pathology_resnet",
]
