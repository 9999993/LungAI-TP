"""
LungAI-TP Pathology ResNet
简化版：3个简单创新 + 1个高级创新

简单创新：
1. SE注意力 - 通道注意力
2. Spatial注意力 - 空间注意力
3. 多尺度融合 - 特征金字塔

高级创新：
4. 病理感知模块 - 结合医学先验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============ 简单创新1: SE通道注意力 ============
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    简单但有效的通道注意力
    """

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.squeeze(x).view(B, C)
        y = self.excitation(y).view(B, C, 1, 1)
        return x * y.expand_as(x)


# ============ 简单创新2: 空间注意力 ============
class SpatialAttention(nn.Module):
    """
    空间注意力模块
    学习"关注图像的哪些区域"
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention


# ============ 简单创新3: 多尺度特征融合 ============
class MultiScaleFusion(nn.Module):
    """
    多尺度特征融合模块
    融合不同感受野的特征
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        quarter = out_channels // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, quarter, 1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, quarter, 3, padding=1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, quarter, 5, padding=2),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, quarter, 1),
            nn.BatchNorm2d(quarter),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        combined = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        return self.fusion(combined)


# ============ 高级创新: 病理感知注意力 ============
class PathologyAwareModule(nn.Module):
    """
    病理感知模块
    结合医学先验知识的注意力机制
    """

    def __init__(self, in_channels=3):
        super().__init__()

        # 颜色分离：HE染色特征
        self.color_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, 1),
            nn.Softmax(dim=1),
        )

        # 核浆比估计
        self.nucleus_detector = nn.Sequential(
            nn.Conv2d(in_channels + 4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        # 纹理特征
        self.texture_conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + 4 + 1 + 4, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        color_features = self.color_conv(x)
        nucleus_map = self.nucleus_detector(torch.cat([x, color_features], dim=1))
        texture_features = self.texture_conv(x)
        combined = torch.cat([x, color_features, nucleus_map, texture_features], dim=1)
        attention = self.fusion(combined)
        return x * attention


# ============ 带注意力的ResNet块 ============
class ResBlockWithAttention(nn.Module):
    """带SE和空间注意力的残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)
        self.spatial = SpatialAttention()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)
        out = self.spatial(out)

        out += residual
        out = self.relu(out)

        return out


# ============ 完整的病理ResNet ============
class PathologyResNet(nn.Module):
    """
    专为病理图像设计的ResNet
    """

    def __init__(
        self,
        num_classes=3,
        pretrained=True,
        use_pathology_module=True,
    ):
        super().__init__()

        self.use_pathology_module = use_pathology_module

        # 病理感知模块
        if use_pathology_module:
            self.pathology_module = PathologyAwareModule(in_channels=3)

        # 加载预训练ResNet
        resnet = models.resnet50(pretrained=pretrained)

        # 基础层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 残差层（使用带注意力的块）
        self.layer1 = self._make_layer(64, 256, 3, stride=1)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)

        # 多尺度特征融合
        self.multi_scale = MultiScaleFusion(2048, 2048)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        # 初始化
        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """创建残差层"""
        layers = []
        layers.append(ResBlockWithAttention(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlockWithAttention(out_channels, out_channels, 1))
        return nn.ModuleList(layers)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _forward_layer(self, layer, x):
        """前向传播一个层"""
        for block in layer:
            x = block(x)
        return x

    def forward(self, x):
        """前向传播"""
        # 病理感知预处理
        if self.use_pathology_module:
            x = self.pathology_module(x)

        # 基础卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层
        x = self._forward_layer(self.layer1, x)
        x = self._forward_layer(self.layer2, x)
        x = self._forward_layer(self.layer3, x)
        x = self._forward_layer(self.layer4, x)

        # 多尺度融合
        x = self.multi_scale(x)

        # 全局池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)

        return x


class PathologyResNetLight(nn.Module):
    """
    轻量版病理ResNet
    减少参数量，加快训练
    """

    def __init__(self, num_classes=3, pretrained=True, use_pathology_module=True):
        super().__init__()

        self.use_pathology_module = use_pathology_module

        # 病理感知模块
        if use_pathology_module:
            self.pathology_module = PathologyAwareModule(in_channels=3)

        # 使用ResNet-18作为backbone（更轻量）
        resnet = models.resnet18(pretrained=pretrained)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # SE注意力（比完整注意力轻量）
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 添加SE注意力
        self.se1 = SEBlock(64)
        self.se2 = SEBlock(128)
        self.se3 = SEBlock(256)
        self.se4 = SEBlock(512)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 病理感知预处理
        if self.use_pathology_module:
            x = self.pathology_module(x)

        # 基础卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 残差层 + SE注意力
        x = self.se1(self.layer1(x))
        x = self.se2(self.layer2(x))
        x = self.se3(self.layer3(x))
        x = self.se4(self.layer4(x))

        # 全局池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # 分类
        x = self.classifier(x)

        return x

    def predict(self, x):
        """推理模式，返回与原有接口兼容的输出"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)

            return {
                "subtype_pred": pred_class,
                "subtype_prob": probs,
                "response_pred": (pred_class != 2).long(),  # 非正常组织为癌症
                "response_prob": torch.stack(
                    [
                        probs[:, 2],  # 正常概率
                        1 - probs[:, 2],  # 癌症概率
                    ],
                    dim=1,
                ),
            }


def create_pathology_resnet(
    num_classes=3, pretrained=True, use_pathology_module=True, light=False
):
    """创建病理ResNet模型"""
    if light:
        return PathologyResNetLight(
            num_classes=num_classes,
            pretrained=pretrained,
            use_pathology_module=use_pathology_module,
        )
    return PathologyResNet(
        num_classes=num_classes,
        pretrained=pretrained,
        use_pathology_module=use_pathology_module,
    )
