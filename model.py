import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class TaskHead(nn.Module):
    """通用任务头"""

    def __init__(
        self, in_features=2048, hidden_dim=256, num_classes=2, dropout_rate=0.3
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class AttentionFusion(nn.Module):
    """注意力融合层"""

    def __init__(self, feature_dim=2048, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)


class MolecularPredictor(nn.Module):
    """分子标记预测模块
    预测EGFR, ALK, KRAS, PD-L1状态
    """

    def __init__(self, in_features=2048, hidden_dim=256):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True)
        )
        self.egfr_head = nn.Linear(hidden_dim, config.NUM_EGFR_CLASSES)
        self.alk_head = nn.Linear(hidden_dim, config.NUM_ALK_CLASSES)
        self.kras_head = nn.Linear(hidden_dim, config.NUM_KRAS_CLASSES)
        self.pdl1_head = nn.Linear(hidden_dim, config.NUM_PDL1_CLASSES)

    def forward(self, x):
        shared = self.shared_layer(x)
        return {
            "egfr": self.egfr_head(shared),
            "alk": self.alk_head(shared),
            "kras": self.kras_head(shared),
            "pdl1": self.pdl1_head(shared),
        }


class TreatmentPredictor(nn.Module):
    """治疗响应预测模块
    预测靶向治疗、免疫治疗、化疗、联合治疗的响应
    """

    def __init__(self, in_features=2048, hidden_dim=256):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True)
        )
        # 每种治疗一个分类头
        self.targeted_head = nn.Linear(hidden_dim, config.NUM_RESPONSE_CATEGORIES)
        self.immunotherapy_head = nn.Linear(hidden_dim, config.NUM_RESPONSE_CATEGORIES)
        self.chemotherapy_head = nn.Linear(hidden_dim, config.NUM_RESPONSE_CATEGORIES)
        self.combined_head = nn.Linear(hidden_dim, config.NUM_RESPONSE_CATEGORIES)

    def forward(self, x):
        shared = self.shared_layer(x)
        return {
            "targeted": self.targeted_head(shared),
            "immunotherapy": self.immunotherapy_head(shared),
            "chemotherapy": self.chemotherapy_head(shared),
            "combined": self.combined_head(shared),
        }


class PrognosisPredictor(nn.Module):
    """预后预测模块
    预测1年/3年/5年生存率和复发风险
    """

    def __init__(self, in_features=2048, hidden_dim=256):
        super().__init__()
        self.shared_layer = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_features, hidden_dim), nn.ReLU(inplace=True)
        )
        # 生存率预测 (sigmoid输出0-1概率)
        self.survival_head = nn.Sequential(
            nn.Linear(hidden_dim, len(config.SURVIVAL_TIMEPOINTS)), nn.Sigmoid()
        )
        # 复发风险分类
        self.recurrence_head = nn.Linear(hidden_dim, config.NUM_RISK_CLASSES)

    def forward(self, x):
        shared = self.shared_layer(x)
        return {
            "survival_prob": self.survival_head(shared),
            "recurrence_risk": self.recurrence_head(shared),
        }


class LungAIDualTaskModel(nn.Module):
    """原有双任务模型 (兼容现有model.pth)"""

    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor(pretrained=config.PRETRAINED)
        self.subtype_head = TaskHead(
            in_features=2048,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_SUBTYPES,
        )
        self.response_head = TaskHead(
            in_features=2048,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_RESPONSES,
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        subtype_pred = self.subtype_head(features)
        response_pred = self.response_head(features)
        return subtype_pred, response_pred

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            subtype_pred, response_pred = self.forward(x)
            subtype_prob = F.softmax(subtype_pred, dim=1)
            response_prob = F.softmax(response_pred, dim=1)
            return {
                "subtype_pred": torch.argmax(subtype_pred, 1),
                "subtype_prob": subtype_prob,
                "response_pred": torch.argmax(response_pred, 1),
                "response_prob": response_prob,
            }


class LungAICompleteModel(nn.Module):
    """完整肺癌诊断与治疗预测模型"""

    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor(pretrained=config.PRETRAINED)

        # 已有任务
        self.subtype_head = TaskHead(
            in_features=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_SUBTYPES,
        )
        self.response_head = TaskHead(
            in_features=config.FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=config.NUM_RESPONSES,
        )

        # 新增任务
        self.molecular_head = MolecularPredictor(
            in_features=config.FEATURE_DIM, hidden_dim=config.HIDDEN_DIM
        )
        self.treatment_head = TreatmentPredictor(
            in_features=config.FEATURE_DIM, hidden_dim=config.HIDDEN_DIM
        )
        self.prognosis_head = PrognosisPredictor(
            in_features=config.FEATURE_DIM, hidden_dim=config.HIDDEN_DIM
        )

        # 注意力融合
        self.attention_fusion = AttentionFusion(feature_dim=config.FEATURE_DIM)

    def forward(self, x):
        features = self.feature_extractor(x)

        # 基础诊断
        subtype_pred = self.subtype_head(features)
        response_pred = self.response_head(features)

        # 分子标记预测
        molecular_pred = self.molecular_head(features)

        # 治疗响应预测
        treatment_pred = self.treatment_head(features)

        # 预后预测
        prognosis_pred = self.prognosis_head(features)

        return {
            "subtype_pred": subtype_pred,
            "response_pred": response_pred,
            "molecular_pred": molecular_pred,
            "treatment_pred": treatment_pred,
            "prognosis_pred": prognosis_pred,
            "features": features,
        }

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            output = self.forward(x)

            # 基础诊断概率
            subtype_prob = F.softmax(output["subtype_pred"], dim=1)
            response_prob = F.softmax(output["response_pred"], dim=1)

            # 分子标记概率
            molecular_prob = {
                "egfr": F.softmax(output["molecular_pred"]["egfr"], dim=1),
                "alk": F.softmax(output["molecular_pred"]["alk"], dim=1),
                "kras": F.softmax(output["molecular_pred"]["kras"], dim=1),
                "pdl1": F.softmax(output["molecular_pred"]["pdl1"], dim=1),
            }

            # 治疗响应概率
            treatment_prob = {
                "targeted": F.softmax(output["treatment_pred"]["targeted"], dim=1),
                "immunotherapy": F.softmax(
                    output["treatment_pred"]["immunotherapy"], dim=1
                ),
                "chemotherapy": F.softmax(
                    output["treatment_pred"]["chemotherapy"], dim=1
                ),
                "combined": F.softmax(output["treatment_pred"]["combined"], dim=1),
            }

            # 预后结果
            survival_prob = output["prognosis_pred"]["survival_prob"]
            recurrence_prob = F.softmax(
                output["prognosis_pred"]["recurrence_risk"], dim=1
            )

            return {
                # 基础诊断
                "subtype_pred": torch.argmax(output["subtype_pred"], 1),
                "subtype_prob": subtype_prob,
                "response_pred": torch.argmax(output["response_pred"], 1),
                "response_prob": response_prob,
                # 分子标记
                "molecular_pred": {
                    "egfr": torch.argmax(output["molecular_pred"]["egfr"], 1),
                    "alk": torch.argmax(output["molecular_pred"]["alk"], 1),
                    "kras": torch.argmax(output["molecular_pred"]["kras"], 1),
                    "pdl1": torch.argmax(output["molecular_pred"]["pdl1"], 1),
                },
                "molecular_prob": molecular_prob,
                # 治疗响应
                "treatment_pred": {
                    "targeted": torch.argmax(output["treatment_pred"]["targeted"], 1),
                    "immunotherapy": torch.argmax(
                        output["treatment_pred"]["immunotherapy"], 1
                    ),
                    "chemotherapy": torch.argmax(
                        output["treatment_pred"]["chemotherapy"], 1
                    ),
                    "combined": torch.argmax(output["treatment_pred"]["combined"], 1),
                },
                "treatment_prob": treatment_prob,
                # 预后
                "survival_prob": survival_prob,
                "recurrence_pred": torch.argmax(
                    output["prognosis_pred"]["recurrence_risk"], 1
                ),
                "recurrence_prob": recurrence_prob,
                # 特征
                "features": output["features"],
            }


class MultiTaskLoss(nn.Module):
    """多任务损失函数"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.mse_criterion = nn.MSELoss()

        # 损失权重
        self.w_subtype = config.LOSS_WEIGHT_SUBTYPE
        self.w_response = config.LOSS_WEIGHT_RESPONSE
        self.w_molecular = config.LOSS_WEIGHT_MOLECULAR
        self.w_treatment = config.LOSS_WEIGHT_TREATMENT
        self.w_prognosis = config.LOSS_WEIGHT_PROGNOSIS

    def forward(self, predictions, targets):
        losses = {}

        # 基础诊断损失
        losses["subtype"] = self.criterion(
            predictions["subtype_pred"], targets["subtype"]
        )
        losses["response"] = self.criterion(
            predictions["response_pred"], targets["response"]
        )

        # 分子标记损失
        mol_pred = predictions["molecular_pred"]
        mol_target = targets["molecular"]
        losses["molecular"] = (
            self.criterion(mol_pred["egfr"], mol_target["egfr"])
            + self.criterion(mol_pred["alk"], mol_target["alk"])
            + self.criterion(mol_pred["kras"], mol_target["kras"])
            + self.criterion(mol_pred["pdl1"], mol_target["pdl1"])
        ) / 4

        # 治疗响应损失
        treat_pred = predictions["treatment_pred"]
        treat_target = targets["treatment"]
        losses["treatment"] = (
            self.criterion(treat_pred["targeted"], treat_target["targeted"])
            + self.criterion(treat_pred["immunotherapy"], treat_target["immunotherapy"])
            + self.criterion(treat_pred["chemotherapy"], treat_target["chemotherapy"])
            + self.criterion(treat_pred["combined"], treat_target["combined"])
        ) / 4

        # 预后损失
        prog_pred = predictions["prognosis_pred"]
        prog_target = targets["prognosis"]
        # 生存率用MSE
        losses["survival"] = self.mse_criterion(
            prog_pred["survival_prob"], prog_target["survival_prob"]
        )
        # 复发风险用交叉熵
        losses["recurrence"] = self.criterion(
            prog_pred["recurrence_risk"], prog_target["recurrence_risk"]
        )
        losses["prognosis"] = losses["survival"] + losses["recurrence"]

        # 总损失
        total_loss = (
            self.w_subtype * losses["subtype"]
            + self.w_response * losses["response"]
            + self.w_molecular * losses["molecular"]
            + self.w_treatment * losses["treatment"]
            + self.w_prognosis * losses["prognosis"]
        )
        losses["total"] = total_loss

        return total_loss, losses
