import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model.pth")

IMAGE_SIZE = 224
NUM_SAMPLES = 15000

# 类别定义 (3类: 腺癌, 鳞癌, 正常)
BACKBONE = "resnet50"
PRETRAINED = True
FEATURE_DIM = 2048  # ResNet-50特征维度
HIDDEN_DIM = 256
NUM_SUBTYPES = 3  # 腺癌=0, 鳞癌=1, 正常=2
NUM_RESPONSES = 2  # 癌症=1, 正常=0
SUBTYPE_NAMES = [
    "肺腺癌 (Adenocarcinoma)",
    "肺鳞癌 (Squamous Cell Carcinoma)",
    "正常肺组织 (Normal)",
]
RESPONSE_NAMES = ["正常 (Normal)", "癌症 (Cancer)"]

# ============ 新增：分子标记预测配置 ============
NUM_MOLECULAR_MARKERS = 4  # EGFR, ALK, KRAS, PD-L1
MOLECULAR_MARKER_NAMES = ["EGFR", "ALK", "KRAS", "PD-L1"]
MOLECULAR_STATUS_NAMES = {
    "EGFR": ["阴性 (Negative)", "阳性 (Positive)"],
    "ALK": ["阴性 (Negative)", "阳性 (Positive)"],
    "KRAS": ["野生型 (Wild-type)", "G12C突变", "其他突变"],
    "PD-L1": ["阴性 (<1%)", "低表达 (1-49%)", "高表达 (>=50%)"],
}
NUM_EGFR_CLASSES = 2  # 阳性/阴性
NUM_ALK_CLASSES = 2  # 阳性/阴性
NUM_KRAS_CLASSES = 3  # 野生型/G12C/其他
NUM_PDL1_CLASSES = 3  # 阴性/低/高

# ============ 新增：治疗响应预测配置 ============
TREATMENT_TYPES = ["靶向治疗", "免疫治疗", "化疗", "联合治疗"]
RESPONSE_CATEGORIES = [
    "CR (完全缓解)",
    "PR (部分缓解)",
    "SD (疾病稳定)",
    "PD (疾病进展)",
]
NUM_TREATMENT_TYPES = 4
NUM_RESPONSE_CATEGORIES = 4

# ============ 新增：预后预测配置 ============
SURVIVAL_TIMEPOINTS = [1, 3, 5]  # 1年/3年/5年生存率
RISK_LEVELS = ["低风险", "中风险", "高风险"]
NUM_RISK_CLASSES = 3

# ============ 新增：损失权重配置 ============
LOSS_WEIGHT_SUBTYPE = 1.0
LOSS_WEIGHT_RESPONSE = 1.0
LOSS_WEIGHT_MOLECULAR = 0.5
LOSS_WEIGHT_TREATMENT = 0.8
LOSS_WEIGHT_PROGNOSIS = 0.6

BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 15
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0
PIN_MEMORY = True

for dir_path in [DATA_DIR, TRAIN_DIR, VAL_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

print(f"Device: {DEVICE}")
