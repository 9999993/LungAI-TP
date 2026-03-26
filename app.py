"""
LungAI-TP 肺癌智能诊断与治疗预测系统
专业可视化界面
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 字体配置
def find_chinese_font():
    font_paths = [
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None


font_path = find_chinese_font()
if font_path:
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font_prop.get_name()
else:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from models import create_pathology_resnet
from knowledge_reasoner import knowledge_base


# 常量定义
SUBTYPE_INFO = {
    0: {
        "name": "肺腺癌",
        "code": "LUAD",
        "color": "#1976d2",
        "desc": "最常见的非小细胞肺癌，约占40-50%",
    },
    1: {
        "name": "肺鳞状细胞癌",
        "code": "LUSC",
        "color": "#d32f2f",
        "desc": "与吸烟高度相关，约占25-30%",
    },
    2: {
        "name": "正常肺组织",
        "code": "NORMAL",
        "color": "#388e3c",
        "desc": "未发现恶性病变特征",
    },
}

TREATMENT_NAMES = {
    "targeted": "靶向治疗",
    "immunotherapy": "免疫治疗",
    "chemotherapy": "化疗",
}

RESPONSE_NAMES = ["CR", "PR", "SD", "PD"]
RESPONSE_COLORS = ["#4caf50", "#8bc34a", "#ff9800", "#f44336"]
RESPONSE_LABELS = ["完全缓解", "部分缓解", "疾病稳定", "疾病进展"]


@st.cache_resource
def load_model():
    """加载训练好的模型"""
    model = create_pathology_resnet(
        num_classes=config.NUM_SUBTYPES,
        pretrained=False,
        use_pathology_module=True,
        light=True,
    ).to(config.DEVICE)
    if os.path.exists(config.MODEL_PATH):
        model.load_state_dict(
            torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True)
        )
        model.eval()
        return model, True
    return model, False


def set_page_style():
    """设置页面样式"""
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
    
    * {
        font-family: 'Noto Sans SC', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #1976d2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid #f0f0f0;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.8rem;
        border-bottom: 2px solid #f5f5f5;
    }
    
    .step-badge {
        background: linear-gradient(135deg, #1976d2, #42a5f5);
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 12px;
        font-size: 14px;
        box-shadow: 0 2px 8px rgba(25,118,210,0.3);
    }
    
    .step-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
    }
    
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid;
        margin: 1rem 0;
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .result-desc {
        color: #666;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .metric-box {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e8e8e8;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1976d2;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    .molecular-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e8e8e8;
        transition: transform 0.2s;
    }
    
    .molecular-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .molecular-name {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.5rem;
    }
    
    .molecular-value {
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    .molecular-desc {
        font-size: 0.75rem;
        color: #999;
        margin-top: 0.3rem;
    }
    
    .treatment-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .treatment-rank {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-right: 10px;
    }
    
    .treatment-name {
        font-weight: 600;
        font-size: 1rem;
    }
    
    .treatment-meta {
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    .prognosis-card {
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e8e8e8;
    }
    
    .prognosis-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .prognosis-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.3rem;
    }
    
    .risk-low { color: #4caf50; }
    .risk-medium { color: #ff9800; }
    .risk-high { color: #f44336; }
    
    .upload-area {
        border: 2px dashed #1976d2;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa, #ffffff);
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        border-color: #42a5f5;
        background: linear-gradient(135deg, #e3f2fd, #ffffff);
    }
    
    .sidebar-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e8e8e8;
    }
    
    .status-success {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    .status-info {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 2rem;
    }
    
    .feature-item {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #e8e8e8;
    }
    
    .feature-icon {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: #666;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def show_diagnosis_page():
    """诊断分析页面"""
    model, loaded = load_model()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            """
        <div class="card">
            <div class="card-header">
                <span class="step-badge">📤</span>
                <span class="step-title">上传病理图像</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        uploaded = st.file_uploader(
            "", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True, caption="已上传图像")

            if loaded:
                st.markdown(
                    '<div class="status-success">✓ 模型已就绪，可进行诊断</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
            <div class="upload-area">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🔬</div>
                <div style="font-size: 1.1rem; color: #666;">点击或拖拽上传病理切片图像</div>
                <div style="font-size: 0.85rem; color: #999; margin-top: 0.5rem;">支持 JPG、PNG 格式</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        if uploaded:
            with st.spinner("🔄 正在分析..."):
                transform = transforms.Compose(
                    [
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                )
                img_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

                with torch.no_grad():
                    result = model.predict(img_tensor)

                subtype_idx = result["subtype_pred"].item()
                subtype_prob = result["subtype_prob"][0].cpu().numpy()
                molecular_inference = knowledge_base.infer_molecular_markers(
                    subtype_idx
                )
                molecular_profile = knowledge_base.get_molecular_profile(subtype_idx)

            # ========== 诊断结果 ==========
            info = SUBTYPE_INFO[subtype_idx]
            color = info["color"]

            st.markdown(
                f"""
            <div class="result-card" style="border-left-color: {color};">
                <div class="result-title" style="color: {color};">{info["name"]}</div>
                <div class="result-desc">{info["desc"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(
                    f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: {color};">{info["code"]}</div>
                    <div class="metric-label">诊断分类</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with m2:
                confidence = subtype_prob[subtype_idx] * 100
                st.markdown(
                    f"""
                <div class="metric-box">
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">置信度</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with m3:
                response = "阳性" if subtype_idx != 2 else "阴性"
                response_color = "#f44336" if subtype_idx != 2 else "#4caf50"
                st.markdown(
                    f"""
                <div class="metric-box">
                    <div class="metric-value" style="color: {response_color};">{response}</div>
                    <div class="metric-label">癌症检测</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # 概率分布图
            fig, ax = plt.subplots(figsize=(10, 2.5))
            colors = ["#1976d2", "#d32f2f", "#388e3c"]
            labels = ["肺腺癌 LUAD", "肺鳞癌 LUSC", "正常组织"]
            bars = ax.barh(labels, subtype_prob * 100, color=colors, height=0.5)
            for bar, prob in zip(bars, subtype_prob):
                ax.text(
                    bar.get_width() + 1,
                    bar.get_y() + bar.get_height() / 2,
                    f"{prob * 100:.1f}%",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )
            ax.set_xlim(0, 100)
            ax.set_xlabel("置信度 (%)", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            # ========== 分子标记 ==========
            st.markdown(
                """
            <div class="card">
                <div class="card-header">
                    <span class="step-badge">🧬</span>
                    <span class="step-title">分子标记预测</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            mol_cols = st.columns(4)
            markers = [
                (
                    "EGFR",
                    "EGFR突变",
                    molecular_inference["EGFR"]["probability"],
                    "常见驱动基因",
                ),
                (
                    "ALK",
                    "ALK融合",
                    molecular_inference["ALK"]["probability"],
                    "靶向治疗靶点",
                ),
                (
                    "KRAS",
                    "KRAS突变",
                    molecular_inference["KRAS"]["probability"],
                    "G12C可靶向",
                ),
                (
                    "PD-L1",
                    "PD-L1高表达",
                    molecular_inference["PD_L1"]["high"],
                    "免疫治疗标志物",
                ),
            ]

            for idx, (key, name, value, desc) in enumerate(markers):
                with mol_cols[idx]:
                    color = "#f44336" if value > 0.3 else "#4caf50"
                    st.markdown(
                        f"""
                    <div class="molecular-card">
                        <div class="molecular-name">{name}</div>
                        <div class="molecular-value" style="color: {color};">{value * 100:.1f}%</div>
                        <div class="molecular-desc">{desc}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            # ========== 治疗方案 ==========
            if subtype_idx != 2:
                st.markdown(
                    """
                <div class="card">
                    <div class="card-header">
                        <span class="step-badge">💊</span>
                        <span class="step-title">推荐治疗方案</span>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                recommendations = knowledge_base.get_treatment_recommendations(
                    subtype_idx, molecular_profile
                )

                for i, rec in enumerate(recommendations[:3]):
                    colors = ["#1976d2", "#4caf50", "#ff9800"]
                    ranks = ["首选方案", "备选方案", "备选方案"]
                    st.markdown(
                        f"""
                    <div class="treatment-card" style="border-left-color: {colors[i]};">
                        <span class="treatment-rank" style="background: {colors[i]};">{ranks[i]}</span>
                        <span class="treatment-name">{rec["name"]}</span>
                        <div class="treatment-meta">
                            类型：{rec["category"]} | 预期响应率：{rec["response_rate"] * 100:.0f}%
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            # ========== 预后评估 ==========
            st.markdown(
                """
            <div class="card">
                <div class="card-header">
                    <span class="step-badge">📊</span>
                    <span class="step-title">预后评估</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            prognosis = knowledge_base.get_prognosis(subtype_idx, molecular_profile)

            prog_cols = st.columns(4)
            survivals = [prognosis["1yr"], prognosis["3yr"], prognosis["5yr"]]
            labels = ["1年生存率", "3年生存率", "5年生存率"]

            for col, label, surv in zip(prog_cols[:3], labels, survivals):
                with col:
                    color = (
                        "#4caf50"
                        if surv > 0.6
                        else ("#ff9800" if surv > 0.3 else "#f44336")
                    )
                    st.markdown(
                        f"""
                    <div class="prognosis-card">
                        <div class="prognosis-value" style="color: {color};">{surv * 100:.1f}%</div>
                        <div class="prognosis-label">{label}</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            with prog_cols[3]:
                risk = (
                    "低风险"
                    if prognosis["5yr"] > 0.4
                    else ("中风险" if prognosis["5yr"] > 0.2 else "高风险")
                )
                risk_class = (
                    "risk-low"
                    if risk == "低风险"
                    else ("risk-medium" if risk == "中风险" else "risk-high")
                )
                st.markdown(
                    f"""
                <div class="prognosis-card">
                    <div class="prognosis-value {risk_class}">{risk}</div>
                    <div class="prognosis-label">复发风险</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # 生存曲线
            fig, ax = plt.subplots(figsize=(8, 3))
            timepoints = ["1年", "3年", "5年"]
            ax.plot(
                timepoints,
                [s * 100 for s in survivals],
                "o-",
                color="#1976d2",
                linewidth=2.5,
                markersize=10,
                markerfacecolor="white",
                markeredgewidth=2,
            )
            ax.fill_between(
                timepoints, [s * 100 for s in survivals], alpha=0.15, color="#1976d2"
            )
            ax.set_ylim(0, 100)
            ax.set_ylabel("生存率 (%)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for i, y in enumerate(survivals):
                ax.text(
                    i,
                    y * 100 + 4,
                    f"{y * 100:.1f}%",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.markdown(
                """
            <div class="welcome-card">
                <div class="welcome-icon">🔬</div>
                <h2 style="color: #1e3a5f; margin-bottom: 0.5rem;">LungAI-TP 肺癌智能诊断系统</h2>
                <p style="color: #666; font-size: 1.1rem;">基于深度学习的病理图像分析与治疗预测</p>
                
                <div class="feature-grid">
                    <div class="feature-item">
                        <div class="feature-icon">🔬</div>
                        <div class="feature-title">诊断分型</div>
                        <div class="feature-desc">肺腺癌/肺鳞癌/正常组织识别</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">🧬</div>
                        <div class="feature-title">分子标记</div>
                        <div class="feature-desc">EGFR/ALK/KRAS/PD-L1预测</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">💊</div>
                        <div class="feature-title">治疗方案</div>
                        <div class="feature-desc">基于NCCN指南的个性化推荐</div>
                    </div>
                    <div class="feature-item">
                        <div class="feature-icon">📊</div>
                        <div class="feature-title">预后评估</div>
                        <div class="feature-desc">生存率与复发风险预测</div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )


def show_dataset_page():
    """数据集信息页面"""
    st.markdown("## 📊 数据集信息")

    # 数据概览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-box"><div class="metric-value">LC25000</div><div class="metric-label">数据来源</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-box"><div class="metric-value">15,000</div><div class="metric-label">图像总数</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-box"><div class="metric-value">3</div><div class="metric-label">类别数量</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-box"><div class="metric-value">224×224</div><div class="metric-label">图像尺寸</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 数据划分表格
    st.markdown("### 数据划分")
    data_split = pd.DataFrame(
        {
            "数据集": ["训练集 (70%)", "验证集 (15%)", "测试集 (15%)", "总计"],
            "肺腺癌 LUAD": ["3,500", "750", "750", "5,000"],
            "肺鳞癌 LUSC": ["3,500", "750", "750", "5,000"],
            "正常组织": ["3,500", "750", "750", "5,000"],
            "总计": ["10,500", "2,250", "2,250", "15,000"],
        }
    )
    st.dataframe(data_split, use_container_width=True, hide_index=True)

    # 数据分布图
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = ["LUAD\n肺腺癌", "LUSC\n肺鳞癌", "Normal\n正常组织"]
        sizes = [5000, 5000, 5000]
        colors = ["#1976d2", "#d32f2f", "#388e3c"]
        explode = (0.05, 0.05, 0.05)
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=explode,
            textprops={"fontsize": 12},
        )
        ax.set_title("类别分布", fontsize=14, fontweight="bold", pad=20)
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        x = np.arange(3)
        width = 0.25
        ax.bar(x - width, [3500, 3500, 3500], width, label="训练集", color="#1976d2")
        ax.bar(x, [750, 750, 750], width, label="验证集", color="#ff9800")
        ax.bar(x + width, [750, 750, 750], width, label="测试集", color="#4caf50")
        ax.set_xticks(x)
        ax.set_xticklabels(["LUAD", "LUSC", "Normal"], fontsize=12)
        ax.set_ylabel("样本数量")
        ax.set_title("数据划分", fontsize=14, fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close()


def show_performance_page():
    """模型性能页面"""
    st.markdown("## 🧠 模型性能")

    # 模型信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
        <div class="card">
            <h4>模型架构</h4>
            <ul>
                <li>名称: PathologyResNet-Light</li>
                <li>Backbone: ResNet-18</li>
                <li>参数量: 11.4M</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
        <div class="card">
            <h4>创新模块</h4>
            <ul>
                <li>PAAM 病理感知模块</li>
                <li>SE 通道注意力</li>
                <li>多尺度特征融合</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
        <div class="card">
            <h4>训练配置</h4>
            <ul>
                <li>Epochs: 15</li>
                <li>Batch Size: 64</li>
                <li>GPU: RTX 5060</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 训练历史
    history_path = os.path.join(config.RESULTS_DIR, "history.csv")
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        best_epoch = df.loc[df["val_acc"].idxmax()]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value" style="color: #4caf50;">{best_epoch["val_acc"]:.2%}</div><div class="metric-label">最佳验证准确率</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{best_epoch["train_acc"]:.2%}</div><div class="metric-label">训练准确率</div></div>',
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">{best_epoch["val_loss"]:.4f}</div><div class="metric-label">验证损失</div></div>',
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                f'<div class="metric-box"><div class="metric-value">Epoch {int(best_epoch["epoch"])}</div><div class="metric-label">最佳轮次</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # 训练曲线
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                df["epoch"],
                df["train_acc"],
                "b-o",
                label="训练准确率",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                df["epoch"],
                df["val_acc"],
                "r-s",
                label="验证准确率",
                linewidth=2,
                markersize=6,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title("准确率曲线", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0.5, 1.02])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close()
        with col2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                df["epoch"],
                df["train_loss"],
                "b-o",
                label="训练损失",
                linewidth=2,
                markersize=6,
            )
            ax.plot(
                df["epoch"],
                df["val_loss"],
                "r-s",
                label="验证损失",
                linewidth=2,
                markersize=6,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("损失曲线", fontsize=14, fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close()

        # 详细历史
        with st.expander("📋 查看详细训练历史"):
            st.dataframe(df, use_container_width=True)


def main():
    st.set_page_config(
        page_title="LungAI-TP 肺癌诊断系统",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    set_page_style()

    # 主标题
    st.markdown(
        """
    <div class="main-header">
        <h1>🔬 LungAI-TP 肺癌智能诊断与治疗预测系统</h1>
        <p>基于深度学习的病理图像分析与治疗预测</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 侧边栏
    with st.sidebar:
        st.markdown("## 📍 功能导航")
        page = st.radio(
            "",
            ["🔬 诊断分析", "📊 数据集信息", "🧠 模型性能"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown("## ⚙️ 系统状态")
        model, loaded = load_model()
        if loaded:
            st.markdown(
                '<div class="status-success">✓ 模型已加载</div>', unsafe_allow_html=True
            )
        else:
            st.error("✗ 模型未加载")

    # 页面路由
    if page == "📊 数据集信息":
        show_dataset_page()
    elif page == "🧠 模型性能":
        show_performance_page()
    else:
        show_diagnosis_page()


if __name__ == "__main__":
    main()
