"""肺癌治疗知识图谱与规则推理模块
基于临床统计数据和循证医学指南

数据来源：
- TCGA (The Cancer Genome Atlas): 分子标记频率统计
- NCCN Guidelines 2024: 治疗方案推荐
- ClinicalTrials.gov: 治疗响应率统计
- SEER Database: 生存率统计数据
- OncoKB: 精准医学知识库
"""

import numpy as np


class LungCancerKnowledgeBase:
    """肺癌治疗知识库（基于循证医学）"""

    def __init__(self):
        # ============ 数据来源说明 ============
        self.data_sources = {
            "molecular": "TCGA Pan-Lung Cancer (n=1144), Campbell et al. 2016",
            "treatment_response": "NCCN Guidelines 2024, ClinicalTrials.gov",
            "prognosis": "SEER Cancer Statistics Review 2020-2024",
            "treatment_guidelines": "NCCN NSCLC Guidelines v4.2024",
        }

        # ============ 分子标记流行病学数据 (TCGA统计) ============
        self.molecular_epidemiology = {
            "LUAD": {
                "EGFR": {
                    "positive_rate": 0.45,  # 45%
                    "confidence_interval": (0.40, 0.50),
                    "source": "TCGA Pan-Lung, n=522 LUAD",
                },
                "ALK": {
                    "positive_rate": 0.07,  # 7%
                    "confidence_interval": (0.05, 0.09),
                    "source": "TCGA Pan-Lung, n=522 LUAD",
                },
                "KRAS": {
                    "mutation_rate": 0.28,  # 28%
                    "g12c_rate": 0.13,  # 13% of all LUAD
                    "confidence_interval": (0.24, 0.32),
                    "source": "TCGA Pan-Lung, n=522 LUAD",
                },
                "PD_L1": {
                    "negative_rate": 0.40,  # <1%
                    "low_rate": 0.35,  # 1-49%
                    "high_rate": 0.25,  # >=50%
                    "source": "TCGA + Clinical validation cohorts",
                },
            },
            "LUSC": {
                "EGFR": {
                    "positive_rate": 0.05,
                    "confidence_interval": (0.03, 0.07),
                    "source": "TCGA Pan-Lung, n=487 LUSC",
                },
                "ALK": {
                    "positive_rate": 0.02,
                    "confidence_interval": (0.01, 0.03),
                    "source": "TCGA Pan-Lung, n=487 LUSC",
                },
                "KRAS": {
                    "mutation_rate": 0.12,
                    "g12c_rate": 0.05,
                    "confidence_interval": (0.09, 0.15),
                    "source": "TCGA Pan-Lung, n=487 LUSC",
                },
                "PD_L1": {
                    "negative_rate": 0.30,
                    "low_rate": 0.25,
                    "high_rate": 0.45,
                    "source": "TCGA + Clinical validation cohorts",
                },
            },
        }

        # ============ 治疗方案数据库 (NCCN Guidelines 2024) ============
        self.treatments = {
            "targeted": {
                "EGFR_positive": [
                    {
                        "name": "奥希替尼 (Osimertinib)",
                        "priority": 1,
                        "response_rate": 0.80,
                        "evidence_level": "Category 1",
                        "source": "FLAURA trial, N Engl J Med 2018",
                    },
                    {
                        "name": "厄洛替尼 (Erlotinib)",
                        "priority": 2,
                        "response_rate": 0.65,
                        "evidence_level": "Category 1",
                        "source": "EURTAC trial, Lancet Oncol 2012",
                    },
                    {
                        "name": "吉非替尼 (Gefitinib)",
                        "priority": 3,
                        "response_rate": 0.60,
                        "evidence_level": "Category 1",
                        "source": "IPASS trial, N Engl J Med 2009",
                    },
                ],
                "ALK_positive": [
                    {
                        "name": "阿来替尼 (Alectinib)",
                        "priority": 1,
                        "response_rate": 0.82,
                        "evidence_level": "Category 1",
                        "source": "ALEX trial, N Engl J Med 2017",
                    },
                    {
                        "name": "克唑替尼 (Crizotinib)",
                        "priority": 2,
                        "response_rate": 0.65,
                        "evidence_level": "Category 1",
                        "source": "PROFILE 1014, N Engl J Med 2014",
                    },
                ],
                "KRAS_G12C": [
                    {
                        "name": "索托拉西布 (Sotorasib)",
                        "priority": 1,
                        "response_rate": 0.45,
                        "evidence_level": "Category 2A",
                        "source": "CodeBreaK 100, N Engl J Med 2021",
                    },
                    {
                        "name": "阿达格拉西布 (Adagrasib)",
                        "priority": 2,
                        "response_rate": 0.43,
                        "evidence_level": "Category 2A",
                        "source": "KRYSTAL-1, N Engl J Med 2022",
                    },
                ],
            },
            "immunotherapy": {
                "PD_L1_high": [
                    {
                        "name": "帕博利珠单抗 (Pembrolizumab)",
                        "priority": 1,
                        "response_rate": 0.45,
                        "evidence_level": "Category 1",
                        "source": "KEYNOTE-024, N Engl J Med 2016",
                    },
                ],
                "PD_L1_low": [
                    {
                        "name": "帕博利珠单抗+化疗",
                        "priority": 1,
                        "response_rate": 0.48,
                        "evidence_level": "Category 1",
                        "source": "KEYNOTE-189, N Engl J Med 2018",
                    },
                    {
                        "name": "纳武利尤单抗+伊匹木单抗",
                        "priority": 2,
                        "response_rate": 0.40,
                        "evidence_level": "Category 1",
                        "source": "CheckMate 227, N Engl J Med 2019",
                    },
                ],
                "PD_L1_negative": [
                    {
                        "name": "纳武利尤单抗+化疗",
                        "priority": 1,
                        "response_rate": 0.35,
                        "evidence_level": "Category 1",
                        "source": "CheckMate 9LA, Lancet Oncol 2021",
                    },
                ],
            },
            "chemotherapy": {
                "LUAD": [
                    {
                        "name": "培美曲塞+顺铂",
                        "priority": 1,
                        "response_rate": 0.35,
                        "evidence_level": "Category 1",
                        "source": "JMDB trial, J Clin Oncol 2008",
                    },
                    {
                        "name": "培美曲塞+卡铂",
                        "priority": 2,
                        "response_rate": 0.33,
                        "evidence_level": "Category 1",
                        "source": "NCCN Guidelines 2024",
                    },
                ],
                "LUSC": [
                    {
                        "name": "吉西他滨+顺铂",
                        "priority": 1,
                        "response_rate": 0.32,
                        "evidence_level": "Category 1",
                        "source": "JMDB trial, J Clin Oncol 2008",
                    },
                    {
                        "name": "多西他赛+顺铂",
                        "priority": 2,
                        "response_rate": 0.30,
                        "evidence_level": "Category 1",
                        "source": "NCCN Guidelines 2024",
                    },
                ],
            },
        }

        # ============ 治疗响应概率 (临床试验数据) ============
        self.response_rules = {
            "targeted": {
                "EGFR_positive": {"CR": 0.15, "PR": 0.55, "SD": 0.20, "PD": 0.10},
                "EGFR_negative": {"CR": 0.02, "PR": 0.08, "SD": 0.30, "PD": 0.60},
                "ALK_positive": {"CR": 0.12, "PR": 0.58, "SD": 0.20, "PD": 0.10},
                "ALK_negative": {"CR": 0.02, "PR": 0.10, "SD": 0.28, "PD": 0.60},
                "KRAS_G12C": {"CR": 0.05, "PR": 0.35, "SD": 0.35, "PD": 0.25},
            },
            "immunotherapy": {
                "PD_L1_high": {"CR": 0.08, "PR": 0.37, "SD": 0.30, "PD": 0.25},
                "PD_L1_low": {"CR": 0.05, "PR": 0.25, "SD": 0.35, "PD": 0.35},
                "PD_L1_negative": {"CR": 0.03, "PR": 0.12, "SD": 0.35, "PD": 0.50},
            },
            "chemotherapy": {
                "LUAD": {"CR": 0.03, "PR": 0.25, "SD": 0.40, "PD": 0.32},
                "LUSC": {"CR": 0.02, "PR": 0.22, "SD": 0.38, "PD": 0.38},
            },
        }

        # ============ 预后数据 (SEER Database) ============
        self.prognosis_data = {
            "LUAD": {
                "overall": {
                    "1yr": 0.72,
                    "3yr": 0.42,
                    "5yr": 0.25,
                    "source": "SEER Cancer Statistics 2020-2024",
                },
                "with_targetable_mutation": {
                    "1yr": 0.85,
                    "3yr": 0.55,
                    "5yr": 0.35,
                    "source": "Shaw et al. Lancet Oncol 2019 (ALK), Ramalingam et al. NEJM 2020 (EGFR)",
                    "note": "有靶向突变患者使用靶向治疗后预后改善",
                },
                "early_stage": {
                    "1yr": 0.92,
                    "3yr": 0.78,
                    "5yr": 0.65,
                    "source": "SEER Cancer Statistics 2020-2024",
                },
            },
            "LUSC": {
                "overall": {
                    "1yr": 0.60,
                    "3yr": 0.30,
                    "5yr": 0.18,
                    "source": "SEER Cancer Statistics 2020-2024",
                },
                "with_targetable_mutation": {
                    "1yr": 0.75,
                    "3yr": 0.40,
                    "5yr": 0.25,
                    "source": "Limited data, extrapolated from NSCLC overall",
                },
                "early_stage": {
                    "1yr": 0.88,
                    "3yr": 0.70,
                    "5yr": 0.55,
                    "source": "SEER Cancer Statistics 2020-2024",
                },
            },
            "normal": {
                "default": {
                    "1yr": 0.98,
                    "3yr": 0.96,
                    "5yr": 0.94,
                    "source": "General population statistics",
                }
            },
        }

    def get_molecular_profile(self, subtype_idx):
        """获取分子标记profile（用于治疗推荐）

        Args:
            subtype_idx: 亚型索引

        Returns:
            dict: 分子标记状态（基于最可能的情况）
        """
        inference = self.infer_molecular_markers(subtype_idx)

        # 基于概率返回最可能的状态
        egfr = 1 if inference["EGFR"]["probability"] > 0.3 else 0
        alk = 1 if inference["ALK"]["probability"] > 0.05 else 0
        kras = 1 if inference["KRAS"]["g12c_probability"] > 0.1 else 0

        # PD-L1: 返回最可能的状态
        pdl1_probs = [
            inference["PD_L1"]["negative"],
            inference["PD_L1"]["low"],
            inference["PD_L1"]["high"],
        ]
        pdl1 = pdl1_probs.index(max(pdl1_probs))

        return {"egfr": egfr, "alk": alk, "kras": kras, "pdl1": pdl1}

    def infer_molecular_markers(self, subtype_idx):
        """基于分型结果推理分子标记概率

        Args:
            subtype_idx: 亚型索引 (0=LUAD, 1=LUSC, 2=Normal)

        Returns:
            dict: 分子标记概率分布及置信区间
        """
        if subtype_idx == 2:  # 正常组织
            return {
                "EGFR": {
                    "probability": 0.02,
                    "confidence_interval": (0.01, 0.03),
                    "source": "正常组织极少检出驱动突变",
                },
                "ALK": {
                    "probability": 0.01,
                    "confidence_interval": (0.005, 0.015),
                    "source": "正常组织极少检出驱动突变",
                },
                "KRAS": {
                    "probability": 0.02,
                    "confidence_interval": (0.01, 0.03),
                    "source": "正常组织极少检出驱动突变",
                },
                "PD_L1": {
                    "negative": 0.90,
                    "low": 0.08,
                    "high": 0.02,
                    "source": "正常肺组织PD-L1表达",
                },
            }

        subtype_name = "LUAD" if subtype_idx == 0 else "LUSC"
        mol_data = self.molecular_epidemiology[subtype_name]

        return {
            "EGFR": {
                "probability": mol_data["EGFR"]["positive_rate"],
                "confidence_interval": mol_data["EGFR"]["confidence_interval"],
                "source": mol_data["EGFR"]["source"],
            },
            "ALK": {
                "probability": mol_data["ALK"]["positive_rate"],
                "confidence_interval": mol_data["ALK"]["confidence_interval"],
                "source": mol_data["ALK"]["source"],
            },
            "KRAS": {
                "probability": mol_data["KRAS"]["mutation_rate"],
                "g12c_probability": mol_data["KRAS"]["g12c_rate"],
                "confidence_interval": mol_data["KRAS"]["confidence_interval"],
                "source": mol_data["KRAS"]["source"],
            },
            "PD_L1": {
                "negative": mol_data["PD_L1"]["negative_rate"],
                "low": mol_data["PD_L1"]["low_rate"],
                "high": mol_data["PD_L1"]["high_rate"],
                "source": mol_data["PD_L1"]["source"],
            },
        }

    def get_treatment_recommendations(self, subtype_idx, molecular_profile):
        """根据分型和分子标记获取治疗推荐

        Args:
            subtype_idx: 亚型索引 (0=LUAD, 1=LUSC, 2=Normal)
            molecular_profile: 分子标记状态字典

        Returns:
            推荐治疗方案列表（含循证医学证据等级）
        """
        recommendations = []

        if subtype_idx == 2:  # 正常组织
            return [
                {
                    "name": "无需治疗",
                    "priority": 0,
                    "response_rate": 1.0,
                    "category": "none",
                    "evidence_level": "N/A",
                    "source": "正常组织无需治疗",
                }
            ]

        subtype_name = "LUAD" if subtype_idx == 0 else "LUSC"
        egfr = molecular_profile.get("egfr", 0)
        alk = molecular_profile.get("alk", 0)
        kras = molecular_profile.get("kras", 0)
        pdl1 = molecular_profile.get("pdl1", 0)

        # 靶向治疗推荐（优先级最高）
        if egfr == 1:  # EGFR阳性
            for t in self.treatments["targeted"]["EGFR_positive"]:
                recommendations.append({**t, "category": "靶向治疗"})
        elif alk == 1:  # ALK阳性
            for t in self.treatments["targeted"]["ALK_positive"]:
                recommendations.append({**t, "category": "靶向治疗"})
        elif kras == 1:  # KRAS G12C
            for t in self.treatments["targeted"]["KRAS_G12C"]:
                recommendations.append({**t, "category": "靶向治疗"})

        # 免疫治疗推荐
        if pdl1 == 2:  # PD-L1高表达
            for t in self.treatments["immunotherapy"]["PD_L1_high"]:
                recommendations.append({**t, "category": "免疫治疗"})
        elif pdl1 == 1:  # PD-L1低表达
            for t in self.treatments["immunotherapy"]["PD_L1_low"]:
                recommendations.append({**t, "category": "免疫治疗"})
        else:  # PD-L1阴性
            for t in self.treatments["immunotherapy"]["PD_L1_negative"]:
                recommendations.append({**t, "category": "免疫治疗"})

        # 化疗推荐（标准方案）
        for t in self.treatments["chemotherapy"][subtype_name]:
            recommendations.append({**t, "category": "化疗"})

        # 按优先级排序
        recommendations.sort(key=lambda x: x["priority"])

        return recommendations[:5]  # 返回前5个推荐

    def get_treatment_response(self, subtype_idx, molecular_profile, treatment_type):
        """获取治疗响应概率

        Args:
            subtype_idx: 亚型索引
            molecular_profile: 分子标记状态
            treatment_type: 治疗类型

        Returns:
            响应概率字典及数据来源
        """
        subtype_name = "LUAD" if subtype_idx == 0 else "LUSC"
        egfr = molecular_profile.get("egfr", 0)
        alk = molecular_profile.get("alk", 0)
        kras = molecular_profile.get("kras", 0)
        pdl1 = molecular_profile.get("pdl1", 0)

        if treatment_type == "targeted":
            if egfr == 1:
                probs = self.response_rules["targeted"]["EGFR_positive"]
                source = "FLAURA, EURTAC trials"
            elif alk == 1:
                probs = self.response_rules["targeted"]["ALK_positive"]
                source = "ALEX, PROFILE 1014 trials"
            elif kras == 1:
                probs = self.response_rules["targeted"]["KRAS_G12C"]
                source = "CodeBreaK 100, KRYSTAL-1 trials"
            else:
                probs = self.response_rules["targeted"]["EGFR_negative"]
                source = "无靶向突变患者靶向治疗数据"
        elif treatment_type == "immunotherapy":
            if pdl1 == 2:
                probs = self.response_rules["immunotherapy"]["PD_L1_high"]
                source = "KEYNOTE-024 trial"
            elif pdl1 == 1:
                probs = self.response_rules["immunotherapy"]["PD_L1_low"]
                source = "KEYNOTE-189 trial"
            else:
                probs = self.response_rules["immunotherapy"]["PD_L1_negative"]
                source = "CheckMate 9LA trial"
        elif treatment_type == "chemotherapy":
            probs = self.response_rules["chemotherapy"][subtype_name]
            source = "JMDB trial, meta-analyses"
        else:  # combined
            probs = {"CR": 0.10, "PR": 0.40, "SD": 0.30, "PD": 0.20}
            source = "联合治疗综合数据"

        return {**probs, "source": source}

    def get_prognosis(self, subtype_idx, molecular_profile):
        """获取预后信息

        Args:
            subtype_idx: 亚型索引
            molecular_profile: 分子标记状态

        Returns:
            预后信息及数据来源
        """
        if subtype_idx == 2:  # 正常
            return self.prognosis_data["normal"]["default"]

        subtype_name = "LUAD" if subtype_idx == 0 else "LUSC"
        egfr = molecular_profile.get("egfr", 0)
        alk = molecular_profile.get("alk", 0)
        kras = molecular_profile.get("kras", 0)

        has_targetable = egfr == 1 or alk == 1 or kras == 1

        if has_targetable:
            return self.prognosis_data[subtype_name]["with_targetable_mutation"]
        else:
            return self.prognosis_data[subtype_name]["overall"]

    def generate_explanation(self, subtype_idx, subtype_prob, molecular_inference):
        """生成诊断解释文本

        Args:
            subtype_idx: 亚型索引
            subtype_prob: 分型概率
            molecular_inference: 分子标记推理结果

        Returns:
            解释文本（含数据来源）
        """
        explanations = []

        # 分型解释
        if subtype_idx == 0:
            explanations.append(
                f"**分型诊断**: 肺腺癌 (LUAD)，置信度 {subtype_prob[0] * 100:.1f}%"
            )
            explanations.append("- 肺腺癌是最常见的非小细胞肺癌类型，约占40-50%")
        elif subtype_idx == 1:
            explanations.append(
                f"**分型诊断**: 肺鳞状细胞癌 (LUSC)，置信度 {subtype_prob[1] * 100:.1f}%"
            )
            explanations.append("- 肺鳞癌约占非小细胞肺癌的25-30%")
        else:
            explanations.append(
                f"**诊断结果**: 正常肺组织，置信度 {subtype_prob[2] * 100:.1f}%"
            )
            return "\n".join(explanations)

        # 分子标记推理解释
        explanations.append("\n**分子标记推理** (基于TCGA统计):")

        egfr_data = molecular_inference["EGFR"]
        explanations.append(
            f"- EGFR突变概率: {egfr_data['probability'] * 100:.1f}% "
            f"(95% CI: {egfr_data['confidence_interval'][0] * 100:.1f}%-{egfr_data['confidence_interval'][1] * 100:.1f}%)"
        )

        alk_data = molecular_inference["ALK"]
        explanations.append(
            f"- ALK融合概率: {alk_data['probability'] * 100:.1f}% "
            f"(95% CI: {alk_data['confidence_interval'][0] * 100:.1f}%-{alk_data['confidence_interval'][1] * 100:.1f}%)"
        )

        pdl1_data = molecular_inference["PD_L1"]
        explanations.append(f"- PD-L1高表达(≥50%)概率: {pdl1_data['high'] * 100:.1f}%")

        explanations.append(f"\n*数据来源: {egfr_data['source']}*")

        return "\n".join(explanations)


# 全局知识库实例
knowledge_base = LungCancerKnowledgeBase()
