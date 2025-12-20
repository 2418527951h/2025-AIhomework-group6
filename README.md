# 2025-AIhomework-group5
## 大模型微调（数学任务优化）项目 README

### 项目概述

本项目聚焦大模型在数学任务中的性能优化，基于 Qwen3-1.7B 基础模型，通过监督微调（SFT）、偏好对齐类直接优化（GRPO/DPO/ORPO/KTO/RLOO）及低资源微调方法（LoRA），结合 DeepMath 数学数据集完成模型微调，最终提升模型在 AIME 等数学基准测试中的推理能力。项目依托 Llama-Factory、trl主流框架实现，适配仓库现有目录结构，兼顾训练多样性与实用性。
### 核心技术栈

+ 基础模型：Qwen3-1.7B
+ 数据集：DeepMath（10w+ 数学问题、解法及答案样本）
+ 微调框架：Llama-Factory（开源 LLM 微调框架，支持 100 + 模型）、trl（强化学习训练框架）
+ 微调方法：
    + 监督微调（SFT）
    + 偏好对齐算法（GRPO/DPO/ORPO/KTO/RLOO）
    + 低资源优化（LoRA，适配轻量化训练需求）
+ 硬件要求：8*80GB GPU
## 项目结构（基于仓库目录）
```
2025-AIhomework-group5/
├── eval/               # 评估模块
├── grpo/               # GRPO算法
├── kto/                # KTO算法
├── orpo/               # ORPO算法
├── rloo/               # RLOO算法
├── sft/
│   └── LLaMA-Factory/  # SFT训练模块
└── README.md           # 项目说明文档
```

## 实验结果

基准测试性能（AIME 数据集）

|模型|AIME-2024 准确率|AIME-2025 准确率|
| ---------- | ---------- |----------| 
|Qwen3-1.7B（基模）	|13.33%（4/30）	|15.63%（5/32）|
|Qwen3-1.7B-SFT	|26.67%（8/30）	|26.67%（8/30）|
|Qwen3-1.7B-RL（GRPO）	|20.00%（6/30）	|20.00%（6/30）|

## 结论
+ SFT 微调使模型数学推理能力显著提升，准确率较基模翻倍。
+ 偏好对齐算法（GRPO/KTO/ORPO/RLOO）均能优化模型输出一致性，性能受训练参数、数据质量影响。
+ LoRA 低资源微调方案适配消费级 GPU，推理时无额外开销，便于落地使用。

+ **github仓库链接**：[https://github.com/2418527951h/2025-AIhomework-group5](https://github.com/2418527951h/2025-AIhomework-group5)

+ **项目成员**：陈浩宇 黄钰 杨崇昊 杨军 董鹄铭 黄振庭 李一豪 王怡萱 陈嘉伟 倪云昊 
