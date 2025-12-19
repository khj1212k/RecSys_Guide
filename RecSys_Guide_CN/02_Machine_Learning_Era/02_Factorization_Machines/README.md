[< 返回上一级](../README.md)

<details>
<summary><strong>全局导航 (RecSys Guide)</strong></summary>

- [首页](../../README.md)
- [01. 传统模型](../../01_Traditional_Models/README.md)
  - [协同过滤](../../01_Traditional_Models/01_Collaborative_Filtering/README.md)
    - [基于记忆](../../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
    - [基于模型](../../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
  - [基于内容的过滤](../../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 机器学习时代](../../02_Machine_Learning_Era/README.md)
- [03. 深度学习时代](../../03_Deep_Learning_Era/README.md)
  - [基于 MLP](../../03_Deep_Learning_Era/01_MLP_Based/README.md)
  - [基于序列/会话](../../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
  - [基于图](../../03_Deep_Learning_Era/03_Graph_Based/README.md)
  - [基于自编码器](../../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. SOTA 与生成式 AI](../../04_SOTA_GenAI/README.md) - [基于 LLM](../../04_SOTA_GenAI/01_LLM_Based/README.md) - [多模态推荐](../../04_SOTA_GenAI/02_Multimodal_RS.md) - [生成式推荐](../../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 因子分解机 (Factorization Machines)

因子分解机 (FM) 是旨在 **稀疏数据** 环境中对变量间交互进行建模的监督学习算法。

## 核心思想 (Core Idea)

- **SVM + MF**: 结合了支持向量机 (SVM) 的通用性和矩阵分解 (MF) 处理稀疏数据的能力。
- **线性时间 (Linear Time)**: 实现了 $O(kn)$ 的极快预测速度。

## 类型 (Types)

### 1. [FM (Basic)](./01_FM.md)

为每个特征学习一个潜在向量，建模成对交互。

### 2. [FFM (Field-aware FM)](./02_FFM.md)

将特征分为“域 (Fields)”。它根据特征与之交互的域学习不同的潜在向量，在 CTR 预测竞赛中取得了巨大成功。
