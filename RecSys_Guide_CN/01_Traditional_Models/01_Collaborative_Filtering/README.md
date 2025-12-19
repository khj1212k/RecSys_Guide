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

# 协同过滤 (Collaborative Filtering)

协同过滤 (CF) 是传统推荐系统中最著名的技术。它基于这样一个假设：“过去有相似偏好的用户，在未来也会有相似的偏好。”

## 子类别 (Sub-categories)

### 1. [基于记忆的 CF (Memory-based CF)](./01_Memory_Based/README.md)

也称为基于邻域的 CF。使用整个数据库来计算用户或物品之间的相似度。

- **基于用户 (User-based)**: 寻找与目标用户相似的用户。
- **基于物品 (Item-based)**: 寻找与目标用户喜欢的物品相似的物品。

### 2. [基于模型的 CF (Model-based CF)](./02_Model_Based/README.md)

使用机器学习算法学习一个模型，以预测用户对未评分物品的评分。

- **矩阵分解 (Matrix Factorization - SVD, ALS)** -**隐因子模型 (Latent Factor Models)**
