[< 返回上一级](../README.md)

<details>
<summary><strong>全局导航 (RecSys Guide)</strong></summary>

- [首页](../../../README.md)
- [01. 传统模型](../../../01_Traditional_Models/README.md)
  - [协同过滤](../../../01_Traditional_Models/01_Collaborative_Filtering/README.md)
    - [基于记忆](../../../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
    - [基于模型](../../../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
  - [基于内容的过滤](../../../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 机器学习时代](../../../02_Machine_Learning_Era/README.md)
- [03. 深度学习时代](../../../03_Deep_Learning_Era/README.md)
  - [基于 MLP](../../../03_Deep_Learning_Era/01_MLP_Based/README.md)
  - [基于序列/会话](../../../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
  - [基于图](../../../03_Deep_Learning_Era/03_Graph_Based/README.md)
  - [基于自编码器](../../../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. SOTA 与生成式 AI](../../../04_SOTA_GenAI/README.md) - [基于 LLM](../../../04_SOTA_GenAI/01_LLM_Based/README.md) - [多模态推荐](../../../04_SOTA_GenAI/02_Multimodal_RS.md) - [生成式推荐](../../../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 基于模型的协同过滤 (Model-based Collaborative Filtering)

基于模型的方法使用数据集来学习一个逼近潜在模式的数学模型。它不再记住原始数据，而是学习数据的“压缩版本”（参数）。

## 优势 (Advantages)

- **空间效率 (Space Efficiency)**: 只需存储小的参数矩阵，而不是巨大的矩阵。
- **速度 (Speed)**: 一旦模型训练完成，预测速度非常快。
- **克服稀疏性 (Overcoming Sparsity)**: 擅长填补空白（补全）。

## 类型 (Types)

### 1. [矩阵分解 (Matrix Factorization)](./01_Matrix_Factorization.md)

将用户和物品映射到低维潜在空间，将评分矩阵近似为两个较小矩阵的乘积。(例如：SVD, ALS, SGD)

### 2. [隐因子模型 (Latent Factor Models)](./02_Latent_Factor_Models.md)

识别解释观察到的评分模式的隐藏“因子”（例如：流派、情绪）。
