[< 返回上一级](../README.md)

<details>
<summary><strong>全局导航 (RecSys Guide)</strong></summary>

- [首页](../README.md)
- [01. 传统模型](../01_Traditional_Models/README.md)
  - [协同过滤](../01_Traditional_Models/01_Collaborative_Filtering/README.md)
    - [基于记忆](../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
    - [基于模型](../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
  - [基于内容的过滤](../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 机器学习时代](../02_Machine_Learning_Era/README.md)
- [03. 深度学习时代](../03_Deep_Learning_Era/README.md)
  - [基于 MLP](../03_Deep_Learning_Era/01_MLP_Based/README.md)
  - [基于序列/会话](../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
  - [基于图](../03_Deep_Learning_Era/03_Graph_Based/README.md)
  - [基于自编码器](../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. SOTA 与生成式 AI](../04_SOTA_GenAI/README.md) - [基于 LLM](../04_SOTA_GenAI/01_LLM_Based/README.md) - [多模态推荐](../04_SOTA_GenAI/02_Multimodal_RS.md) - [生成式推荐](../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 02. 机器学习时代 (Machine Learning Era)

这一时代标志着从简单的矩阵运算和统计方法向更复杂的机器学习技术的转变。在深度学习兴起之前，它一直主导着行业。

## 关键模型 (Key Models)

### 1. [混合模型 (Hybrid Models)](./01_Hybrid_Models.md)

结合多种推荐算法（例如，CF + 基于内容），以抵消单个模型的缺点（冷启动，数据稀疏性）。作为 Netflix Prize 的获胜策略而闻名。

### 2. [因子分解机 (Factorization Machines)](./02_Factorization_Machines/README.md)

结合了矩阵分解 (MF) 和支持向量机 (SVM) 的优点。即使在稀疏数据环境中，它也能有效地对变量之间的交互进行建模，并在 CTR（点击率）预测任务中取得了巨大成功。
