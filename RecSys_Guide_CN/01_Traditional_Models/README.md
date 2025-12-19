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

# 01. 传统模型 (Traditional Models)

本章涵盖了推荐系统中的基础算法。

## 类别 (Categories)

### 1. [协同过滤 (Collaborative Filtering)](./01_Collaborative_Filtering/README.md)

利用用户-物品的交互数据（评分、点击）来寻找模式。

- **基于记忆 (Memory-based)**: 直接使用原始数据（最近邻算法）。
- **基于模型 (Model-based)**: 从数据中学习一个预测模型（矩阵分解）。

### 2. [基于内容的过滤 (Content-based Filtering)](./02_Content_Based_Filtering/README.md)

根据物品的属性（元数据、文本描述），推荐与用户过去喜欢的物品相似的物品。
