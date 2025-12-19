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

# 基于内容的过滤 (Content-based Filtering)

基于内容的过滤 (CB) 利用物品的 **属性**，较少依赖用户评分数据。它分析物品的元数据（类型、导演、描述、标签等），推荐与用户过去喜欢的物品具有相似特征的物品。

## 主要优势 (Key Advantages)

- **解决冷启动 (新物品)**: 新物品一经注册即可推荐（因为属性是立即可知的）。
- **独立性 (Independence)**: 不需要其他用户的数据。

## 类型 (Types)

### 1. [TF-IDF / 余弦相似度 (Cosine Similarity)](./01_TF_IDF_Cosine_Similarity.md)

一种经典方法，将文本数据（剧情、评论）向量化，并根据文档相似度进行推荐。

### 2. [基于画像的匹配 (Profile-based Matching)](./02_Profile_Based_Matching.md)

为用户和物品创建结构化画像（年龄、类型偏好等），并基于规则或相似度指标进行匹配。
