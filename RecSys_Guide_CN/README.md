<details>
<summary><strong>全局导航 (RecSys Guide)</strong></summary>

- [首页](./README.md)
- [01. 传统模型](./01_Traditional_Models/README.md)
  - [协同过滤](./01_Traditional_Models/01_Collaborative_Filtering/README.md)
    - [基于记忆](./01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
    - [基于模型](./01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
  - [基于内容的过滤](./01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 机器学习时代](./02_Machine_Learning_Era/README.md)
- [03. 深度学习时代](./03_Deep_Learning_Era/README.md)
  - [基于 MLP](./03_Deep_Learning_Era/01_MLP_Based/README.md)
  - [基于序列/会话](./03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
  - [基于图](./03_Deep_Learning_Era/03_Graph_Based/README.md)
  - [基于自编码器](./03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. SOTA 与生成式 AI](./04_SOTA_GenAI/README.md) - [基于 LLM](./04_SOTA_GenAI/01_LLM_Based/README.md) - [多模态推荐](./04_SOTA_GenAI/02_Multimodal_RS.md) - [生成式推荐](./04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 推荐系统指南 (Recommendation Systems Guide)

本文档提供了推荐系统的全面概述，涵盖了从传统模型到最新的生成式 AI 方法。

## 目录 (Table of Contents)

### [01. 传统模型 (Traditional Models)](./01_Traditional_Models/README.md)

- **协同过滤 (Collaborative Filtering)**
  - 基于记忆 (基于用户/物品)
  - 基于模型 (矩阵分解, 隐因子模型)
- **基于内容的过滤 (Content-based Filtering)**
  - TF-IDF / 余弦相似度
  - 基于画像的匹配

### [02. 机器学习时代 (Machine Learning Era)](./02_Machine_Learning_Era/README.md)

- **混合模型 (Hybrid Models)** -**因子分解机 (Factorization Machines - FM, FFM)**

### [03. 深度学习时代 (Deep Learning Era)](./03_Deep_Learning_Era/README.md)

- **基于 MLP (MLP-based)**: NCF, Wide & Deep
- **基于序列/会话 (Sequence/Session-based)**: GRU4Rec, SASRec/BERT4Rec
- **基于图 (Graph-based)**: NGCF, LightGCN
- **基于自编码器 (AutoEncoder-based)**: AutoRec/CDAE

### [04. SOTA 与生成式 AI (SOTA & GenAI)](./04_SOTA_GenAI/README.md)

- **基于 LLM (LLM-based)**: LLM4Rec, P5
- **多模态推荐 (Multimodal RS)** -**生成式推荐 (Generative RS)**
