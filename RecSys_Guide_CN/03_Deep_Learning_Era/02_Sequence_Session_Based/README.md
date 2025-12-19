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

# 序列与基于会话的推荐 (Sequence & Session-based Recommendation)

用户行为不是静态的。你昨天买的东西会影响你今天买的东西。该领域专注于捕捉“顺序”和“上下文”。

## 类型 (Types)

### 1. [GRU4Rec (RNN-based)](./01_GRU4Rec.md)

- 首次将深度学习 (RNN/GRU) 应用于基于会话的推荐，捕捉短会话中的即时用户意图。

### 2. [SASRec / BERT4Rec (Transformer-based)](./02_SASRec_BERT4Rec.md)

- 引入了 Transformer（注意力机制），即 NLP 领域的一场革命。
- **SASRec**: 像 GPT 一样从左到右预测下一个物品。
- **BERT4Rec**: 像 BERT 一样通过考虑双向上下文来预测被掩盖的物品。
