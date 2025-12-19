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

# 基于记忆的协同过滤 (Memory-based Collaborative Filtering)

基于记忆的 CF 利用整个用户-物品数据库（记忆）来进行预测。它没有可学习的参数，完全依赖于相似度计算。

## 类型 (Types)

### 1. [基于用户的 CF (User-based CF)](./01_User_Based_CF.md)

- “如果有个跟我相似的用户喜欢这个物品，那我可能也会喜欢。”
- 寻找与目标用户口味相似的邻居。

### 2. [基于物品的 CF (Item-based CF)](./02_Item_Based_CF.md)

- “如果我喜欢这个物品，那我可能也会喜欢跟它相似的其他物品。”
- 推荐与用户过去高评分物品相似的物品。
