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

# 基于自编码器的模型 (AutoEncoder-based Models)

自编码器是一种将输入复制到输出的神经网络，但通过在中间放置一个“瓶颈”来学习数据的压缩特征。在推荐系统中，这种 **重建** 能力被用于“填补空白（评分补全）”。

## 类型 (Types)

### 1. [AutoRec / CDAE](./01_AutoRec_CDAE.md)

- **AutoRec**: 以用户的评分向量为输入，并通过填补空缺的评分来重建它。
- **CDAE**: 使用降噪（在输入中混合噪声）来构建更鲁棒的模型，并专门用于 Top-N 推荐。
