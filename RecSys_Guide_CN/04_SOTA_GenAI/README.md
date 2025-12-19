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

# 04. SOTA 与生成式 AI (SOTA & GenAI)

大语言模型 (LLM) 和生成式 AI 的出现正在将推荐系统的范式从“预测”转变为“生成”。

## 主要趋势 (Major Trends)

### 1. [基于 LLM (LLM-based)](./01_LLM_Based/README.md)

利用像 GPT 和 LLaMA 这样的大规模模型的世界知识和推理能力进行推荐。(LLM4Rec, P5)

### 2. [多模态推荐 (Multimodal RS)](./02_Multimodal_RS.md)

结合各种模态（如图像、视频和音频，而不仅仅是文本）来更好地理解物品并提供推荐。

### 3. [生成式推荐 (Generative RS)](./03_Generative_RS.md)

最具创新性的方法。它不像传统方法那样从数百万个物品中检索，而是直接生成下一个物品的 ID，就像 ChatGPT 生成单词一样。
