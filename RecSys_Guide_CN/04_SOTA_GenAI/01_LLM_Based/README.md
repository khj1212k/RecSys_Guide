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

# 基于 LLM 的模型 (LLM-based Models)

将 **大语言模型 (LLM)** （如 GPT, BERT, T5, LLaMA）引入推荐系统的模型。它们利用 LLM 庞大的世界知识和推理能力来克服传统基于 ID 的推荐的局限性（例如，冷启动、缺乏解释）。

## 类型 (Types)

### 1. [LLM4Rec](./01_LLM4Rec.md)

- 使用 LLM 作为组件（特征提取器）或直接作为推荐器来增强性能。

### 2. [P5](./02_P5.md)

- **P5** (Pretrain, Personalized Prompt, Prediction Paradigm): 使用提示（prompts）将各种推荐任务（顺序、评分、解释）统一为单一的文本到文本格式。
