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

# 03. 深度学习时代 (Deep Learning Era)

深度学习的引入为推荐系统带来了非线性和表示学习的能力。

## 分类 (Classification)

### 1. [MLP 基 (MLP-based)](./01_MLP_Based/README.md)

使用最基本的神经网络结构来建模用户-物品交互。(NCF, Wide & Deep)

### 2. [基于序列/会话 (Sequence/Session-based)](./02_Sequence_Session_Based/README.md)

建模用户随时间变化的行为序列。利用 RNN, LSTM, Transformer (Attention) 等。(GRU4Rec, SASRec)

### 3. [基于图 (Graph-based)](./03_Graph_Based/README.md)

将用户-物品交互视为图结构，并利用 GNN（图神经网络）捕捉高阶连接性。(NGCF, LightGCN)

### 4. [基于自编码器 (AutoEncoder-based)](./04_AutoEncoder_Based/README.md)

在压缩和重建输入数据的过程中填充缺失的评分信息。(AutoRec, CDAE)
