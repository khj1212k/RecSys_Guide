[< Up to Parent](../README.md)

<details>
<summary><strong>Global Navigation</strong></summary>

- [Home](../README.md)
- [01. Traditional Models](../01_Traditional_Models/README.md)
    - [Collaborative Filtering](../01_Traditional_Models/01_Collaborative_Filtering/README.md)
        - [Memory-based](../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
        - [Model-based](../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
    - [Content-based Filtering](../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. Machine Learning Era](../02_Machine_Learning_Era/README.md)
- [03. Deep Learning Era](../03_Deep_Learning_Era/README.md)
    - [MLP-based](../03_Deep_Learning_Era/01_MLP_Based/README.md)
    - [Sequence/Session-based](../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
    - [Graph-based](../03_Deep_Learning_Era/03_Graph_Based/README.md)
    - [AutoEncoder-based](../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. SOTA & GenAI](../04_SOTA_GenAI/README.md)
    - [LLM-based](../04_SOTA_GenAI/01_LLM_Based/README.md)
    - [Multimodal RS](../04_SOTA_GenAI/02_Multimodal_RS.md)
    - [Generative RS](../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 03. Deep Learning Era

The introduction of Deep Learning brought non-linearity and Representation Learning capabilities to recommendation systems.

## Classification

### 1. [MLP-based](./01_MLP_Based/README.md)

Uses the most basic neural network structure to model user-item interactions. (NCF, Wide & Deep)

### 2. [Sequence/Session-based](./02_Sequence_Session_Based/README.md)

Models the sequence of user actions over time. Utilizes RNN, LSTM, Transformer (Attention), etc. (GRU4Rec, SASRec)

### 3. [Graph-based](./03_Graph_Based/README.md)

Views user-item interactions as a graph structure and utilizes GNNs (Graph Neural Networks) to capture high-order connectivity. (NGCF, LightGCN)

### 4. [AutoEncoder-based](./04_AutoEncoder_Based/README.md)

Fills in missing rating information during the process of compressing and reconstructing input data. (AutoRec, CDAE)