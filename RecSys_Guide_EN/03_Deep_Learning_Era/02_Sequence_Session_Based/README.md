[< Up to Parent](../README.md)

<details>
<summary><strong>Global Navigation</strong></summary>

- [Home](../../README.md)
- [01. Traditional Models](../../01_Traditional_Models/README.md)
    - [Collaborative Filtering](../../01_Traditional_Models/01_Collaborative_Filtering/README.md)
        - [Memory-based](../../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
        - [Model-based](../../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
    - [Content-based Filtering](../../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. Machine Learning Era](../../02_Machine_Learning_Era/README.md)
- [03. Deep Learning Era](../../03_Deep_Learning_Era/README.md)
    - [MLP-based](../../03_Deep_Learning_Era/01_MLP_Based/README.md)
    - [Sequence/Session-based](../../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
    - [Graph-based](../../03_Deep_Learning_Era/03_Graph_Based/README.md)
    - [AutoEncoder-based](../../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. SOTA & GenAI](../../04_SOTA_GenAI/README.md)
    - [LLM-based](../../04_SOTA_GenAI/01_LLM_Based/README.md)
    - [Multimodal RS](../../04_SOTA_GenAI/02_Multimodal_RS.md)
    - [Generative RS](../../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# Sequence & Session-based Recommendation

User behavior is not static. What you bought yesterday influences what you buy today. This field focuses on capturing "Order" and "Context".

## Types

### 1. [GRU4Rec (RNN-based)](./01_GRU4Rec.md)

- Applies Deep Learning (RNN/GRU) to session-based recommendation for the first time, capturing immediate user intent within short sessions.

### 2. [SASRec / BERT4Rec (Transformer-based)](./02_SASRec_BERT4Rec.md)

- Introduce the Transformer (Attention mechanism), a revolution in NLP.
- **SASRec**: Predicts the next item from left to right like GPT.
- **BERT4Rec**: Predicts masked items by considering bidirectional context like BERT.