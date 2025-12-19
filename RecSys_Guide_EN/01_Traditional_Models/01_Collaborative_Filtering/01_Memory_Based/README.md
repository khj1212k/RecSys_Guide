[< Up to Parent](../README.md)

<details>
<summary><strong>Global Navigation</strong></summary>

- [Home](../../../README.md)
- [01. Traditional Models](../../../01_Traditional_Models/README.md)
    - [Collaborative Filtering](../../../01_Traditional_Models/01_Collaborative_Filtering/README.md)
        - [Memory-based](../../../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
        - [Model-based](../../../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
    - [Content-based Filtering](../../../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. Machine Learning Era](../../../02_Machine_Learning_Era/README.md)
- [03. Deep Learning Era](../../../03_Deep_Learning_Era/README.md)
    - [MLP-based](../../../03_Deep_Learning_Era/01_MLP_Based/README.md)
    - [Sequence/Session-based](../../../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
    - [Graph-based](../../../03_Deep_Learning_Era/03_Graph_Based/README.md)
    - [AutoEncoder-based](../../../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. SOTA & GenAI](../../../04_SOTA_GenAI/README.md)
    - [LLM-based](../../../04_SOTA_GenAI/01_LLM_Based/README.md)
    - [Multimodal RS](../../../04_SOTA_GenAI/02_Multimodal_RS.md)
    - [Generative RS](../../../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# Memory-based Collaborative Filtering

Memory-based CF utilizes the entire user-item database (memory) to make predictions. It has no learnable parameters and relies solely on similarity calculations.

## Types

### 1. [User-based CF](./01_User_Based_CF.md)

- "If a user similar to me liked this item, I will probably like it too."
- Finds neighbors with similar tastes to the target user.

### 2. [Item-based CF](./02_Item_Based_CF.md)

- "If I liked this item, I will probably like other items similar to it."
- Recommends items similar to those the user rated highly in the past.