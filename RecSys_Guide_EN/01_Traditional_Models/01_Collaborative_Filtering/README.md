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

# Collaborative Filtering

Collaborative Filtering (CF) is the most prominent technique in traditional recommender systems. It relies on the assumption that "users who had similar tastes in the past will have similar tastes in the future."

## Sub-categories

### 1. [Memory-based CF](./01_Memory_Based/README.md)

Also known as Neighborhood-based CF. Uses the entire database to calculate similarity between users or items.

- **User-based**: Finds users similar to the target user.
- **Item-based**: Finds items similar to those liked by the target user.

### 2. [Model-based CF](./02_Model_Based/README.md)

Uses machine learning algorithms to learn a model that predicts a user's rating for unrated items.

- **Matrix Factorization (SVD, ALS)**
- **Latent Factor Models**