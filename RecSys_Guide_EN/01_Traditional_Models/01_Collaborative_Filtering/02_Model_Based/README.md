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

# Model-based Collaborative Filtering

Model-based approaches use the dataset to learn a mathematical model that approximates the underlying patterns. Instead of memorizing the original data, it learns a "compressed version" (parameters) of the data.

## Advantages

- **Space Efficiency**: Only needs to store small parameter matrices instead of a giant matrix.
- **Speed**: Prediction is very fast once the model is trained.
- **Overcoming Sparsity**: Excellent at filling in the blanks (Completion).

## Types

### 1. [Matrix Factorization](./01_Matrix_Factorization.md)

Maps users and items to a low-dimensional Latent Space, approximating the rating matrix as the product of two smaller matrices. (e.g., SVD, ALS, SGD)

### 2. [Latent Factor Models](./02_Latent_Factor_Models.md)

Identifies hidden 'factors' (e.g., genre, mood) that explain the observed rating patterns.