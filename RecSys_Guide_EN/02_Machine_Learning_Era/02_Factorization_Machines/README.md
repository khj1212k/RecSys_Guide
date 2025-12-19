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

# Factorization Machines

Factorization Machines (FM) are supervised learning algorithms designed to model interactions between variables in **sparse data** environments.

## Core Idea

- **SVM + MF**: Combines the general applicability of Support Vector Machines (SVM) with the sparse data handling capability of Matrix Factorization (MF).
- **Linear Time**: Achieves very fast prediction speeds of $O(kn)$.

## Types

### 1. [FM (Basic)](./01_FM.md)

Learns a single latent vector for every feature, modeling pairwise interactions.

### 2. [FFM (Field-aware FM)](./02_FFM.md)

Groups features into 'Fields'. It learns different latent vectors depending on which field the feature interacts with, achieving great success in CTR prediction competitions.