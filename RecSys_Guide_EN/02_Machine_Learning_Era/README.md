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

# 02. Machine Learning Era

This era marked the transition from simple matrix operations and statistical approaches to more complex machine learning techniques. It dominated the industry until the rise of Deep Learning.

## Key Models

### 1. [Hybrid Models](./01_Hybrid_Models.md)

Combines multiple recommendation algorithms (e.g., CF + Content-based) to offset the disadvantages (Cold Start, Data Sparsity) of individual models. Famous for being the winning strategy of the Netflix Prize.

### 2. [Factorization Machines](./02_Factorization_Machines/README.md)

Combines the advantages of Matrix Factorization (MF) and Support Vector Machines (SVM). It effectively models interactions between variables even in sparse data environments and achieved great success in CTR (Click-Through Rate) prediction tasks.