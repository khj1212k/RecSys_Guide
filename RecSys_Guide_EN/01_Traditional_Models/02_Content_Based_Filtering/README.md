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

# Content-based Filtering

Content-based Filtering (CB) utilizes item **attributes**, relying less on user rating data. It analyzes item metadata (genre, director, description, tags, etc.) to recommend items with similar characteristics to those the user preferred in the past.

## Key Advantages

- **Solving Cold Start (New Item)**: New items can be recommended as soon as they are registered (since attributes are known immediately).
- **Independence**: Does not require data from other users.

## Types

### 1. [TF-IDF / Cosine Similarity](./01_TF_IDF_Cosine_Similarity.md)

A classic method that vectorizes text data (plot, reviews) and recommends based on document similarity.

### 2. [Profile-based Matching](./02_Profile_Based_Matching.md)

Creates structured profiles (age, genre preference, etc.) for users and items and matches them based on rules or similarity metrics.