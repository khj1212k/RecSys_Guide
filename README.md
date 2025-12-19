# RecSys Guide 📚

> **A Comprehensive Guide to Recommender Systems**  
> From Traditional Collaborative Filtering to State-of-the-Art Generative AI Models.

This repository contains a structured educational guide on Recommender Systems, covering the evolution of algorithms from simple heuristics to complex deep learning architectures.

## 🌍 Languages

Please select your preferred language:

- [🇺🇸 **English**](RecSys_Guide_EN/README.md)
- [🇰🇷 **한국어 (Korean)**](RecSys_Guide_KO/README.md)
- [🇨🇳 **简体中文 (Simplified Chinese)**](RecSys_Guide_CN/README.md)

## 📂 Project Structure

```text
Recommender Systems
├── 01. Traditional/Classical Models
│   ├── Collaborative Filtering
│   │   ├── Memory-based
│   │   │   ├── User-based CF
│   │   │   └── Item-based CF
│   │   └── Model-based
│   │       ├── Matrix Factorization: SVD, ALS
│   │       └── Latent Factor Models
│   └── Content-based Filtering
│       ├── TF-IDF / Cosine Similarity
│       └── Profile-based Matching
│
├── 02. Machine Learning Era
│   ├── Hybrid Methods
│   └── Factorization Machines
│       ├── FM (Factorization Machines)
│       └── FFM (Field-aware FM)
│
├── 03. Deep Learning Era
│   ├── MLP-based
│   │   ├── NCF (Neural Collaborative Filtering)
│   │   └── Wide & Deep Learning
│   ├── Sequence/Session-based
│   │   ├── GRU4Rec
│   │   └── SASRec / BERT4Rec
│   ├── Graph-based
│   │   ├── NGCF (Neural Graph Collaborative Filtering)
│   │   └── LightGCN
│   └── AutoEncoder-based
│       └── AutoRec / CDAE
│
└── 04. State-of-the-Art / GenAI
    ├── LLM-based RS
    │   ├── LLM4Rec
    │   └── P5 (Pretrain, Personalized, Prompt, Predict, Recommendation)
    ├── Multimodal RS: Image/Text Combination
    └── Generative RS
```

---

## 📖 Content Overview

This guide is structured into four major eras of Recommender Systems development:

### [1. Traditional Models](./RecSys_Guide_EN/01_Traditional_Models/README.md)

The foundational algorithms that started it all.

- **Collaborative Filtering**: User-based, Item-based, Matrix Factorization.
- **Content-based Filtering**: TF-IDF, Profile Matching.

### [2. Machine Learning Era](./RecSys_Guide_EN/02_Machine_Learning_Era/README.md)

The transition to statistical learning and feature interaction modeling.

- **Hybrid Models**
- **Factorization Machines (FM, FFM)**

### [3. Deep Learning Era](./RecSys_Guide_EN/03_Deep_Learning_Era/README.md)

The rise of neural networks to capture non-linear relationships.

- **MLP-based**: Neural CF, Wide & Deep.
- **Sequence-based**: RNN (GRU4Rec), Transformer (SASRec, BERT4Rec).
- **Graph-based**: NGCF, LightGCN.
- **AutoEncoders**: AutoRec, CDAE.

### [4. SOTA & GenAI](./RecSys_Guide_EN/04_SOTA_GenAI/README.md)

The latest trends leveraging Large Language Models and Generative AI.

- **LLM-based**: LLM4Rec, P5.
- **Multimodal RS**: Handling Images and Text.
- **Generative RS**: Generative Retrieval.

---

## 🚀 How to Use

1.  Navigate to your language of choice (English or Korean).
2.  Follow the folders in numerical order (01 -> 04) to understand the history and evolution.
3.  Each section contains detailed explanations, mathematical principles, and flow examples.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>This documentation was generated with the assistance of <strong>Google Gemini</strong>.</em>
</p>
