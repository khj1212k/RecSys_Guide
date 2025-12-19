[< 상위 폴더로 이동](../README.md)

<details>
<summary><strong>전체 탐색 (RecSys 가이드)</strong></summary>

- [홈](../../README.md)
- [01. 전통적 모델](../../01_Traditional_Models/README.md)
    - [협업 필터링](../../01_Traditional_Models/01_Collaborative_Filtering/README.md)
        - [메모리 기반](../../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
        - [모델 기반](../../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
    - [콘텐츠 기반 필터링](../../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 과도기 및 통계적 모델](../../02_Machine_Learning_Era/README.md)
- [03. 딥러닝 기반 모델](../../03_Deep_Learning_Era/README.md)
    - [MLP 기반](../../03_Deep_Learning_Era/01_MLP_Based/README.md)
    - [순차/세션 기반](../../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
    - [그래프 기반](../../03_Deep_Learning_Era/03_Graph_Based/README.md)
    - [오토인코더 기반](../../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. 최신 및 생성형 모델](../../04_SOTA_GenAI/README.md)
    - [LLM 기반](../../04_SOTA_GenAI/01_LLM_Based/README.md)
    - [멀티모달 추천](../../04_SOTA_GenAI/02_Multimodal_RS.md)
    - [생성형 추천](../../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# MLP 기반 모델 (MLP-based Models)

다층 퍼셉트론(Multi-Layer Perceptron, MLP)은 딥러닝의 가장 기본적인 형태입니다. 추천 시스템에서는 주로 행렬 분해가 사용하던 "내적(Dot Product)"이라는 선형적인 상호작용 계산 방식을, "신경망을 통한 비선형적 학습"으로 대체하는 데 사용됩니다.

## 종류

### 1. [NCF (Neural Collaborative Filtering)](./01_NCF.md)

- 행렬 분해를 일반화한 모델입니다. 사용자와 아이템 벡터를 결합(Concatenate)하여 MLP에 통과시켜, 복잡한 비선형적 상호작용을 학습합니다.

### 2. [Wide & Deep Learning](./02_Wide_and_Deep.md)

- Google이 제안한 모델로, 선형 모델(Wide)의 암기 능력(Memorization)과 딥 모델(Deep)의 일반화 능력(Generalization)을 결합했습니다.