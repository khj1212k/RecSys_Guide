[< 상위 폴더로 이동](../README.md)

<details>
<summary><strong>전체 탐색 (RecSys 가이드)</strong></summary>

- [홈](../README.md)
- [01. 전통적 모델](../01_Traditional_Models/README.md)
    - [협업 필터링](../01_Traditional_Models/01_Collaborative_Filtering/README.md)
        - [메모리 기반](../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
        - [모델 기반](../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
    - [콘텐츠 기반 필터링](../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 과도기 및 통계적 모델](../02_Machine_Learning_Era/README.md)
- [03. 딥러닝 기반 모델](../03_Deep_Learning_Era/README.md)
    - [MLP 기반](../03_Deep_Learning_Era/01_MLP_Based/README.md)
    - [순차/세션 기반](../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
    - [그래프 기반](../03_Deep_Learning_Era/03_Graph_Based/README.md)
    - [오토인코더 기반](../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. 최신 및 생성형 모델](../04_SOTA_GenAI/README.md)
    - [LLM 기반](../04_SOTA_GenAI/01_LLM_Based/README.md)
    - [멀티모달 추천](../04_SOTA_GenAI/02_Multimodal_RS.md)
    - [생성형 추천](../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 03. 딥러닝 기반 모델 (Deep Learning Era)

딥러닝의 도입은 추천 시스템에 비선형성(Non-linearity)과 표현 학습(Representation Learning) 능력을 가져왔습니다.

## 분류

### 1. [MLP 기반 (다층 퍼셉트론 기반)](./01_MLP_Based/README.md)

- 가장 기본적인 신경망 구조를 사용하여 사용자와 아이템의 상호작용을 모델링합니다. (NCF, Wide & Deep)

### 2. [순차/세션 기반 (순차/세션 기반)](./02_Sequence_Session_Based/README.md)

- 시간의 흐름에 따른 사용자의 행동 순서를 모델링합니다. RNN, LSTM, Transformer(Attention) 등을 사용합니다. (GRU4Rec, SASRec)

### 3. [그래프 기반 (그래프 기반)](./03_Graph_Based/README.md)

- 사용자와 아이템 간의 상호작용을 그래프 구조로 보고, GNN(Graph Neural Networks)을 활용하여 고차원 연결성을 포착합니다. (NGCF, LightGCN)

### 4. [오토인코더 기반 (오토인코더 기반)](./04_AutoEncoder_Based/README.md)

- 입력 데이터를 압축했다가 복원하는 과정에서 손실된 평점 정보를 채워 넣습니다. (AutoRec, CDAE)