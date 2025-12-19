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

# 그래프 기반 모델 (Graph-based Models)

사용자와 아이템 간의 관계를 **이분 그래프(Bipartite Graph)**로 모델링합니다. 그래프 신경망(GNN)을 활용하여, 직접 연결된 이웃뿐만 아니라 "이웃의 이웃"(High-order Connectivity) 정보까지 전파(Propagation)하여 임베딩을 학습합니다.

## 종류

### 1. [NGCF (Neural Graph Collaborative Filtering)](./01_NGCF.md)

- GCN(Graph Convolutional Networks)을 추천 시스템에 적용하여, 임베딩 전파(Embedding Propagation) 개념을 도입했습니다.

### 2. [LightGCN](./02_LightGCN.md)

- NGCF에서 불필요한 비선형 활성화 함수와 변환 행렬을 제거하여 성능과 속도를 모두 획기적으로 개선한 모델입니다. 현재 그래프 기반 추천의 표준(Standard)입니다.