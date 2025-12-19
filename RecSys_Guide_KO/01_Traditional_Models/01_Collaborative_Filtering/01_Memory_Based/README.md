[< 상위 폴더로 이동](../README.md)

<details>
<summary><strong>전체 탐색 (RecSys 가이드)</strong></summary>

- [홈](../../../README.md)
- [01. 전통적 모델](../../../01_Traditional_Models/README.md)
    - [협업 필터링](../../../01_Traditional_Models/01_Collaborative_Filtering/README.md)
        - [메모리 기반](../../../01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
        - [모델 기반](../../../01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
    - [콘텐츠 기반 필터링](../../../01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 과도기 및 통계적 모델](../../../02_Machine_Learning_Era/README.md)
- [03. 딥러닝 기반 모델](../../../03_Deep_Learning_Era/README.md)
    - [MLP 기반](../../../03_Deep_Learning_Era/01_MLP_Based/README.md)
    - [순차/세션 기반](../../../03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
    - [그래프 기반](../../../03_Deep_Learning_Era/03_Graph_Based/README.md)
    - [오토인코더 기반](../../../03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. 최신 및 생성형 모델](../../../04_SOTA_GenAI/README.md)
    - [LLM 기반](../../../04_SOTA_GenAI/01_LLM_Based/README.md)
    - [멀티모달 추천](../../../04_SOTA_GenAI/02_Multimodal_RS.md)
    - [생성형 추천](../../../04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 메모리 기반 협업 필터링 (Memory-based CF)

메모리 기반 CF는 예측을 수행하기 위해 전체 사용자-아이템 데이터베이스(메모리)를 활용합니다. 학습할 파라미터가 없으며, 유사도 계산에 전적으로 의존합니다.

## 종류

### 1. [사용자 기반 CF (User-based CF)](./01_User_Based_CF.md)

- "나와 비슷한 사용자가 이 아이템을 좋아했다면, 나도 좋아할 것이다."
- 타겟 사용자와 유사한 취향을 가진 이웃들을 찾습니다.

### 2. [아이템 기반 CF (Item-based CF)](./02_Item_Based_CF.md)

- "내가 이 아이템을 좋아했다면, 이와 비슷한 다른 아이템도 좋아할 것이다."
- 사용자가 과거에 높게 평가했던 아이템과 유사한 아이템을 추천합니다.