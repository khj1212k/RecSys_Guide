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

# 01. 전통적 모델 (Traditional Models)

이 섹션에서는 추천 시스템의 기본이 되는 알고리즘들을 다룹니다.

## 분류 (Categories)

### 1. [협업 필터링 (Collaborative Filtering)](./01_Collaborative_Filtering/README.md)

사용자-아이템 상호작용 데이터(평점, 클릭 등)를 활용하여 패턴을 찾습니다.

- **메모리 기반 (Memory-based)**: 원본 데이터를 직접 사용합니다 (최근접 이웃).
- **모델 기반 (Model-based)**: 데이터로부터 예측 모델을 학습합니다 (행렬 분해).

### 2. [콘텐츠 기반 필터링 (Content-based Filtering)](./02_Content_Based_Filtering/README.md)

아이템의 속성(메타데이터, 텍스트 설명)을 기반으로 사용자가 과거에 좋아했던 아이템과 유사한 아이템을 추천합니다.