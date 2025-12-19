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

# 협업 필터링 (Collaborative Filtering)

협업 필터링 (CF)은 전통적인 추천 시스템에서 가장 두드러진 기술입니다. "과거에 비슷한 취향을 가졌던 사용자들은 미래에도 비슷한 취향을 가질 것이다"라는 가정에 의존합니다.

## 하위 분류 (Sub-categories)

### 1. [메모리 기반 CF (Memory-based)](./01_Memory_Based/README.md)

이웃 기반(Neighborhood-based) CF라고도 합니다. 전체 데이터베이스를 사용하여 사용자 또는 아이템 간의 유사도를 계산합니다.

- **사용자 기반 (User-based)**: 타겟 사용자와 유사한 사용자를 찾습니다.
- **아이템 기반 (Item-based)**: 타겟 사용자가 좋아했던 아이템과 유사한 아이템을 찾습니다.

### 2. [모델 기반 CF (Model-based)](./02_Model_Based/README.md)

머신 러닝 알고리즘을 사용하여 아직 평가되지 않은 아이템에 대한 사용자의 평점을 예측하는 모델을 학습합니다.

- **행렬 분해 (Matrix Factorization: SVD, ALS)**
- **잠재 요인 모델 (Latent Factor Models)**