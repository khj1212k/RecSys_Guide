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

# 모델 기반 협업 필터링 (Model-based CF)

모델 기반 접근 방식은 데이터셋을 사용하여 기본 패턴을 근사하는 수학적 모델을 학습합니다. 원본 데이터를 모두 기억하는 대신, 데이터의 "압축된 버전"(파라미터)을 학습합니다.

## 장점

- **공간 효율성**: 거대한 행렬 대신 작은 파라미터 행렬만 저장하면 됩니다.
- **속도**: 모델이 학습되고 나면 예측 속도가 매우 빠릅니다.
- **희소성 극복**: 빈 공간을 채우는(Completion) 능력이 뛰어납니다.

## 종류

### 1. [행렬 분해 (Matrix Factorization)](./01_Matrix_Factorization.md)

사용자와 아이템을 저차원 잠재 공간(Latent Space)으로 매핑하여, 평점 행렬을 두 개의 작은 행렬의 곱으로 분해합니다. (예: SVD, ALS, SGD)

### 2. [잠재 요인 모델 (Latent Factor Models)](./02_Latent_Factor_Models.md)

관찰된 평점 패턴을 설명할 수 있는 숨겨진 '요인(Factor)'(예: 장르, 분위기)을 찾아냅니다.