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

# 요인화 기계 (Factorization Machines)

Factorization Machines (FM)은 **희소한 데이터(Sparse Data)** 환경에서 변수 간의 상호작용을 모델링하기 위해 고안된 지도 학습 알고리즘입니다.

## 핵심 아이디어

- **SVM + MF**: 서포트 벡터 머신(SVM)의 일반적인 적용성과 행렬 분해(MF)의 희소 데이터 처리 능력을 결합했습니다.
- **선형 시간**: $O(kn)$의 매우 빠른 예측 속도를 가집니다.

## 종류

### 1. [FM (Basic)](./01_FM.md)

모든 특징(Feature)에 대해 하나의 잠재 벡터(Latent Vector)를 학습하여, 2차 상호작용(Pairwise Interaction)을 모델링합니다.

### 2. [FFM (Field-aware FM)](./02_FFM.md)

특징들을 '필드(Field)'로 그룹화합니다. 같은 특징이라도 어떤 필드와 상호작용하느냐에 따라 다른 잠재 벡터를 사용하도록 하여, CTR 예측 대회 등을 휩쓸었습니다.