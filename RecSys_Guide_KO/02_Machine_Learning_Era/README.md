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

# 02. 과도기 및 통계적 모델 (Machine Learning Era)

단순한 행렬 연산과 통계적 접근을 넘어, 보다 복잡한 머신 러닝 기법들이 도입된 시기입니다. 딥러닝이 등장하기 전까지 산업계를 지배했습니다.

## 주요 모델

### 1. [하이브리드 모델 (Hybrid Models)](./01_Hybrid_Models.md)

여러 추천 알고리즘(예: CF + Content-based)을 결합하여 각 알고리즘의 단점(콜드 스타트, 희소성)을 상호 보완합니다. 넷플릭스 프라이즈(Netflix Prize)의 우승 솔루션으로 유명합니다.

### 2. [요인화 기계 (Factorization Machines)](./02_Factorization_Machines/README.md)

행렬 분해(MF)의 장점과 서포트 벡터 머신(SVM)의 장점을 결합했습니다. 희소한 데이터 환경에서도 변수 간의 상호작용을 효과적으로 모델링하여, CTR(클릭률) 예측 등에서 큰 성공을 거두었습니다.