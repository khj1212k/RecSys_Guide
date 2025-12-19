<details>
<summary><strong>전체 탐색 (RecSys 가이드)</strong></summary>

- [홈](./README.md)
- [01. 전통적 모델](./01_Traditional_Models/README.md)
    - [협업 필터링](./01_Traditional_Models/01_Collaborative_Filtering/README.md)
        - [메모리 기반](./01_Traditional_Models/01_Collaborative_Filtering/01_Memory_Based/README.md)
        - [모델 기반](./01_Traditional_Models/01_Collaborative_Filtering/02_Model_Based/README.md)
    - [콘텐츠 기반 필터링](./01_Traditional_Models/02_Content_Based_Filtering/README.md)
- [02. 과도기 및 통계적 모델](./02_Machine_Learning_Era/README.md)
- [03. 딥러닝 기반 모델](./03_Deep_Learning_Era/README.md)
    - [MLP 기반](./03_Deep_Learning_Era/01_MLP_Based/README.md)
    - [순차/세션 기반](./03_Deep_Learning_Era/02_Sequence_Session_Based/README.md)
    - [그래프 기반](./03_Deep_Learning_Era/03_Graph_Based/README.md)
    - [오토인코더 기반](./03_Deep_Learning_Era/04_AutoEncoder_Based/README.md)
- [04. 최신 및 생성형 모델](./04_SOTA_GenAI/README.md)
    - [LLM 기반](./04_SOTA_GenAI/01_LLM_Based/README.md)
    - [멀티모달 추천](./04_SOTA_GenAI/02_Multimodal_RS.md)
    - [생성형 추천](./04_SOTA_GenAI/03_Generative_RS.md)
</details>

# 추천 시스템 가이드 (Recommendation Systems Guide)

이 문서는 전통적인 모델부터 최신 생성형 AI 접근 방식까지 추천 시스템에 대한 포괄적인 개요를 제공합니다.

## 목차 (Table of Contents)

### [01. 전통적 모델 (Traditional Models)](./01_Traditional_Models/README.md)

- **협업 필터링 (Collaborative Filtering)**
  - 메모리 기반 (사용자/아이템 기반)
  - 모델 기반 (행렬 분해, 잠재 요인)
- **콘텐츠 기반 필터링 (Content-based Filtering)**
  - TF-IDF / 코사인 유사도
  - 프로필 기반 매칭

### [02. 과도기 및 통계적 모델 (Machine Learning Era)](./02_Machine_Learning_Era/README.md)

- **하이브리드 모델 (Hybrid Models)**
- **요인화 기계 (Factorization Machines - FM, FFM)**

### [03. 딥러닝 기반 모델 (Deep Learning Era)](./03_Deep_Learning_Era/README.md)

- **MLP 기반 (NCF, Wide & Deep)**
- **순차/세션 기반 (GRU4Rec, SASRec/BERT4Rec)**
- **그래프 기반 (NGCF, LightGCN)**
- **오토인코더 기반 (AutoRec/CDAE)**

### [04. 최신 및 생성형 모델 (SOTA & GenAI)](./04_SOTA_GenAI/README.md)

- **LLM 기반 (LLM4Rec, P5)**
- **멀티모달 추천 (Multimodal RS)**
- **생성형 추천 (Generative RS)**