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

# 순차 및 세션 기반 추천 (Sequence & Session-based Recommendation)

사용자의 행동은 정적이지 않습니다. 어제 산 물건이 오늘 살 물건에 영향을 줍니다. 이 분야는 "순서(Order)"와 "맥락(Context)"을 포착하는 데 중점을 둡니다.

## 종류

### 1. [GRU4Rec (RNN 기반)](./01_GRU4Rec.md)

- 세션 기반 추천에 최초로 딥러닝(RNN/GRU)을 적용하여, 짧은 세션 내에서의 즉각적인 사용자 의도를 파악합니다.

### 2. [SASRec / BERT4Rec (Transformer 기반)](./02_SASRec_BERT4Rec.md)

- NLP의 혁명인 Transformer(Attention 메커니즘)를 도입했습니다.
- **SASRec**: GPT처럼 왼쪽에서 오른쪽으로 다음 아이템을 예측합니다.
- **BERT4Rec**: BERT처럼 문맥의 양방향을 모두 고려하여 빈 칸(Masked Item)을 예측합니다.