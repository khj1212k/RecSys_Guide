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

# 콘텐츠 기반 필터링 (Content-based Filtering)

콘텐츠 기반 필터링 (CB)은 아이템의 **속성(Attributes)**을 활용하므로 사용자의 평점 데이터에 덜 의존합니다. 아이템 자체의 메타데이터(장르, 감독, 설명, 태그 등)를 분석하여 사용자가 과거에 선호했던 아이템과 유사한 특성을 가진 아이템을 추천합니다.

## 핵심 장점

- **콜드 스타트 해결 (Cold Start - New Item)**: 새로운 아이템이 시스템에 등록되자마자 추천될 수 있습니다 (속성은 바로 알 수 있으므로).
- **독립성**: 다른 사용자의 데이터가 필요 없습니다.

## 종류

### 1. [TF-IDF / 코사인 유사도](./01_TF_IDF_Cosine_Similarity.md)

텍스트 데이터(줄거리, 리뷰)를 벡터화하여 문서 유사도를 기반으로 추천하는 고전적인 방법입니다.

### 2. [프로필 기반 매칭 (Profile-based Matching)](./02_Profile_Based_Matching.md)

사용자와 아이템의 구조화된 프로필(나이, 장르 선호도 등)을 만들어 규칙이나 유사도 메트릭으로 매칭합니다.