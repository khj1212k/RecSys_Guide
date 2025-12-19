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

# 04. 최신 및 생성형 모델 (SOTA & GenAI)

거대 언어 모델(LLM)과 생성형 AI의 등장은 추천 시스템의 패러다임을 "예측(Prediction)"에서 "생성(Generation)"으로 바꾸고 있습니다.

## 주요 동향

### 1. [LLM 기반 (LLM-based)](./01_LLM_Based/README.md)

- GPT, LLaMA와 같은 초거대 모델의 세계 지식(World Knowledge)과 추론 능력을 추천에 활용합니다. (LLM4Rec, P5)

### 2. [멀티모달 추천 (Multimodal RS)](./02_Multimodal_RS.md)

- 텍스트뿐만 아니라 이미지, 비디오, 오디오 등 다양한 모달리티를 결합하여 아이템을 더 풍부하게 이해하고 추천합니다.

### 3. [생성형 추천 (Generative RS)](./03_Generative_RS.md)

- 가장 혁신적인 접근법입니다. 수백만 개의 아이템을 검색(Retrieval)하는 대신, ChatGPT가 단어를 생성하듯이 다음에 올 아이템의 ID를 직접 생성해냅니다.