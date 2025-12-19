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

# 오토인코더 기반 모델 (AutoEncoder-based Models)

오토인코더는 입력을 그대로 출력으로 복사하되, 중간에 "병목(Bottleneck)"을 두어 데이터의 압축된 특징을 학습하는 신경망입니다. 추천 시스템에서는 이 **복원(Reconstruction)** 능력을 "빈 칸 채우기(Rating Completion)"에 활용합니다.

## 종류

### 1. [AutoRec / CDAE](./01_AutoRec_CDAE.md)

- **AutoRec**: 사용자의 평점 벡터를 입력으로 받아, 비어 있는 평점까지 채워서 복원합니다.
- **CDAE**: 입력 데이터에 노이즈를 섞어(Denoising) 더 강건한 모델을 만들고, Top-N 추천에 특화되었습니다.