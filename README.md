# 부모 복제 AI 모델 파이프라인 : Cloning AI

<div><img src="./image/MAIN01.png" width="800px;" alt=""/></div>

## 1. AI 개요

* 부모의 **육아일기·대화 기록·음성 데이터**를 활용하여
  **LLM + RAG + TTS 파이프라인**으로 사고방식과 음성을 복제한 AI 모델을 구축
* 부모가 제공한 텍스트 맥락을 기반으로 **개인화된 대화형 모델 생성**,
  TTS를 통해 **부모 음성과 유사한 형태로 응답 생성**
* 자녀가 정서적·심리적 지지를 필요로 하는 순간,
  부모의 말투와 사고 패턴을 반영한 **맞춤형 상담·대화 AI 제공**



## 2. AI 아키텍처 개요

* **LLM:** Llama3 Korean Blossom 8B를 기반으로 공감형 대화 특화 파인튜닝
* **RAG:** 부모의 육아일기·대화 기록을 벡터화하여 문맥 기반 답변 생성
* **TTS:** XTTS 기반 음성 합성으로 부모 음색을 복원
* **Embedding:** sRoBERTa 멀티태스크 기반 문맥 임베딩 추출
* **Pipeline:**
  텍스트 전처리 → LLM 학습 → 문서 벡터 인덱싱 → RAG 검색 → 음성 합성



## 3. 모델 세부 구성

## 📘 3-1. LLM 파인튜닝

<div><img src="./image/LLM01.png" width="800px;" alt=""/></div>
<div><img src="./image/LLM02.png" width="800px;" alt=""/></div>

* Blossom-8B 모델 기반
* 부모-자녀 대화 구조화 데이터로 Instruction Fine-tuning
* AI Hub 공감형 대화 데이터 추가 학습
* LoRA 적용을 통한 파라미터 효율적 학습
* 역할 구분(prompt role) 및 문맥 강화 처리



## 🔍 3-2. RAG (Retrieval-Augmented Generation)

<div><img src="./image/RAG01.png" width="800px;" alt=""/></div>
<div><img src="./image/RAG02.png" width="800px;" alt=""/></div>

* LangChain + FAISS 기반 검색 구조
* 육아일기·상담기록 등을 chunking 후 벡터 인덱싱
* 검색된 실제 부모 기록을 LLM 답변 생성에 주입하여
  **더 일관적이고 부모 특화 맥락을 반영한 대화 생성**



## 🎧 3-3. TTS (XTTS 기반 Voice Cloning)

<div><img src="./image/tts01.png" width="800px;" alt=""/></div>
<div><img src="./image/tts02.png" width="800px;" alt=""/></div>

* 6초 분량 부모 음성으로 화자 임베딩 생성
* 텍스트 → 멜스펙트로그램 → waveform으로 단계적 합성
* 한국어 음성 스타일에 최적화된 XTTS v2 모델 적용
* 사용자 질문 → 부모 음성으로 응답 생성 end-to-end 처리

<br>

## 4. 개발 스택

### 🧠 AI / Machine Learning

<div> <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/langchain-%231C3C3C?style=for-the-badge&logo=langchain&logoColor=white"> <img src="https://img.shields.io/badge/Llama 3 Korean Blossom 8B-FF8C00?style=for-the-badge&logo=meta&logoColor=white"> <img src="https://img.shields.io/badge/XTTS-009630?style=for-the-badge&logo=coqui&logoColor=white"> <img src="https://img.shields.io/badge/Conversational Dataset-0066FF?style=for-the-badge&logo=google&logoColor=white"> </div>


### 📦 Dataset

<div><img src="https://img.shields.io/badge/AI Hub-0066FF?style=for-the-badge&logo=google&logoColor=white"></div>


## 6. 참고 자료

* 공감형 대화 데이터셋 (AI Hub)
* Llama3 Korean Blossom 8B
* XTTS v2 (Coqui)
* FAISS Dense Vector Index
