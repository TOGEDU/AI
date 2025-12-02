# CLONING AI

<div><img src="./image/MAIN01.png" width="800px;" alt=""/></div>

# 😀 백엔드 팀원 및 역할

| 이름   | 역할      | 담당   |
| ------ | --------- | ----- |
| 소원 | Back-end / AI | <details><summary>채팅, 배지, 오늘의질문, RAG, DB, 배포/인프라, ERD</summary><b>🗨️ 실시간 채팅 시스템</b><ul><li>채팅방 생성, 메시지 전송 및 답변 생성 API</li><li>메시지 히스토리 조회 / 채팅방 목록 조회 API</li><li>Spring Boot ↔ FastAPI 간 텍스트 → 음성 합성 연동</li><li>Whisper 기반 음성 → 텍스트(STT) 변환 API</li><li>초기 메시지 기반 채팅방 생성 및 메시지 소속 관리 로직 설계</li></ul><b>🗂️ 기록 현황 / 배지 시스템</b><ul><li>주간 기준 배지 획득 여부 조회</li><li>달력 기반 활동 현황 조회 API</li><li>일기/질문/기록 여부에 따른 UI 표시 데이터 제공</li><li>누적 기록 기반 진행률 계산 로직 구현</li></ul><b>📝 오늘의 질문(Daily Question)</b><ul><li>질문 리스트 조회(아코디언용 전체 텍스트 포함)</li><li>답변 작성/수정/조회 API</li><li>날짜별 답변 유무 기반 분기 처리 로직</li><li>페이징 구조 설계 반영</li><li>자정 기준 자동 질문 로테이션 로직 구현</li></ul><b>🧠 RAG 파이프라인 구축</b><ul><li>LangChain 기반 RAG 구조 설계 및 구현</li><li>텍스트 청크 분할 후 임베딩 생성</li><li>Cosine Similarity 기반 문맥 검색</li><li>검색된 컨텍스트를 활용한 정확한 답변 생성</li></ul><b>🗂️ 데이터베이스 인프라 구성</b><ul><li>ChromaDB 기반 벡터 저장소 초기 구축 및 임베딩 관리 파이프라인 설계</li><li>AWS RDS(MySQL) 기반 서비스 데이터 저장 환경 구성</li></ul><b>☁️ 배포 및 서버 인프라</b><ul><li>Docker 기반 AWS EC2 컨테이너 빌드 & 배포</li><li>RDS–Spring Boot–FastAPI 연동</li><li>CUDA + NVIDIA Docker 기반 GPU 추론 환경 구축</li></ul><b>🧱 ERD 및 도메인 모델링</b><ul><li>전체 서비스 ERD 설계</li><li>도메인 구조화(Child, ChatRoom, Message 등)</li></ul></details>|
| 조소윤 | Back-end / AI | <details><summary>인증/보안, 음성기록, 육아일기, 푸시알림, 부모/자녀 마이페이지, TTS, 배포/인프라, DB, ERD</summary><b>🔐 인증/보안</b><ul><li>부모와 자녀 회원가입 로직 구현</li><li>이메일 중복 검사 API</li><li>Spring Security 기반 JWT 인증/인가 구조 구축</li><li>Access Token 발급 로직 구현</li><li>로그아웃 API</li><li>회원 탈퇴 API</li></ul><b>🎙️ 음성 기록 기능</b><ul><li>TTS 모델 학습용 음성 녹음 업로드 처리</li><li>음성 파일 생성·병합·저장 로직</li><li>음성 기록 진행 현황 API</li></ul><b>📷 육아일기(사진 포함) API</b><ul><li>부모 공통 일기 작성 및 자녀별 개별 일기 작성 API</li><li>일기 작성 전 기존 작성 여부 확인 API</li><li>육아일기 수정/작성/날짜별 기록 조회/캘린더 API 구현</li></ul><b>📱 푸시 알림</b><ul><li>Firebase Cloud Messaging(FCM) 사용</li><li>Spring Boot 스케줄러 기반 지정 시간 푸시 알림</li></ul><b>👨‍👧 부모 마이페이지</b><ul><li>부모 정보 조회 및 수정(이름, 프로필 사진)</li><li>자녀 목록 조회 및 관리(추가, 이름 변경)</li><li>알림 시간 설정(on/off 포함)</li><li>육아일기 캘린더 조회 및 날짜별 일기 확인</li></ul><b>🔊 텍스트 음성 변환(TTS) 기능</b><ul><li>XTTS 모델 하이퍼파라미터 파인튜닝</li><li>XTTS 기반 음성 재생 기능 API</li></ul><b>🧒 자녀 마이페이지</b><ul><li>자녀 정보 조회</li><li>사진첩 조회</li><li>알림 시간 설정(on/off 포함)</li></ul><b>☁️ 배포 및 인프라</b><ul><li>Docker 기반 AWS EC2 컨테이너 빌드 & 배포</li><li>CUDA + NVIDIA Docker 기반 GPU 추론 환경 구축</li><li>S3와 Spring Boot 연동</li></ul><b>🗂️ 데이터베이스 인프라 구성</b><ul><li>FAISS 기반 벡터 저장소 초기 구축 및 임베딩 파이프라인 설계</li></ul><b>🧱 ERD 및 도메인 모델링</b><ul><li>전체 서비스 ERD 설계</li><li>도메인 구조화(DailyQuestion, DailyQuestionRecord, Parent, VoiceRecordingRecord, VoiceRecordingSentence)</li></ul></details> |
| 홍다인 | Back-end  / AI  |<details><summary>음성처리, 육아일기, 사진첩, LLM 전처리, LLM 파인튜닝, 배포/인프라, ERD</summary><b>🔊 부모 음성 처리 기능</b><ul><li>부모 음성 녹음 업로드 처리</li><li>음성 파일 생성·병합·저장 로직</li><li>음성 기록 조회/관리 API</li></ul><b>📷 육아일기(사진 포함) API</b><ul><li>텍스트 + 이미지 업로드/조회 API</li><li>AWS S3 기반 이미지 저장소 구축</li><li>캘린더 기반 전체 육아일기 조회</li><li>날짜별 육아일기 상세 조회</li><li>육아일기 작성 기능</li><li>작성된 일기 수정 기능</li></ul><b>📷 사진첩 기능</b><ul><li>사진 업로드 API</li><li>사진 저장 및 관리 로직</li><li>업로드된 이미지 조회 기능</li><li>S3 기반 사진 파일 저장 구조 적용</li></ul><b>🧠 LLM 데이터 수집·전처리</b><ul><li>AI Hub 공감형 대화 데이터에서 부모–자녀 대화만 선별</li><li>발화 역할 통일(‘어머니/아버지 → 부모’) 및 문장 정제·구조화</li><li>LLM 학습용 Instruction 형식 데이터셋 구성</li></ul><b>🤖 LLM 파인튜닝</b><ul><li>Llama 3 Korean Blossom 8B로 부모 말투·대화 패턴 학습</li><li>전처리된 대화 데이터를 활용해 공감형 응답 생성 모델 구축</li><li>FastAPI 환경에서 학습 모델 로딩하여 실시간 대화 서비스 적용</li></ul><b>☁️ 배포 및 인프라</b><ul><li>Docker 기반 컨테이너화</li><li>Docker 기반 AWS EC2 컨테이너 빌드 & 배포</li><li>EC2에서 Spring Boot · FastAPI 서버 연동</li></ul><b>🧱 ERD 및 도메인 모델링</b><ul><li>전체 서비스 ERD 설계</li><li>도메인 구조화(ParentChild, Diary 등)</li></ul></details> |
<br>

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


## 5. 참고 자료

* 공감형 대화 데이터셋 (AI Hub)
* Llama3 Korean Blossom 8B
* XTTS v2 (Coqui)
* FAISS Dense Vector Index
