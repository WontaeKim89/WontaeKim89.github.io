---
title: "프로젝트"
description: "직접 설계하고 만들어서 실제 환경에 올린 작업들"
showDate: false
showReadingTime: false
showAuthor: false
showHero: false
showTableOfContents: true
---

직접 설계하고 만들어서 실제 환경에 올린 작업들입니다. 고객사 정보는 적지 않았고, 수치는 각 프로젝트의 평가셋 기준입니다.

## 공개 모델

### Veil-PII-Ko-Lite — 경량 한국어 개인정보 탐지 모델 `2026`

폐쇄망 CPU 환경에서 쓸 수 있는 한국어 PII 탐지 모델이 없어 직접 만들었습니다. KoELECTRA 백본에 BIES 태깅과 Viterbi 디코딩, 포맷 규칙 기반 합성 데이터를 더했습니다.

- 110M 파라미터 · INT8 143MB로 1.4B 비교 모델보다 넓은 **32개 라벨** 지원
- KDPII test F1 <span class="metric">0.934</span> (1.4B 비교 모델 0.453)
- [개발 리포트](/posts/veil-pii-ko-lite/) · [Hugging Face](https://huggingface.co/1T/veil-pii-ko-lite) · [Docker Hub](https://hub.docker.com/repository/docker/zzang9680/veil-pii/tags)

### 보험 도메인 검색 임베딩 (E5 · LoRA)

에이전트 안에서 매우 자주 호출되는 Tool·Agent 검색용 모델. CPU에서 가볍게 돌도록 118M · 384dim 모델을 골라 LoRA(0.44%)로 보험 도메인에 맞췄습니다.

- 도메인 평가셋(104 query) NDCG@10 <span class="metric">0.773</span> (base 0.630, +14.3pt) · R@1 <span class="metric">0.596</span> (+21.1pt)
- OpenAI text-embedding-3-large보다 도메인 NDCG가 높고(0.773 vs 0.634), 차원은 1/8
- [Hugging Face](https://huggingface.co/1T/wt-e5-small-ko-insurance-v1)

### 보험 약관 색인 임베딩 (KURE · LoRA)

긴 약관 문서를 페이지 단위로 색인하려고 8,192 토큰을 받는 KURE-v1을 보험 문서 42,626페이지로 튜닝했습니다. 256~7,500 토큰 멀티 레졸루션 커리큘럼과 hard-negative mining을 썼습니다.

- 긴 문서(7.5K 토큰) R@1 <span class="metric">77.7</span> — base 55.6, OpenAI 18.2 대비 +59.5pt
- 일반 한국어 성능(KorSTS) 변화 −0.07pt로 망각 없이 유지
- [Hugging Face](https://huggingface.co/1T/wt-kure-insurance-v1)

## 현장 프로젝트

### 금융 폐쇄망 · 보험 설계 멀티 에이전트 `2026 – 진행 중`

설계사의 가입 설계 업무를 돕는 멀티 에이전트. Supervisor가 6종 서브 에이전트를 ReAct 방식으로 호출하고 50종 API를 Tool로 다룹니다.

- 역할: Supervisor 라우팅 아키텍처 설계, Private VNet · Private Endpoint 인프라, Container Apps 오토스케일 운영, Durable Functions 인덱싱, 임베딩 Fine-Tuning
- 스택: LangGraph · FastAPI · Azure AI Search · Cosmos DB · PostgreSQL · Container Apps · Langfuse

### 금융 폐쇄망 · 보험 세일즈 어시스턴트 `2025`

복잡한 보험 질의를 Plan&Execute와 Multi-hop 검색으로 단계적으로 푸는 에이전트. 근거가 부족하면 질의 재작성 → 재계획 → 재실행을 반복하고, Rerank로 핵심 근거를 골라 답합니다.

- 약관 300GB를 Parent-Child로 청킹해 벡터 검색 구성, Durable Functions 실시간 인덱싱
- 스택: LangGraph · Azure AI Search · Container Apps (Private Endpoint) · Arize Phoenix

### FDE 딜리버리 에이전트 (Harness) `리드`

고객 요구사항(PRD)을 받아 도메인 에이전트를 해석 → 설계 → 구현 → 검증까지 자동화하는 내부 멀티 에이전트. 전체 구조 설계와 내부 에이전트 개발을 리드했고, 실제 보험 도메인 딜리버리에 쓰였습니다.

### ERP 업무비서 에이전트 `~ 2025.04`

1,000개 기업이 쓰는 B2B SaaS 업무비서. 자연어 요청을 의도 분류해 회계·세무·인사·그룹웨어 기능으로 라우팅하고, 사내 문서 RAG QA를 제공했습니다.

- 400개 Tool 중 하나를 고르는 2단계 Hybrid Tool Retrieval(임베딩 → LLM)로 실운영 정확도 <span class="metric">93%+</span>
- CPU에서 도는 0.4B RoBERTa 임베딩을 LoRA로 회계·세법 도메인에 튜닝
- 64대 운영 서버에 10% → 20% → 50% → 80% → 전체 순으로 점진 배포

### 그 밖의 ML

- 법인세 신고서식 예측 — FT-Transformer 기반 Tabular 모델, 정확도 87.3%
- 기업 거래 데이터 기반 계정과목 추천 · 기업 분류 모델
- 이상 경비 청구 탐지 — GloVe · BM25 앙상블, NLP + Clustering
