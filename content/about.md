---
title: "소개"
description: "기술보다, 해결할 문제의 본질을 정확히 보려고 노력합니다."
showDate: false
showReadingTime: false
showAuthor: false
showHero: false
showTableOfContents: false
---

고객의 문제를 직접 들여다보고, 그 상황에 맞는 에이전트와 모델을 만들어 현장에 배포합니다.
아래는 지금까지 만들어 온 것들입니다.

{{< timeline >}}

{{< timelineItem icon="star" header="사내 AX 경진대회 대상" badge="2026.09" subheader="Knowledge Layer 표준화 플랫폼" md="true" >}}
기업 폐쇄망에 흩어진 레거시 리소스(API · DB · 비정형 문서)를 에이전트가 이해할 수 있는 **Knowledge Layer**로 통합하고 관리하는 플랫폼.
{{< /timelineItem >}}

{{< timelineItem icon="shield" header="Veil-PII-Ko-Lite 공개" badge="2026.09" subheader="경량 한국어 개인정보 탐지 모델" md="true" >}}
폐쇄망 CPU 환경용 110M 모델. 1.4B 비교 모델보다 넓은 32개 라벨을 KDPII F1 **0.934**로 탐지.
[개발 리포트](/posts/veil-pii-ko-lite/) · [Hugging Face](https://huggingface.co/1T/veil-pii-ko-lite)
{{< /timelineItem >}}

{{< timelineItem icon="wand-magic-sparkles" header="보험 설계 멀티 에이전트" badge="2026.03 – 진행 중" subheader="금융 폐쇄망 · Azure Private Endpoint" md="true" >}}
Supervisor가 6종 서브 에이전트를 ReAct로 호출하고 50종 API를 Tool로 다루는 구조를 설계. Private VNet 인프라, Container Apps 오토스케일, Durable Functions 인덱싱까지 담당.
{{< /timelineItem >}}

{{< timelineItem icon="search" header="보험 도메인 임베딩 Fine-Tuning" badge="2026" subheader="E5 (검색) · KURE (색인) · LoRA" md="true" >}}
Tool·Agent 검색용 E5는 도메인 NDCG@10 **0.773** (+14.3pt), 약관 색인용 KURE는 7.5K 토큰 긴 문서 R@1 **77.7** (OpenAI 대비 +59.5pt).
[E5 모델](https://huggingface.co/1T/wt-e5-small-ko-insurance-v1) · [KURE 모델](https://huggingface.co/1T/wt-kure-insurance-v1)
{{< /timelineItem >}}

{{< timelineItem icon="list-check" header="FDE 딜리버리 에이전트 (Harness)" badge="2025 –" subheader="내부 멀티 에이전트 · 리드" md="true" >}}
고객 요구사항(PRD)을 받아 도메인 에이전트를 해석 → 설계 → 구현 → 검증까지 자동화하는 Harness 전체 구조를 설계하고 개발을 리드.
{{< /timelineItem >}}

{{< timelineItem icon="comment" header="보험 세일즈 어시스턴트" badge="2025.06 – 2025.12" subheader="Plan & Execute · Multi-hop 검색" md="true" >}}
근거가 부족하면 질의 재작성 → 재계획 → 재실행을 반복하는 Multi-hop 루프. 약관 300GB를 Parent-Child로 청킹하고 Durable Functions로 실시간 인덱싱.
{{< /timelineItem >}}

{{< timelineItem icon="code" header="ERP 업무비서 에이전트" badge="2023.10 – 2025.04" subheader="1,000개 기업이 쓰는 B2B SaaS" md="true" >}}
400개 Tool 중 하나를 고르는 2단계 Hybrid Tool Retrieval로 실운영 정확도 **93%+**. CPU에서 도는 0.4B RoBERTa 임베딩을 LoRA · SimCSE로 도메인 튜닝하고, 64대 서버에 10% → 전체로 점진 배포.
{{< /timelineItem >}}

{{< timelineItem icon="lightbulb" header="ML 모델 개발" badge="이전 작업" subheader="Tabular · NLP" md="true" >}}
법인세 신고서식 예측(FT-Transformer, 정확도 87.3%), 거래 데이터 기반 계정 추천 · 기업 분류 모델, 이상 경비 청구 탐지(GloVe · BM25 앙상블).
{{< /timelineItem >}}

{{< /timeline >}}

## Stack

- **Agent** LangGraph · Multi-Agent · ReAct · Plan&Execute · RAG
- **Retrieval** E5 · KURE · RoBERTa · LoRA · Azure AI Search · FAISS · Re-ranking
- **Backend** Python · FastAPI · PostgreSQL · Cosmos DB · Redis · PyTorch
- **Cloud** Azure Container Apps · Durable Functions · Key Vault · Docker · GitHub Actions · Langfuse

## Contact

[zzang891014@gmail.com](mailto:zzang891014@gmail.com) · [GitHub](https://github.com/WontaeKim89) · [Hugging Face](https://huggingface.co/1T)
