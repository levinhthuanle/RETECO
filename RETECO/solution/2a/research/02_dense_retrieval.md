# Direction 2: Dense Retrieval cho Conversational Search

Hướng này tập trung vào việc *train hoặc adapt bi-encoder* để hiểu conversation context
thay vì chỉ dùng rewritten query. Khác với direction 1 (fix query rồi dùng generic retriever),
hướng này fix retriever để hiểu hội thoại trực tiếp.

---

## P1 — ContextualRetriever (EMNLP 2025)
**EMNLP 2025** | arXiv:2509.19700  
**Authors:** Yang, Lee, Bang, Shim, Kim, Chang (Samsung Research)

**Core idea:** Fine-tune LLM-based bi-encoder với 2 thành phần:
1. **Context-aware embedding:** Nhấn mạnh current query trong dialogue history — không chỉ concatenate mà học cách *weight* token importance theo vị trí trong conversation.
2. **Intent-guided supervision:** Training signal lấy từ high-quality rewritten queries (như GPT-4 rewrites) → model học biểu diễn tương đương rewrite nhưng không cần rewrite ở inference.

**Key strength:** *Zero inference overhead* — không cần gọi LLM khi deploy.

**Results:** Outperforms existing methods across multiple conversational search benchmarks.

**Relevance: HIGH** — Đây là hướng lý tưởng cho RETECO: train một lần, dùng không cần LLM call tại inference. Nhưng cần fine-tuning data (có thể dùng RETECO train set + GPT rewrites làm supervision).

---

## P2 — QRACDR: Aligning Query Representation (CIKM 2024)
**CIKM 2024** | arXiv:2407.20189  
**Authors:** Mo, Qu, Mao, Wu, Su, Huang, Nie

**Core idea:** Train bi-encoder bằng cách align query representation với CẢ HAI:
1. Rewritten query embedding (consistency với rewrite)
2. Relevant document embedding (consistency với relevance judgments)

Triple alignment: `encode(conv_query) ≈ encode(rewritten_query) ≈ encode(relevant_doc)`

**Results:** SOTA trên 8 datasets, cả conversational và ad-hoc search.

**Relevance: HIGH** — Approach alignment này không yêu cầu rewrite ở inference. Distill rewriting knowledge vào retriever weights.

---

## P3 — Domain Adaptation for Conversational Dense Retrieval (COLING 2024)
**COLING 2024** | arXiv:2403.08970  
**Authors:** Li, Gaussier

**Core idea:** Adapt dense retriever cho domain mới *không cần manual annotation*:
1. Dùng T5-3B để generate pseudo-positive labels từ documents
2. Mining hard negatives kỹ lưỡng trong target domain
3. Fine-tune bi-encoder + query rewriter trên pseudo-labeled data

**Key relevance:** RETECO có 11 domains, nhiều domain đặc thù (drones, robotics, medicalsciences). Domain-specific adaptation rất phù hợp.

**Results:** Outperforms baseline adaptation methods trên dense và conversational dense retrieval.

**Relevance: HIGH** — RETECO là multi-domain task, domain gap là vấn đề thực. Có thể generate pseudo-data cho mỗi domain.

---

## P4 — DiSCo: LLM Knowledge Distillation cho Sparse Retrieval (SIGIR 2025)
**SIGIR 2025** | arXiv:2410.14609  
**Authors:** Lupart, Aliannejadi, Kanoulas (IRLab Amsterdam)

**Core idea:** Distill LLM (teacher) vào SPLADE (student) cho conversational search:
- Không distill embeddings, mà distill *similarity scores* giữa conversation và document
- Multi-teacher strategy: nhiều LLMs → ensemble → distill vào 1 student model
- KLD loss + contrastive loss

**Results:** Up to 6-point recall improvement trên out-of-domain datasets. Tested trên 5 conversational search datasets.

**Relevance: MEDIUM-HIGH** — SPLADE + conversation context là hybrid approach tốt. 6-point recall gain quan trọng. Nhóm IRLab Amsterdam (Simon Lupart, Aliannejadi) có nhiều papers về conversational IR 2024-2026, đáng follow.

---

## P5 — IRLab@iKAT24: SPLADE + Multi-Aspect Queries (iKAT 2024)
**TREC iKAT 2024** | arXiv:2411.14739  
**Authors:** Lupart, Abbasiantaeb, Aliannejadi

**Core idea:** System bài thi iKAT 2024 (personalized conversational search) dùng:
1. MQ4CS để sinh multi-aspect rewrites
2. SPLADE cho learned sparse retrieval
3. Cross-encoder reranking

Alternative aggregation: thay vì interleave multi-aspect ngay ở retrieval, aggregate ở reranking stage.

**Results:** Outperforms human rewrite performance khi kết hợp LLM multi-aspect + SPLADE + reranker.

**Relevance: HIGH** — Full pipeline tốt cho reference implementation.

---

## P6 — CONVINV: Interpreting Conversational Dense Retrieval (ACL 2024)
**ACL 2024** | arXiv:2402.12774  
**Authors:** Cheng, Mao, Dou (Renmin University)

**Core idea:** Giải thích *tại sao* conversational dense retriever retrieve tài liệu đó bằng cách "invert" session embedding → text. Không phải retrieval method mới mà là interpretability tool. Tích hợp query rewrites vào inversion process để transparent hơn.

**Relevance: LOW** — Không trực tiếp cải thiện performance, nhưng hữu ích để debug retriever.

---

## Tóm tắt xu hướng

**2024–2025**: Trọng tâm là *distillation từ rewrite supervision vào retriever*:
- Train retriever để behave như đã rewrite query (không cần rewrite lúc inference)
- QRACDR, ContextualRetriever là examples tốt

**Emerging (2025–2026)**: 
- Domain adaptation không cần annotation (Li & Gaussier)
- LLM-scored distillation thay vì just representation alignment (DiSCo)

**Key gap với optionB của chúng ta:** e5-base-v2 không được fine-tune trên conversational data → bỏ sót conversation context. Nên fine-tune trên RETECO train set với intent-guided supervision (như ContextualRetriever).
