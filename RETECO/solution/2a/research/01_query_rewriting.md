# Direction 1: LLM-based Query Rewriting / Decontextualization

Hướng lâu đời nhất và vẫn là state-of-practice. Ý tưởng cốt lõi: chuyển câu hỏi
ngắn/mơ hồ phụ thuộc context thành query độc lập trước khi đưa vào retriever.

---

## P1 — CHIQ: Contextual History Enhancement for Improving Query Rewriting
**EMNLP 2024** | arXiv:2406.05013  
**Authors:** Mo, Ghaddar, Mao, Rezagholizadeh, Chen, Liu, Nie

**Core idea:** Hai bước:
1. Dùng LLM để *disambiguate conversation history* trước (làm rõ các tham chiếu mơ hồ, pronoun, ellipsis trong lịch sử).
2. Sau đó mới rewrite query dựa trên history đã được làm rõ.

Khác với các approach trước là rewrite thẳng từ raw history. Dùng open-source LLMs (không cần GPT-4).

**Results:** SOTA trên 5 benchmarks conversational search. Competitive với closed-source LLM systems.

**Relevance: HIGH** — Pipeline 2-bước này phù hợp với RETECO vì conversation history trong dataset dài và có nhiều coreference. Thay vì chỉ rewrite query, nên clean history trước.

---

## P2 — Multi-Query Rewriting via Beam Search (SIGIR 2024)
**SIGIR 2024** | arXiv:2406.18960  
**Authors:** Kostric, Balog

**Core idea:** Thay vì chỉ lấy top-1 rewrite từ beam search, lấy *tất cả beam* (multi-query). Mỗi beam là một cách paraphrase khác nhau. Khi retrieve, merge kết quả từ nhiều queries (RRF hoặc score aggregation).

Không cần training thêm, không tốn thêm LLM calls — chỉ tận dụng beam search sẵn có.

**Results:** SOTA trên conversational passage retrieval, cả sparse và dense.

**Relevance: HIGH** — Rất dễ áp dụng lên optionA/B. Beam search trên query rewriter → multiple rewrites → RRF.

---

## P3 — MQ4CS: Generating Multi-Aspect Queries for Conversational Search
**arXiv 2403.19302** (revised Oct 2024) — dùng tại iKAT 2024  
**Authors:** Abbasiantaeb, Lupart, Aliannejadi

**Core idea:** Dùng LLM để sinh nhiều rewrites *từ các góc độ khác nhau* của câu hỏi (multi-aspect). Ví dụ cùng 1 turn có thể có aspect "tên chủ thể", "thuộc tính", "so sánh"... Mỗi aspect → 1 rewritten query.

**Results:** 85% trường hợp nDCG@3 tốt hơn với nhiều hơn 1 rewrite. Outperforms SOTA query rewriting.

**Relevance: HIGH** — Kết hợp tốt với SPLADE hay BM25 (nhiều queries → nhiều keywords).

---

## P4 — RetPO: Ask Optimal Questions (NAACL 2025)
**NAACL 2025** | arXiv:2402.11827  
**Authors:** Yoon, Kim, Jeon, Kim, Jo, Kang

**Core idea:** Fine-tune LLM rewriter dựa trên *feedback từ retriever*. Sinh nhiều rewrites → rank theo retrieval performance → dùng DPO để train model nhỏ hơn prefer rewrite tốt.  
Dataset: RF collection với 410K query rewrites từ 12K conversations.

**Results:** Outperforms SOTA rewrite-then-retrieve approaches trên 2 benchmarks.

**Relevance: MEDIUM** — Cần fine-tuning, nhưng idea là train rewriter phù hợp với *specific retriever* rất đáng tham khảo cho RETECO. Retriever-aware rewriting.

---

## P5 — ConvSearch-R1: RL for Query Reformulation (EMNLP 2025)
**EMNLP 2025** | arXiv:2505.15776  
**Authors:** Zhu, Wang, Feng, Song, Qiu

**Core idea:** Dùng Reinforcement Learning để train query reformulator, loại bỏ hoàn toàn human rewrite supervision. 2 giai đoạn:
1. Warm-up: retrieval-guided self-distillation
2. RL với reward từ retrieval signal

Model 3B parameters, không cần GPT-4.

**Results:** >10% improvement trên TopiOCQA (rất challenging benchmark).

**Relevance: MEDIUM** — Approach rất clean (no human supervision). Nhưng cần fine-tune model, không phải zero-shot. Phù hợp nếu có resource để train.

---

## P6 — CompCQR: Compositional Query Generation (EMNLP 2026 Findings)
**EMNLP 2026 Findings** | arXiv:2609.14646  
**Authors:** Jang, Lee, Shin, Jung

**Core idea:** Thay vì gọi LLM nhiều lần (tốn kém), sinh ra *atomic components* (noun phrases, predicates, constraints) rồi *compositionally combine* chúng để tạo rất nhiều query variants. Chi phí LLM rất thấp.

Phát hiện: retrievers nhạy cảm với *content ordering* trong query → cần nhiều variants.

**Results:** 22.5% relative MRR improvement over SOTA, ít LLM calls hơn nhiều.

**Relevance: HIGH** — Training-free, low LLM cost. Rất phù hợp cho production system và SemEval deadline.

---

## Tóm tắt xu hướng

| Trend | 2024 | 2025 | 2026 |
|-------|------|------|------|
| Single rewrite | baseline | giảm dần | ít dùng |
| Multi-query rewriting | SIGIR'24 SOTA | standard | extended |
| RL-based rewriting | emerging | EMNLP'25 | mainstream |
| Retriever-aware rewriting | NAACL'25 | growing | |
| Compositional/low-cost | | | EMNLP'26 |

**Key takeaway:** Xu hướng chuyển từ "1 LLM call → 1 rewrite" sang "multi-query" hoặc "RL-optimized rewriting". Với RETECO, nên thử multi-query (P2/P3/P6) trước vì không cần training.
