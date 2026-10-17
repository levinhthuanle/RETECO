# RETECO Track 2a: Research Summary & Recommended Roadmap

*Tổng hợp từ research về SOTA conversational retrieval (2024–2026)*

---

## Bức tranh tổng thể SOTA

Hệ thống tốt nhất tại các shared tasks gần nhất (iKAT 2024, iKAT 2025, SemEval-2026) đều theo pipeline sau:

```
Conversation history + Current query
        ↓
[Stage 1] LLM Query Rewriting → Multi-query variants
        ↓
[Stage 2] Hybrid Retrieval (Sparse + Dense) → Top-100
        ↓
[Stage 3] Cross-encoder / LLM Reranking → Top-10
        ↓
nDCG@10 evaluation
```

---

## Điểm yếu của hệ thống hiện tại (RETECO)

| Component | Hiện tại | SOTA |
|-----------|----------|------|
| Query rewriting | 1 rewrite / turn (GPT-4o-mini, single-step) | Multi-query (beam search / compositional) |
| History handling | 2-step raw history → rewrite | CHIQ: clarify history *trước* rồi rewrite |
| Retrieval | BM25 *hoặc* dense (riêng lẻ) | Hybrid RRF (BM25 + dense) |
| Dense model | e5-base-v2 (generic, không fine-tune) | Fine-tuned với conversational supervision |
| Reranking | Không có | Cross-encoder hoặc LLM listwise |

---

## Recommended Next Steps (theo priority)

### Priority 1 — Quick wins, no training (1–2 ngày)
**Expected gain: +8–15% nDCG@10**

1. **Hybrid RRF** (Direction 3): Merge BM25 + dense scores bằng RRF@60  
   `rank_final = RRF(bm25_rank, e5_rank)`

2. **Multi-query rewriting** (Direction 1, P2): Dùng beam search (beam=4) trên rewriter  
   Generate 4 rewrites/turn → retrieve với cả 4 → merge bằng RRF  
   Hoặc CompCQR: atomic components → combinations (P6)

3. **Cross-encoder reranking** (Direction 4): Add reranking layer  
   `ms-marco-MiniLM-L-6-v2` với input = `[history][SEP][query][SEP][doc]`

4. **CHIQ history clarification** (Direction 1, P1): Thêm bước pre-rewrite  
   `clarify_history(history) → clean_history → rewrite(clean_history, query)`

### Priority 2 — Medium effort, high gain (1 tuần)
**Expected gain: additional +5–10%**

5. **LLM Listwise Reranking** (Direction 4, P2):  
   Top-20 docs → GPT-4o-mini via SAP AI Core → ranked list  
   Prompt: "Given conversation [ctx], rank these 20 passages by relevance"

6. **SPLADE thay BM25** (Direction 2, P4):  
   `naver/splade-v3` hay `prithivida/splade-v3` → learned sparse với vocab expansion  
   Giải quyết vocabulary mismatch trong technical domains

### Priority 3 — Requires fine-tuning (2+ tuần)
**Expected gain: additional +5–15%**

7. **ContextualRetriever** (Direction 2, P1):  
   Fine-tune e5-base-v2 với intent-guided supervision từ GPT-4 rewrites  
   Distill rewriting knowledge vào retriever → no rewrite at inference

8. **Domain-specific pseudo-labeling** (Direction 2, P3):  
   Generate pseudo-relevance labels cho mỗi domain bằng T5 → fine-tune per-domain

9. **CLEAR reranker** (Direction 4, P1):  
   Fine-tune cross-encoder với answerability signal → best cho technical domains

---

## Papers quan trọng nhất để đọc full

1. **CHIQ** (EMNLP 2024) — arXiv:2406.05013 → cải thiện rewriting ngay
2. **Multi-Query Rewriting** (SIGIR 2024) — arXiv:2406.18960 → implement ngay, no code change
3. **ContextualRetriever** (EMNLP 2025) — arXiv:2509.19700 → optionC (fine-tuned dense)
4. **CLEAR** (EMNLP 2026) — arXiv:2609.03482 → answerability-focused reranking
5. **CompCQR** (EMNLP 2026) — arXiv:2609.14646 → compositional multi-query, low LLM cost
6. **DiSCo** (SIGIR 2025) — arXiv:2410.14609 → SPLADE with conversational distillation
7. **Sifei@SemEval-2026** — arXiv:2606.28352 → system paper closest to our task

---

## Key groups/researchers to follow

- **IRLab Amsterdam** (Simon Lupart, Mohammad Aliannejadi, Evangelos Kanoulas): most prolific 2024-2026, iKAT + SemEval systems
- **Renmin University IR group** (Kelong Mao, Fengran Mo, Zhicheng Dou): conversational dense retrieval
- **Samsung Research Korea** (Seunghan Yang et al.): ContextualRetriever

---

## Baseline comparison (hiện tại vs literature)

| System | nDCG@10 |
|--------|---------|
| BM25 + history (chúng ta) | 0.3603 |
| e5-base-v2 dense (chúng ta, partial) | ~0.27 |
| **Hybrid BM25+dense (literature target)** | **~0.42–0.45** |
| **+ Reranking (literature target)** | **~0.48–0.55** |
| SemEval-2026 top-3 (related task) | ~0.54 nDCG@5 |

*Note: So sánh rough vì different datasets, nhưng magnitude gain từ hybrid+reranking là ~+15–20%*
