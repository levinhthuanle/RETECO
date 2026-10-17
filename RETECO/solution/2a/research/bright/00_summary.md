# BRIGHT + RECOR: Research Summary

*Research cho Track 2a = RECOR benchmark (BRIGHT-based reasoning-intensive retrieval)*

---

## Bài toán thực sự

**RETECO Track 2a = RECOR** (Reasoning-focused Multi-turn Conversational Retrieval)
- 707 conversations, 2,971 turns, 11 domains
- 507,141 documents (Wikipedia + StackExchange-derived)
- Built on **BRIGHT benchmark** (6 of 11 domains)
- Metric: **macro-averaged nDCG@10**

BRIGHT là retrieval benchmark đặc biệt — queries yêu cầu multi-step reasoning để tìm relevant documents, **không phải keyword/semantic matching thông thường**. BM25 chỉ đạt 0.04 trên BRIGHT single-turn. RECOR mở rộng sang multi-turn conversational setting.

---

## SOTA Performance trên RECOR (Macro nDCG@10)

### Bảng chính — RECOR paper Table 3 (5 query strategies × 7 retrievers):

| Retriever | Baseline | QueryRewrite | Reasoning | History | Hist+Reas |
|-----------|----------|-------------|-----------|---------|-----------|
| **DIVER** | .347 | .430 | .496 | .545 | **.584** |
| **ReasonIR** | .266 | .357 | .494 | .496 | **.552** |
| Qwen-TE | .269 | .345 | .399 | .425 | .461 |
| SFR-Embed | .240 | .324 | .396 | .429 | .464 |
| BGE-M3 | .230 | .328 | .347 | .411 | .445 |
| E5-Instruct | .183 | .272 | .352 | .404 | .429 |
| BM25 | .185 | .288 | .360 | .446 | .489 |
| **Average** | .236 | .322 | .393 | .440 | .479 |

**Published SOTA: DIVER + History+Reasoning = 0.584**

### BRIGHT Leaderboard (single-turn, nDCG@10, short docs):

| System | Score | Method |
|--------|-------|--------|
| Mira-Reasoning-Retrieval | 66.9 | GRPO + multi-reward + 3-stage training |
| INF-X-Retriever | 63.4 | LLM-based embedding + reasoning |
| RakanEmbed4B | 52.4 | 4B LLM-based embedder |
| NeMo Agentic Retrieval | 50.9 | Agentic multi-step |
| DIVER-v3 | 46.8 | Reasoning-specialist retriever |
| BGE-Reasoner | 46.4 | BGE + reasoning augmentation |
| ReasonIR + reranker | 36.9 | 8B + Qwen reranker |
| BM25 + GPT-4 + Qwen rerank | 37.8 | Classical + LLM |
| BM25 alone | 14.5 | Baseline |

---

## Ceiling Analysis (từ RECOR paper Table 6)

- **Maximum per-query ceiling** (bất kỳ retriever nào + H+R): **~0.81 macro**
- 63.2% queries có thể đạt ≥0.8 với right approach
- DIVER + H+R đạt 0.584 → còn khoảng cách lớn tới ceiling
- Gap chủ yếu từ: (1) multi-turn reasoning chưa optimal, (2) domain gaps

---

## Per-Domain Performance (DIVER + H+R, Table 5)

| Domain | nDCG@10 |
|--------|---------|
| Psychology | .733 |
| Biology | .732 |
| Earth Science | .694 |
| Sustainable Living | .682 |
| Robotics | .587 |
| Politics | .582 |
| Economics | .572 |
| Law | .540 |
| Drones | .499 |
| Medical | .488 |
| Hardware | .455 |

Easy domains: Knowledge-heavy (Psychology, Biology). Hard: Technical instruction-following (Hardware, Medical).

---

## Tại sao bài toán khó

1. **BRIGHT = reasoning-intensive**: Không thể match keywords. Phải "hiểu" chuỗi reasoning cần thiết để đi từ query tới relevant doc.
2. **Multi-turn amplifies complexity**: Mỗi turn mới thêm context cần resolve + có thể shift topic.
3. **Domain diversity**: 11 domains rất khác nhau → no one-size-fits-all approach.
4. **Conversation = implicit reasoning**: User không nói rõ "tôi đang hỏi về X" — phải infer từ history.

---

## Recommended Strategy để đạt 0.7+

### Must-have (mọi submission):
1. **History + Reasoning query strategy** (H+R) — tăng tất cả retrievers ~2× baseline
2. **Reasoning-specialized retriever** — DIVER hoặc ReasonIR thay vì generic e5/BM25

### High-impact additions:
3. **Chain-of-thought reasoning** trong LLM query construction (CoT trước khi generate H+R query)
4. **LLM-generated hypothetical document** (HyDE-R) — tạo "ideal answer" trước khi embed
5. **Multi-query fusion** — generate 3-5 queries với different reasoning angles, RRF merge

### Advanced (requires more compute):
6. **DIVER-style fine-tuning** — train retriever với reasoning-rich contrastive data
7. **Mira-GRPO approach** — RL với multi-dimensional reward bao gồm retrieval signal

---

## File Reference

- `A_bright_specific.md` — BRIGHT benchmark design, why standard methods fail, BRIGHT-specific retrievers
- `B_instruction_llm.md` — LLM-based embedding: instruction-following, hypothetical docs, reasoning augmentation
- `C_recor_paper.md` — RECOR paper full analysis: all tables, query strategies, domain results
- `D_cot_reasoning.md` — CoT/RL-based reasoning for retrieval: Search-R1, Mira, ReasonIR training
- `E_embedding_models.md` — Embedding models on BRIGHT: rankings, architecture differences, how to pick
