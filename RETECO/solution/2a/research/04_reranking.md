# Direction 4: LLM-based Reranking cho Conversational Retrieval

Reranking là bước sau first-stage retrieval: nhận top-k candidates (thường 100),
re-score bằng model nặng hơn. Trong conversational IR, reranker cần hiểu cả context
của conversation + document content.

---

## P1 — CLEAR: Entailment Distillation for Conversational Retrieval (EMNLP 2026 Findings)
**EMNLP 2026 Findings** | arXiv:2609.03482  
**Authors:** Qin, An, Guo, Ke, Wei, Yang, Zou

**Core idea:** Shift từ *topical relevance* sang *answerability*.  
Vấn đề: cross-encoder thường rank cao documents match topic của conversation, nhưng không nhất thiết chứa câu trả lời.

CLEAR framework:
1. **Entailment distillation:** Train cross-encoder reranker bằng tín hiệu "passage này có entail câu trả lời không?" thay vì chỉ topical relevance. Không cần actual answers ở inference.
2. **Passage-centric abductive recall:** Dùng LLM infer query từ passage → surface low-similarity but answerable passages bị bỏ sót ở first-stage.

**Results:** Consistently improves top-ranked precision vs strong query-rewriting + dense retrieval baselines. Gains lớn nhất ở conversations có nhiều topical noise.

**Relevance: HIGH** — RETECO có nhiều domains với technical jargon → topical noise cao. Answerability-focused reranker rất phù hợp. Tuy nhiên cần fine-tuning.

---

## P2 — Listwise Reranking với LLM (SemEval 2026)
**ACL 2026 (SemEval workshop)** | arXiv:2606.11945 (xem cả Direction 3)  
**Authors:** Lupart et al. (IRLab Amsterdam)

**Core idea:** Thay vì pointwise reranking (LLM score từng doc × N calls), dùng **listwise**: đưa cả top-k candidates cho LLM trong một prompt, LLM output ranked permutation.

Ưu điểm:
- 1 LLM call thay vì N calls (efficiency)
- LLM thấy được *relative quality* giữa các docs
- Có thể exploit long-context window của LLM hiện đại

**Relevance: HIGH** — Với SAP AI Core access, listwise reranking là implementable và sẽ significantly improve precision.

---

## P3 — Sifei@SemEval-2026: Cross-Encoder Reranking (SemEval 2026)
**ACL 2026** | arXiv:2606.28352

**Core idea:** Dùng cross-encoder (bi-encoder → cross-encoder là standard pipeline) sau hybrid retrieval. Cross-encoder thấy cả (query + conversation history) và document.

**Results:** Xếp hạng 3/38 trong SemEval-2026 Task 8. nDCG@5 = 0.5453.

**Relevance: HIGH** — Baseline implementation đơn giản nhất: `ms-marco-MiniLM` cross-encoder với concatenated [history + query + document].

---

## P4 — IRLab@iKAT24: Cross-encoder trong full pipeline (iKAT 2024)
**TREC iKAT 2024** | arXiv:2411.14739

**Cross-encoder setup:**
- Input: `[CLS] query_rewrite [SEP] document [SEP]`
- Hoặc: `[CLS] [history] query [SEP] document [SEP]` (conversation-aware)
- Trained trên MSMARCO hoặc fine-tuned trên CAsT/iKAT

**Key finding:** Multi-aspect query generation kết hợp với cross-encoder tốt hơn là chỉ dùng best single rewrite + cross-encoder. Reranker benefits từ multi-query diversity.

**Relevance: HIGH** — Kết hợp multi-query (Direction 1) + reranking là strongest combo.

---

## P5 — ChatR1: RL với Intent-Aware Reward (ACL 2026)
**ACL 2026 Main** | arXiv:2510.13312  
**Authors:** Lupart, Aliannejadi, Kanoulas

**Core idea:** Không phải traditional reranker mà là RL agent học *khi nào cần retrieve, khi nào generate*. Intent-aware reward cung cấp turn-level feedback.

Nhưng relevant với reranking: hệ thống learn quality signal từ retrieval outcomes → có thể dùng để generate training data cho reranker.

**Relevance: MEDIUM** — Quá phức tạp cho competition setting, nhưng idea về intent-aware reward cho reranking training là valuable.

---

## Practical Reranking Stack cho RETECO

### Option A: Lightweight (no training)
```
Top-100 từ hybrid retrieval
→ cross-encoder (ms-marco-MiniLM-L-6-v2) 
→ input: "[conversation_history] [SEP] [current_query] [SEP] [document]"
→ Top-10
```
Estimted gain: +3–8% nDCG@10

### Option B: LLM Listwise (training-free, costs API)
```
Top-20 từ hybrid retrieval
→ LLM listwise reranker (GPT-4o-mini qua SAP AI Core)
→ Prompt: "Given conversation: {history}\nQuery: {query}\nRank these passages: {P1..P20}"
→ Top-10
```
Estimated gain: +5–12% nDCG@10

### Option C: Answerability-focused (needs fine-tuning — CLEAR approach)
```
Top-100 → cross-encoder fine-tuned với entailment signal → Top-10
```
Strongest but requires training data with answer annotations.

---

## Tóm tắt

Reranking là stage tốt nhất để inject conversation awareness mà không thay đổi retriever:
- Cross-encoder thấy full context (history + query + document)
- LLM listwise là SOTA và feasible với SAP AI Core
- Answerability-based (CLEAR) là direction mới nhất và mạnh nhất cho domains có technical content
