# Track 2a — Plan: Highest-Accuracy Approach

**Target:** Maximize macro nDCG@10 on RECOR (RETECO Track 2a)  
**Benchmark:** RECOR — 707 conversations, 2,971 turns, 11 domains, 507,141 docs  
**Evaluation:** Evaluation window Jan 10–31, 2027

---

## Baseline và Gap Analysis

| System | Macro nDCG@10 | Source |
|--------|--------------|--------|
| BM25 (current turn only) | 0.1827 | Official RETECO baseline (dev) |
| BM25 + History (Lucene) | **0.4379** | Official RETECO baseline (dev) |
| Our optionA: BM25 + QR | ~0.3603 | Measured on dev |
| **SOTA: DIVER + H+R** | **0.584** | RECOR paper Table 3 |
| Theoretical ceiling (oracle) | ~0.81 | RECOR paper Table 6 |

Gap từ SOTA tới ceiling còn lớn (~0.23) → có room để vượt SOTA bằng ensemble + reranking + fine-tuning.

---

## Phase 1 — Quick Win: Upgrade Query Strategy (H+R)

**Estimated score: ~0.48–0.49 | Effort: 1–2 giờ | Risk: Thấp**

### Thay đổi:
Thay prompt trong `optionA/rewriter.py` từ QueryRewrite → History+Reasoning (H+R).

H+R strategy (từ RECOR paper):
```
Given conversation history: {full_history}
Current question: {current_turn}

Step 1: Reason step by step about what specific information is needed to answer 
the current question, given the conversation context.
Step 2: Formulate a standalone search query that captures this information need,
incorporating the key concepts from your reasoning and the conversation history.

Format:
Reasoning: <step-by-step reasoning about what information is needed>
Search query: <the search query>
```

### Tại sao H+R là biggest gain:
- Từ RECOR Table 3: BM25 + H+R = 0.489 vs BM25 + QR = 0.288 (official) — +70%
- Query rewriting chỉ giải quyết coreference. Reasoning chain bridges lexical gap giữa conversational query và document.
- BRIGHT domains đặc biệt cần reasoning: docs không chứa keywords từ query, mà chứa kiến thức cần thiết để reason tới câu trả lời.

### Implementation notes:
- Giữ nguyên file-based cache per qid (quan trọng để không gọi LLM lại)
- Turn 1: no history → dùng Reasoning-only prompt (không có H phần)
- Turn 2+: dùng full H+R prompt với full conversation history
- Extract chỉ phần "Search query:" từ response để dùng làm BM25 query

---

## Phase 2 — Retriever Upgrade: ReasonIR-8B

**Estimated score: ~0.55 | Effort: 2–3 ngày | Risk: Medium (memory, thời gian encode)**

### Thay đổi:
Thay `intfloat/e5-base-v2` bằng `facebook/ReasonIR-8B` trong `optionB/embedder.py`.

### Tại sao ReasonIR:
- ReasonIR + H+R = 0.552 trong RECOR paper (gần SOTA)
- Training data: 100K+ synthetic (query, reasoning_chain, doc) triplets từ GPT-4 → embedding space aligned với reasoning paths
- E5-base-v2 (109M, general) → BRIGHT score ~4. ReasonIR (8B, reasoning-specialist) → BRIGHT 29.9

### Requirements:
- ~20GB VRAM fp16 (A100 40GB hoặc 2×RTX 3090)
- Nếu không có GPU đủ: quantize với bitsandbytes (4-bit: ~6GB) hoặc dùng GGUF

### Query format cho ReasonIR:
```python
# Document side (unchanged)
"passage: {document_text}"

# Query side (H+R)
"query: {current_turn}\nLet me think step by step: {llm_generated_reasoning}"
```

---

## Phase 3 — DIVER: Replicate SOTA

**Estimated score: ~0.58 | Effort: 3–5 ngày | Risk: Medium (tìm model weights)**

### Thay đổi:
Tìm và load DIVER model thay ReasonIR.

### Tại sao DIVER:
- DIVER + H+R = 0.584 (published SOTA trên RECOR)
- DIVER generates diverse reasoning paths → ensemble embedding → captures multiple reasoning perspectives
- Đặc biệt phù hợp với conversational setting vì mỗi conversation turn có thể cần different reasoning angle

### Action items:
1. Tìm DIVER model weights (Long et al., 2025 — arXiv 2502.20134 hoặc tương tự)
2. Check GitHub/HuggingFace cho `DIVER` hoặc `diver-retrieval`
3. Nếu không có official weights: có thể dùng DIVER approach với ReasonIR (generate multiple reasoning paths, embed average)

---

## Phase 4 — Hybrid + Reranking

**Estimated score: ~0.60–0.62 | Effort: 1 tuần | Risk: Medium**

### Components:

#### 4a. BM25 + Dense Hybrid (RRF)
```python
# Reciprocal Rank Fusion
rrf_score(doc) = 1/(k + rank_bm25) + 1/(k + rank_dense)  # k=60
```
- BM25 + H+R tốt cho exact-match aspects của query
- Dense (DIVER/ReasonIR) tốt cho semantic/reasoning aspects
- RRF combination thường +3–5% so với single retriever

#### 4b. LLM Reranker
```python
# Reasoning-augmented reranking prompt
prompt = """
Conversation history: {history}
Current question: {question}
Reasoning chain: {reasoning}

Given the above context, rate the relevance of the following passage 
on a scale of 0-10:
Passage: {doc_text}
Score: 
"""
```
- Top-100 candidates → rerank với LLM (GPT-4o-mini) → take top-10
- Cost: ~100 LLM calls per query → cần budget planning

#### 4c. Cross-encoder Reranker (compute-efficient alternative)
- Model: `BAAI/bge-reranker-v2-m3` hoặc `cross-encoder/ms-marco-MiniLM-L-12-v2`
- Cheaper than LLM reranking, still +3–7% on top of bi-encoder

---

## Phase 5 — Advanced: Multi-Query Fusion

**Estimated score: ~0.63–0.65 | Effort: 1–2 tuần | Risk: Medium-High**

### Approach: Generate 3–5 diverse queries, RRF merge

```python
queries = llm_call(f"""
Generate 3 different search queries for this information need, 
each emphasizing different aspects:

Conversation: {history}
Question: {current_turn}
Reasoning: {reasoning_chain}

Query 1 (literal/factual angle): 
Query 2 (conceptual/mechanistic angle):
Query 3 (application/example angle):
""")

# Retrieve with each query, RRF merge results
results = rrf([retrieve(q) for q in queries])
```

### Tại sao:
- RECOR Table 6: ceiling ~0.81, SOTA 0.584 → gap lớn
- Nhiều query → higher recall → reranker có nhiều candidates tốt hơn
- Different reasoning angles cover different aspects của information need

---

## Phase 6 — Fine-tuning (High-reward, Long-term)

**Estimated score: 0.65+ | Effort: Weeks | Risk: High (compute, data)**

### Option A: ConvSearch-R1 style (RL on query reformulation)
- Self-distillation: generate queries → retrieve → dùng retrieval signal để filter
- RL phase: reward = nDCG improvement
- Applicable trực tiếp cho RECOR (multi-turn conversational)
- Paper: arXiv 2505.15776

### Option B: Mira-GRPO style (RL with multi-reward)
- Phase 1: Mine best query rewrites bằng multiple LLMs
- Phase 2: Cold start training trên mined rewrites
- Phase 3: GRPO với multi-dimensional reward (retrieval + faithfulness + coverage + conciseness)
- Potential score: 0.65–0.70+ (extrapolating from Mira's 66.9 on BRIGHT single-turn)

### Option C: Domain-specific contrastive fine-tuning
- Hard negatives mining: per-domain false positives (docs từ same domain, same topic nhưng sai)
- Contrastive training trên RETECO train split
- Easier than full RL, still meaningful gain

---

## Domain-Specific Strategy

Từ RECOR Table 5 (DIVER + H+R per domain):

| Domain | Score | Strategy note |
|--------|-------|--------------|
| Psychology | .733 | BRIGHT — strong, H+R đủ |
| Biology | .732 | BRIGHT — strong, H+R đủ |
| Earth Science | .694 | BRIGHT — strong |
| Sustainable Living | .682 | BRIGHT — strong |
| Robotics | .587 | BRIGHT — medium |
| Politics | .582 | StackExchange |
| Economics | .572 | BRIGHT — medium |
| Law | .540 | StackExchange — specialized |
| Drones | .499 | StackExchange — technical, hard |
| Medical | .488 | StackExchange — specialized, hard |
| Hardware | .455 | StackExchange — hardest |

**Low-hanging fruit:** BRIGHT domains (Biology, Psychology, Earth Science) đã cao. Focus cải thiện StackExchange domains (Hardware, Medical, Drones) sẽ tăng macro nhiều hơn.

**Domain-specific prompt tuning:** Với technical StackExchange domains, thêm domain hint vào H+R prompt:
```python
domain_hints = {
    "hardware": "Focus on technical specifications, component compatibility, and hardware troubleshooting procedures.",
    "medical": "Focus on medical terminology, clinical procedures, drug mechanisms, and diagnostic criteria.",
    "drones": "Focus on drone regulations, flight mechanics, component specifications, and troubleshooting."
}
```

---

## Execution Roadmap

```
Phase 1: H+R Query Strategy          → ~0.48   [Tuần 1, optionA]
Phase 2: ReasonIR-8B                  → ~0.55   [Tuần 2-3, optionB]
Phase 3: DIVER (if weights found)     → ~0.58   [Tuần 3-4]
Phase 4a: Hybrid BM25 + Dense (RRF)   → ~0.60   [Tuần 4]
Phase 4b: Cross-encoder Reranking     → ~0.62   [Tuần 5]
Phase 5: Multi-query Fusion           → ~0.65   [Tuần 6-7]
Phase 6: Fine-tuning (optional)       → 0.68+   [Tháng 2-3]
```

**Minimum viable submission (nếu chỉ có 1 tuần):** Phase 1 + Phase 2 → ~0.55, vượt BM25+History official baseline (0.4379) đáng kể.

**Target cho competitive submission:** Phase 1 + 2 + 4 → ~0.60–0.62, gap nhỏ so với published SOTA 0.584.

---

## Files cần chỉnh sửa

| File | Phase | Thay đổi |
|------|-------|---------|
| `optionA/rewriter.py` | 1 | Thay prompt QR → H+R; extract "Search query:" từ response |
| `optionA/run_2a.py` | 1 | Pass full conversation history vào rewriter |
| `optionB/embedder.py` | 2 | Thay model e5-base-v2 → ReasonIR-8B |
| `optionB/run_2a.py` | 2 | Generate H+R query với LLM trước khi embed |
| `optionB/reranker.py` | 4 | New file: cross-encoder reranking trên top-100 |
| `optionB/hybrid.py` | 4 | New file: RRF merge BM25 scores + dense scores |

---

## Key References

- **RECOR paper:** ACL 2026 Findings, https://aclanthology.org/2026.findings-acl.129/
- **ReasonIR:** arXiv:2504.20595, model: `facebook/ReasonIR-8B`
- **DIVER:** arXiv:2502.20134 (approximate)
- **ConvSearch-R1:** arXiv:2505.15776
- **BRIGHT benchmark:** arXiv:2407.12883
- **Mira-Reasoning-Retrieval:** BRIGHT leaderboard #1, Forward AI Labs

---

*Plan dựa trên RECOR paper Table 3 numbers và BRIGHT leaderboard data. Estimated scores là extrapolations, không phải guarantees.*
