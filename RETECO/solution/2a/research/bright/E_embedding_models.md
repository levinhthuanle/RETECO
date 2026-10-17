# E: Embedding Models on BRIGHT — Rankings & Architecture

## Overview: Why Models Differ So Much on BRIGHT

Generic models: BRIGHT = 0.04-0.18 nDCG@10
Reasoning-specialized: BRIGHT = 30-67 nDCG@10

The gap is architectural + training data, not just scale.

---

## BRIGHT Leaderboard (nDCG@10, short docs, as of mid-2025)

| Rank | Model | Score | Org | Approach |
|------|-------|-------|-----|----------|
| 1 | Mira-Reasoning-Retrieval | 66.9 | Forward AI Labs | GRPO RL + multi-reward |
| 2 | INF-X-Retriever | 63.4 | Infini-AI | LLM-based + reasoning |
| 3 | RakanEmbed4B | 52.4 | Unknown | 4B embedder |
| 4 | NeMo Agentic Retrieval | 50.9 | NVIDIA | Multi-step agentic |
| 5 | DIVER-v3 | 46.8 | — | Diverse reasoning paths |
| 6 | BGE-Reasoner | 46.4 | BAAI | BGE + reasoning aug. |
| 7 | Voyage-3-reasoning | ~42 | Voyage AI | Reasoning-specific |
| 8 | Qwen3-Embedding-8B | ~40 | Alibaba | Large LLM embedder |
| 9 | ReasonIR + reranker | 36.9 | Meta AI | 8B + Qwen reranker |
| 10 | BM25 + GPT-4 + Qwen rerank | 37.8 | — | Hybrid + LLM |
| — | E5-Mistral-7B | ~11 | Microsoft | Instruct embedding |
| — | BGE-M3 | ~8 | BAAI | Multilingual dense |
| — | BM25 | 14.5 | — | Sparse baseline |

*Scores approximate from multiple sources; exact numbers may vary by BRIGHT subset*

---

## Model-by-Model Analysis

### DIVER (Long et al., 2025)
**Ranking:** #5 single-turn BRIGHT (46.8), **#1 on RECOR** (.584 macro)

**Architecture:**
- Base model: LLM-based encoder (likely 7B class)
- Training: Diverse reasoning contrastive data
- Key: Generates multiple diverse reasoning paths → ensemble embedding

**How it embeds:**
```
Input: query + diverse_reasoning_paths (concatenated)
Output: single embedding capturing multiple reasoning perspectives
```

**Why top on RECOR specifically:**
RECOR has conversational context (history + reasoning) → DIVER's diversity in reasoning paths aligns well with different conversation angles.

**Availability:** Model weights available; check HuggingFace for `diver` or Long et al. GitHub

---

### ReasonIR-8B (Meta AI, 2025)
**Ranking:** #9 single-turn BRIGHT (36.9 with reranker), #2 on RECOR (.552 H+R)

**Architecture:**
- Base: Llama-3-8B fine-tuned as bi-encoder
- Training: 100K+ synthetic reasoning triplets from GPT-4
- Query side: `[question] + [reasoning_chain]` → CLS embedding
- Document side: `[document]` → CLS embedding

**Memory requirements:**
- ~20GB VRAM (fp16)
- Can run on A100 40GB or 2×RTX 3090

**HuggingFace:** `facebook/ReasonIR-8B`

**When to use:** If you can load 8B model, this is the best available open-source reasoning retriever.

---

### Qwen3-Embedding (Alibaba, 2025)
**BRIGHT:** ~35-45 range  
**RECOR H+R:** .461 macro

**Architecture:**
- Qwen3 LLM family → bi-encoder fine-tuning
- Available sizes: 0.6B, 1.8B, 4B, 8B
- Multilingual, long-context (32K tokens)

**Key advantage for RECOR:**
- Long conversation history fits in one pass (32K context)
- Strong on multilingual content (some RECOR StackExchange may have non-English references)
- MTEB top-5 across retrieval tasks

**HuggingFace:** `Qwen/Qwen3-Embedding-8B`, `Qwen/Qwen3-Embedding-4B`

**When to use:** If ReasonIR not available, Qwen3-Embedding-8B is strong alternative. Also if conversation history is very long.

---

### BGE-M3 (BAAI, 2024)
**BRIGHT:** ~8 nDCG@10  
**RECOR H+R:** .445 macro

**Architecture:**
- Multi-lingual, multi-granularity
- Dense + sparse + colbert (3 modes)
- Training: massive multilingual corpus

**Key limitation on BRIGHT:**
Despite impressive general retrieval, BGE-M3 is NOT reasoning-specialized → fails on BRIGHT's implicit reasoning requirement.

**HuggingFace:** `BAAI/bge-m3`

**When to use:** Good general baseline. If switching to reasoning model not feasible, use BGE-M3 but pair with strong H+R query strategy.

---

### E5-Mistral-7B / E5-Instruct (Microsoft, 2024)
**BRIGHT:** ~11 nDCG@10  
**RECOR H+R:** .429 macro

**Architecture:**
- Mistral-7B fine-tuned on 93 embedding tasks with instruction
- Instruction prepended: "Instruct: {task_description}\nQuery: {text}"

**RECOR instruction to use:**
```
"Instruct: Given a multi-turn conversation in a technical domain, retrieve relevant documents that help answer the current question based on the conversation history and reasoning provided.\nQuery: {h+r_query}"
```

**HuggingFace:** `intfloat/e5-mistral-7b-instruct`

---

### SFR-Embedding-Mistral (Salesforce, 2024)
**BRIGHT:** Competitive  
**RECOR H+R:** .464 macro

**Architecture:** Extends E5-Mistral with additional task-specific fine-tuning including:
- Reasoning benchmarks
- Domain adaptation data

Slightly better than E5-Instruct on RECOR.

**HuggingFace:** `Salesforce/SFR-Embedding-Mistral`

---

## Architecture Comparison: Why Some Models Reason Better

| Feature | E5-base-v2 | E5-Instruct | Qwen-TE | ReasonIR | DIVER |
|---------|-----------|------------|---------|---------|-------|
| Model size | 109M | 7B | 8B | 8B | ~7B |
| Instruction-following | No | Yes | Yes | Yes | Yes |
| Reasoning training data | No | Partial | Partial | Yes (synthetic) | Yes (diverse) |
| Reasoning chain input | No | No | Limited | Yes (design) | Yes (diverse) |
| BRIGHT score | ~4 | ~11 | ~35-40 | 29.9 / 36.9 | 46.8 |

**Key architectural insight:** 
- Scale alone: E5-Instruct 7B >> E5-base-v2 109M, but BRIGHT gain minimal (4 → 11)
- Reasoning training data: the gap from ~11 to ~47 is almost entirely due to specialized reasoning training
- Size + reasoning: ReasonIR-8B achieves 29.9 without reranker

---

## Rerankers: Impact on BRIGHT

Standard cross-encoders have same issue as bi-encoders on BRIGHT — they don't reason.

But rerankers have access to full (query, document) pair text → more powerful inference possible.

### Qwen reranker (used by ReasonIR team):
- ReasonIR alone: 29.9 → with Qwen reranker: 36.9 (+7.0 nDCG@10)
- Reranker model: likely `Qwen/Qwen2-Instruct` fine-tuned as reranker, or `BAAI/bge-reranker-v2-m3`

### LLM listwise reranking on BRIGHT-style queries:
- With reasoning-augmented prompt for reranker:
  ```
  "Given this question that requires multi-step reasoning: {query}
  And this reasoning chain: {reasoning}
  Rank the following passages by how well they support the reasoning and answer the question:
  [passages]"
  ```

---

## Practical Guidance: Which Model to Use for RETECO

### If running locally (compute-constrained):
1. **Keep e5-base-v2** but upgrade to H+R query strategy → biggest bang for buck
2. Add cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) 

### If loading 7-8B model feasible (A100 or similar):
1. **ReasonIR-8B** (`facebook/ReasonIR-8B`) — best open-source reasoning retriever
2. Use H+R query strategy (generate with SAP AI Core, embed with ReasonIR)
3. Add Qwen reranker
4. **Expected: ~0.55-0.60 macro nDCG@10** (matching/exceeding published SOTA)

### If targeting 0.7+:
- Need DIVER + H+R approach (current best = .584)
- OR: custom RL fine-tuning (Mira-style GRPO, very expensive)
- OR: iterative refinement with multiple retrieval rounds + LLM reasoning

### Cost/performance matrix:

| Approach | Cost | Complexity | Expected nDCG@10 |
|----------|------|-----------|-----------------|
| BM25 + QR (current) | Low | Done | .36 (dev) |
| BM25 + H+R | Low | Prompt change only | ~.48 |
| e5-base-v2 + H+R | Low | Small code change | ~.43 |
| Hybrid RRF + H+R | Medium | 1-2 days | ~.50-52 |
| ReasonIR + H+R | Medium | Load 8B model | ~.55 |
| DIVER + H+R | High | Load model + optimize | ~.58 |
| DIVER + H+R + reranker | High | Full pipeline | ~.62? |
| GRPO fine-tuning | Very high | Weeks of training | ~.65+ |
