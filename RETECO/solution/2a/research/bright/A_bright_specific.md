# A: BRIGHT Benchmark — Design, Challenges, Specific Methods

## Tổng quan BRIGHT

**BRIGHT** (Benchmarking Retrieval with Implicit Grounding of Thought)
- Paper: "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval" (ACL 2024)
- Authors: Su, Lin, Chen, Shi, Li, Deng, Sabharwal, Xiong, Chen (Stanford, AI2, etc.)
- arXiv: 2407.12883

**Core idea:** Retrieval benchmarks cũ (BEIR, MTEB) đo semantic similarity matching. BRIGHT đo khả năng *reasoning* — relevant doc không chứa keywords từ query mà chứa thông tin cần thiết để reasoning dẫn tới câu trả lời.

---

## Tại sao Standard Methods Fail trên BRIGHT

### BM25 failure:
- nDCG@10 = **0.04** (gần như random)
- Reason: Query và relevant doc dùng hoàn toàn khác nhau từ ngữ (lexical gap lớn)
- Example: Query "How to fix segfault in CUDA?" → Relevant doc nói về memory alignment trong GPU architecture (không có "segfault", "CUDA", "fix")

### Standard dense (E5, BGE base) failure:
- nDCG@10 ≈ 0.06–0.12
- Reason: Embedding space của generic models capture semantic surface similarity, không phải reasoning path

### Instruct embeddings (E5-Instruct, INSTRUCTOR) moderate improvement:
- nDCG@10 ≈ 0.15–0.20
- Instruction giúp model focus đúng khía cạnh, nhưng không đủ để reason

### Why hard:
BRIGHT queries đến từ Stack Overflow (upvoted questions), CodeContests, TheoremQA — real user questions yêu cầu understanding underlying mechanism chứ không phải surface matching.

---

## BRIGHT — 12 Domains (single-turn benchmark)

| Domain | Type | #Queries | #Corpus |
|--------|------|----------|---------|
| Biology | Science | 114 | 3.7M |
| Earth Science | Science | 140 | 3.7M |
| Economics | Social | 104 | 3.7M |
| Psychology | Social | 116 | 3.7M |
| Robotics | Engineering | 119 | 3.7M |
| Stack Overflow | Programming | 130 | 4.5M |
| Sustainable Living | Life | 100 | 3.7M |
| Law | Professional | 93 | 2.1M |
| Leetcode | Programming | 150 | 50k |
| Pony | Niche | 96 | 48k |
| AoPS | Math | 80 | 3.7M |
| TheoremQA | Math | 200 | varies |

---

## BRIGHT-Specific Retrievers

### DIVER (Long et al., 2025)
**Paper:** "DIVER: Diverse and In-context Reasoning for Reasoning-Intensive Retrieval" (2025)
- arXiv: 2502.20134 (not confirmed, may be different)

**Core idea:** Reasoning-specialized embedding model được train để:
1. Generate diverse reasoning paths cho một query
2. Embed reasoning path + query together → retrieve docs that "answer" the reasoning
3. In-context learning: few-shot examples của reasoning chains guide embedding

**BRIGHT single-turn:** **46.8 nDCG@10** (DIVER-v3)
**RECOR multi-turn (H+R):** **0.584 macro** — SOTA trên RECOR

**Why it works:** Documents trên BRIGHT liên quan qua reasoning chain, không qua surface similarity. DIVER explicitly models this reasoning chain connection.

---

### ReasonIR (arXiv:2504.20595)
**Paper:** "ReasonIR: Training Reasoners for Dense Retrieval" (Meta AI, 2025)
- Authors: Shi, Wang, Shi, Chang, Yih, Karpukhin et al.

**Core idea:**
1. Synthetic data generation: GPT-4 tạo (query, reasoning_chain, relevant_doc) triplets từ reasoning-intensive tasks
2. Hard negative mining: docs liên quan surface nhưng không phải answer
3. Fine-tune 8B LLM (Llama-3-8B) làm bi-encoder

**BRIGHT scores:**
- Without reranker: **29.9 nDCG@10**
- With Qwen reranker: **36.9 nDCG@10**
- RECOR (H+R): **0.552 macro**

**Architecture:** 8B parameter LLM → single vector embedding. Memory cost cao hơn, nhưng reasoning capability mạnh hơn.

**Availability:** Model released on HuggingFace. Can run on A100 80GB.

---

### Mira-Reasoning-Retrieval (Forward AI Labs, 2025)
**BRIGHT Leaderboard #1:** 66.9 nDCG@10

**3-stage training pipeline:**
1. **Candidate rewrite mining**: Dùng multiple LLMs để generate diverse rewrite candidates của queries. Filter bằng retrieval performance.
2. **Cold start training**: Train retriever trên mined rewrites → base model.
3. **GRPO-driven intent alignment**: RL phase với multi-dimensional reward:
   - Retrieval reward: nDCG improvement
   - Faithfulness reward: rewrite faithful to original query
   - Coverage reward: rewrite covers all query aspects
   - Conciseness reward: no hallucination

**Key insight:** GRPO (Group Relative Policy Optimization, from DeepSeek-R1) applied to retrieval. Treat candidate rewrites as "thoughts" → optimize end-to-end retrieval quality.

**Gap from RECOR SOTA:** Mira achieves 66.9 on BRIGHT single-turn vs DIVER 0.584 macro on RECOR multi-turn. Direct comparison tricky (different eval), but Mira approach is stronger for single-turn.

---

### BGE-Reasoner
**BRIGHT Leaderboard #6:** 46.4 nDCG@10
- BGE-M3 fine-tuned với reasoning-augmented training data
- Uses CoT-style query augmentation at inference time

---

### NeMo Agentic Retrieval (NVIDIA)
**BRIGHT Leaderboard #4:** 50.9 nDCG@10
- Multi-step agentic approach: decompose query → retrieve → reason → refine query → retrieve again
- Multiple retrieval rounds với LLM-guided refinement
- Expensive at inference (multiple LLM calls per query) but shows ceiling

---

## Key Methods for BRIGHT-Style Retrieval

### Pattern 1: Query Augmentation with Reasoning
Add CoT reasoning step before embedding:
```
Query: "Why does CUDA OOM when batch > 64?"
→ LLM: "This requires reasoning about GPU memory: batch_size × tensor_dims × dtype_bytes > VRAM..."
→ Embed: "Query + reasoning chain" → retrieve
```
Gain: +10–15 nDCG@10 on BRIGHT (from ~0.18 to ~0.30+)

### Pattern 2: Hypothetical Document Embeddings (HyDE)
Generate hypothetical answer document → embed document → retrieve by document similarity:
```
Query: "X" → LLM generates "Ideal answer: ..." → embed answer → retrieve similar docs
```
Works well on BRIGHT because documents contain *answers*, not query paraphrases.

### Pattern 3: Reasoning-Specialized Models
Train retriever specifically for reasoning tasks. Not just "instruct" embeddings, but training data consisting of reasoning-intensive QA → retrieval pairs.

### Pattern 4: Iterative Refinement
Multiple rounds: retrieve → LLM reasons over docs → refine query → retrieve again.
Most expensive but approaching ceiling.
