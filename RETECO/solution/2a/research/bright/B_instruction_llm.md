# B: Instruction-Following LLMs & Hypothetical Document Methods

## Tại sao Instruction-Following Matters cho BRIGHT

Standard embedding: `embed(query)` → compare với `embed(doc)`
Instruction-following: `embed(instruction + query)` → more focused comparison

Với BRIGHT, instruction giúp model:
1. Biết LOẠI thông tin cần tìm (mechanistic explanation, procedure, etc.)
2. Ignore surface similarity, focus vào reasoning relevance

---

## LLM-Based Embedding Models với Instructions

### E5-Instruct (MTEB top, Wang et al., 2024)
**Paper:** "Improving Text Embeddings with Large Language Models" (ICLR 2024)
- arXiv:2401.00368
- Authors: Wang, Yang, Huang, Ma, Liu, He, Lu, Deng (Microsoft Research)

**Core idea:**
1. Synthetic data generation: GPT-4 tạo (instruction, query, positive_doc, hard_neg) tuples
2. 93 embedding tasks → rich instruction diversity
3. Fine-tune Mistral-7B/E5-Mistral → instruction-aware embedder

**BRIGHT performance:**
- E5-Instruct (E5-Mistral-7B): **~11.1 nDCG@10** (from MTEB BRIGHT subset)
- Significantly better than E5-base-v2 (~4.5) but still limited

**Why limited on BRIGHT:** Instructions help focus but don't add reasoning capability. Model still maps query → embedding via single forward pass without explicit reasoning.

**Instruction template cho BRIGHT-style retrieval:**
```
"Given a question that requires multi-step reasoning, retrieve relevant scientific documents"
```

---

### Qwen-Text-Embedding (Alibaba, 2025)
**Model:** Qwen3-Embedding (released 2025)
- Base: Qwen3 LLM family

**BRIGHT:** Strong performance (leaderboard entries ~40-55 range)
**RECOR (H+R):** 0.461 macro

**MTEB rank:** Top-5 on MTEB Retrieval as of 2025
**Key strength:** Multilingual + long context (up to 32K tokens) → handles full conversation history

---

### SFR-Embedding (Salesforce, 2024)
**Paper:** "SFR-Embedding-Mistral: Enhance Text Retrieval with Transfer Learning" (2024)
- Model: SFR-Embedding-Mistral (7B)

**RECOR (H+R):** 0.464 macro (slightly better than Qwen-TE)
**BRIGHT:** Competitive, in 30-40 range

**Method:** Continue from E5-Mistral, add fine-tuning on task-specific data including reasoning benchmarks.

---

## Hypothetical Document Embeddings (HyDE) for Reasoning Retrieval

### Original HyDE (Gao et al., 2022)
**Paper:** "Precise Zero-Shot Dense Retrieval without Relevance Labels" (ACL 2023)
- arXiv:2212.10496

**Core idea:**
```
Query → LLM generates hypothetical document → embed(hyp_doc) → retrieve similar docs
```

Instead of `similarity(embed(query), embed(doc))`, use `similarity(embed(hyp_answer), embed(doc))`

**Why this works for BRIGHT:**
- BRIGHT docs contain *relevant knowledge*, not query paraphrases
- Hypothetical answer is in the same "space" as real answers
- Bridges the lexical and semantic gap between question and answer document

**Gains:** HyDE with GPT-3.5 → +20-30% on dense retrieval benchmarks. On BRIGHT specifically, hypothetical document is much more on-target than query alone.

---

### HyDE-R: Reasoning-Augmented Hypothetical Documents

**Extension for BRIGHT-style retrieval (pattern observed in multiple 2025 papers):**

```
Query → LLM CoT → reasoning chain → hypothetical answer doc → embed → retrieve
```

Step-by-step:
1. LLM generates reasoning: "To answer X, I need to understand Y, which requires Z..."
2. LLM generates hypothetical document: full fake answer as if from a textbook
3. Embed hypothetical document (not query, not reasoning alone)
4. Retrieve by document-to-document similarity

**Why stronger than plain HyDE:** Reasoning chain makes hypothetical doc more accurate; plain HyDE can hallucinate off-target content.

**Estimated gain on BRIGHT:** From ~0.15 (instruct-only) to ~0.25-0.35 (with reasoning + HyDE)

---

## Instruction Design for RECOR/BRIGHT Queries

### Best instruction template per strategy:

**Baseline (current turn only):**
```
Instruct: Given a question in a technical domain, retrieve the most relevant Wikipedia or StackExchange passage that answers it.
Query: {current_turn}
```

**QueryRewrite:**
```
Instruct: Given a standalone information need, retrieve the most relevant passage.
Query: {rewritten_query}
```

**Reasoning strategy (what RECOR paper calls "Reas"):**
```
Instruct: Given a question and its reasoning chain showing what information is needed, retrieve the passage that provides that information.
Query: {current_turn}
Reasoning: {llm_generated_chain_of_thought}
```

**History+Reasoning (H+R — best strategy):**
```
Instruct: Given a conversation history and a question, with a reasoning chain, retrieve the most relevant passage.
History: {conversation_history}
Query: {current_turn}
Reasoning: {llm_generated_chain_of_thought}
```

### How RECOR generates Reasoning queries:
From RECOR paper: Use LLM (GPT-4-class) to generate reasoning chain:
```
"Given this conversation history and current question, generate a reasoning chain explaining 
what information would be needed to answer the question. Then formulate a search query."
```

---

## InstructOR / INSTRUCTOR embeddings

**Paper:** "One Embedder, Any Task: Instruction-Finetuned Text Embeddings" (EMNLP 2023)
- arXiv:2212.09561

**Core idea:** Prepend task + domain instruction directly as text to model:
```
"Represent the {domain} {task_type} for {purpose}: {text}"
Example: "Represent the scientific question for retrieving relevant biology documents: {query}"
```

**Relevance:** Instructions can be domain-specific → useful for RECOR's 11 domains. Instruction format:
- Biology: "Represent the biology question for retrieving relevant scientific passages"
- Law: "Represent the legal question for retrieving relevant law passages"
- Drones: "Represent the robotics/drone question for retrieving relevant technical documents"

---

## Practical: What to implement for RETECO

### Option 1: HyDE with SAP AI Core (training-free, +10-20% estimated)
```python
def hyde_retrieve(query, history, domain):
    hyp_doc = llm_call(
        f"Write a 200-word factual passage that would directly answer: "
        f"Conversation: {history}\nQuestion: {query}"
    )
    query_embed = embed(hyp_doc)  # embed the hypothetical doc, not query
    return dense_retrieve(query_embed, top_k=100)
```

### Option 2: Reasoning-Augmented Query (H+R strategy, +15-25% estimated)
```python
def reasoning_query(query, history, domain):
    reasoning = llm_call(
        f"Conversation: {history}\nQuestion: {query}\n"
        f"Generate a reasoning chain and search query to find the answer."
    )
    combined = f"Conversation: {history}\nReasoning: {reasoning}\nQuery: {query}"
    return embed(combined)
```

### Option 3: Swap retriever to ReasonIR/DIVER
These need their own embedding models. DIVER available, ReasonIR on HuggingFace.
Most impactful but requires loading 8B parameter model.
