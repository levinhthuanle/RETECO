# C: RECOR Paper — Full Analysis

**Paper:** "RECOR: A Benchmark for Reasoning-focused Multi-turn Conversational Retrieval"
**Venue:** ACL 2026 Findings
**Authors:** Mohammed Ali, Zhihao Zhang, David Losada, Guoxin Wang (University of Innsbruck + Oracle AI Research)
**arXiv:** To be confirmed (published ACL 2026)

---

## Dataset Construction

### Sources:
1. **BRIGHT benchmark** (6 domains): Biology, Earth Science, Economics, Psychology, Robotics, Sustainable Living
2. **StackExchange** (5 domains): Drones, Hardware, Law, Medical, Politics

### Statistics:
- **707** multi-turn conversations
- **2,971** turns total (~4.2 turns/conversation average)
- **507,141** documents in corpus
- **11 domains** total

### How conversations were created:
- Automatic conversation generation from BRIGHT single-turn queries
- Each turn is a sub-question toward the final information need
- Last turn = hardest (most context-dependent)
- Manual quality filtering for coherence

### Evaluation:
- Metric: **nDCG@10** per turn → macro-average across all turns and domains
- Relevance judgments: binary (0/1) or graded (BRIGHT uses graded)
- dev/test split provided

---

## 5 Query Strategies Evaluated

### Strategy 1: Baseline (current turn only)
```
Query = current_turn_text
```
No history, no reasoning. Simulates naïve retrieval.

### Strategy 2: Query Rewriting (QR)
```
prompt = "Rewrite this conversation turn as a standalone query: 
          History: {history}, Current: {current_turn}"
Query = LLM(prompt)
```
Resolves coreferences, fills ellipsis. Standard CONQRR/T5QR approach.

### Strategy 3: Reasoning (Reas)
```
prompt = "Generate a reasoning chain and search query: Question: {current_turn}"
Query = LLM(reasoning_chain + current_turn)
```
No history. Pure single-turn reasoning augmentation.

### Strategy 4: History (Hist)
```
Query = conversation_history + current_turn  (concatenated)
```
No rewriting, no reasoning. Raw history prepended.

### Strategy 5: History+Reasoning (H+R) — BEST
```
prompt = "Given conversation: {history}, current question: {current_turn},
          generate: 1) reasoning chain about what information is needed,
          2) a search query incorporating history + reasoning"
Query = LLM(history + reasoning_chain + current_turn)
```
Combines full context with explicit reasoning generation.

---

## Table 3: Main Results (Macro nDCG@10, all 11 domains)

| Retriever | Base | QR | Reas | Hist | H+R |
|-----------|------|-----|------|------|-----|
| DIVER | .347 | .430 | .496 | .545 | **.584** |
| ReasonIR | .266 | .357 | .494 | .496 | **.552** |
| Qwen-TE | .269 | .345 | .399 | .425 | .461 |
| SFR-Embed | .240 | .324 | .396 | .429 | .464 |
| BGE-M3 | .230 | .328 | .347 | .411 | .445 |
| E5-Instruct | .183 | .272 | .352 | .404 | .429 |
| BM25 | .185 | .288 | .360 | .446 | .489 |
| **Avg** | .236 | .322 | .393 | .440 | .479 |

### Key observations:
1. H+R is best strategy for ALL retrievers (avg +103% vs baseline)
2. History alone (Hist) better than Reasoning alone (Reas) for most models
3. BM25 + H+R (.489) > many dense models — reasoning query helps sparse too
4. Reasoning (Reas) helps dense models more than sparse (BM25 Reas .360 < E5 Reas .352 gap shows BM25 less benefit)
5. DIVER .584 vs ReasonIR .552 — both reasoning-specialized, ~5% gap

---

## Table 4: Query Strategy Analysis

**Effect sizes (avg across all retrievers):**

| Strategy | Avg nDCG@10 | Δ from Baseline |
|----------|------------|-----------------|
| Baseline | .236 | — |
| QueryRewrite | .322 | +.086 (+36%) |
| Reasoning | .393 | +.157 (+67%) |
| History | .440 | +.204 (+86%) |
| History+Reasoning | .479 | +.243 (+103%) |

**Interaction effects:**
- Reasoning alone > QR alone: reasoning augmentation is more powerful than coreference resolution
- History alone > Reasoning alone: conversation context is more important than single-turn reasoning
- H+R > History + Reas separately: synergistic, not additive

---

## Table 5: Per-Domain Results (Best system = DIVER + H+R)

| Domain | nDCG@10 | Type |
|--------|---------|------|
| Psychology | .733 | BRIGHT |
| Biology | .732 | BRIGHT |
| Earth Science | .694 | BRIGHT |
| Sustainable Living | .682 | BRIGHT |
| Robotics | .587 | BRIGHT |
| Politics | .582 | StackExchange |
| Economics | .572 | BRIGHT |
| Law | .540 | StackExchange |
| Drones | .499 | StackExchange |
| Medical | .488 | StackExchange |
| Hardware | .455 | StackExchange |

**Pattern:** BRIGHT domains generally easier than StackExchange domains for DIVER.
- BRIGHT domains: avg .657 (DIVER + H+R)
- StackExchange domains: avg .527 (DIVER + H+R)

**Hardest domains:** Hardware (.455), Medical (.488), Drones (.499) — highly technical Q&A with specialized terminology.

---

## Table 6: Per-Query Upper Bound Analysis

**Method:** For each query, take the MAX nDCG@10 across all (retriever × strategy) combinations.

- **Mean per-query maximum:** ~0.81
- **% queries with max ≥ 0.8:** 63.2%
- **% queries with max ≥ 0.6:** ~80%
- **% queries with max < 0.4:** ~12% (hard queries, likely no good approach)

**Implication:** The theoretical ceiling is ~0.81 if we could pick the best retriever+strategy per query. Current SOTA achieves 0.584 → ~28% gap to ceiling. Much of this gap could be closed by:
1. Better multi-turn reasoning integration
2. Ensemble/oracle retriever selection
3. Domain-specific fine-tuning

---

## RECOR vs Our Current System (optionA)

| Component | RECOR best | Our optionA | Gap |
|-----------|-----------|-------------|-----|
| Retriever | DIVER (reasoning-specialist) | BM25 (keyword-based) | Major |
| Query strategy | H+R (history + CoT reasoning) | QR (simple rewrite) | Major |
| Score | .584 (DIVER+H+R) | .3603 (BM25+QR dev) | .224 gap |
| BM25 score (comparable) | .489 (BM25+H+R) | .3603 (BM25+QR) | .129 gap |

**Most impactful change:** Upgrade query strategy from QR → H+R  
**Implementation:** Change LLM prompt in optionA to generate reasoning chain + include full history

---

## RECOR Paper: Reproducibility Notes

- All LLM calls for query generation: GPT-4-class (not specified exactly)
- DIVER model: available, need exact HuggingFace path
- ReasonIR model: `facebook/ReasonIR-8B` (confirmed on HF)
- BM25: Apache Lucene BM25 (not Python rank_bm25) → explains our lower score

**Our BM25 vs official:**
- Our optionA (rank_bm25): .3603 with QueryRewrite
- Official RECOR (Lucene BM25 + QR): ~.288 
- Hmm — our score HIGHER. May be because different eval set (dev vs test) or different rewrite quality.
- BM25 official + H+R: .489 — this is the target if we stay with BM25 retriever.
