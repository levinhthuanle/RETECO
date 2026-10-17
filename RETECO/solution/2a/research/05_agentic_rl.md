# Direction 5: Agentic & RL-based Conversational Search (Emerging, 2025–2026)

Hướng mới nhất, xa hơn retrieval pipeline truyền thống. Thay vì chuỗi cứng
rewrite→retrieve→generate, các approach này dùng agent/RL để *decide* khi nào
và cái gì cần retrieve.

---

## P1 — ConvSearch-R1: RL cho Query Reformulation (EMNLP 2025)
**EMNLP 2025** | arXiv:2505.15776  
**Authors:** Zhu, Wang, Feng, Song, Qiu (Fudan)

**Core idea:** Self-driven RL framework:
1. Warm-up: Retrieval-guided self-distillation (không cần human labels)
2. RL: reward = retrieval signal (recall/nDCG cải thiện hay không)
3. Model 3B params fine-tuned thành specialist rewriter

**Results:** >10% improvement trên TopiOCQA vs prior SOTA. Không cần external supervision.

**Relevance cho RETECO: MEDIUM** — Cần fine-tuning nhưng không cần human-labeled rewrites. Nếu có compute, train trên RETECO dev set.

---

## P2 — Agentic Conversational Search với Contextualized Reasoning (ACL 2026 Findings)
**ACL 2026 Findings** | arXiv:2601.13115  
**Authors:** Mo, Gao, Li, Zeng, Liu, Tan, Li, Chen, Wang, Jiang

**Core idea:** RL agent xử lý *mixed-initiative* actions trong multi-turn conversation:
- Khi nào retrieve
- Khi nào ask clarification
- Khi nào generate
- Khi nào skip (nếu context đủ)

Không dùng static pipeline. Agent học từ reward về *evolving user intent*.

**Results:** Outperforms competitive baselines trên 4 benchmarks.

**Relevance: MEDIUM** — RETECO là offline evaluation (không có interactive), nên full agent overkill. Nhưng *intent-tracking* idea hữu ích.

---

## P3 — ChatR1: RL cho Conversational RAG (ACL 2026 Main)
**ACL 2026 Main** | arXiv:2510.13312  
**Authors:** Lupart, Aliannejadi, Kanoulas (IRLab Amsterdam)

**Core idea:** RL với **intent-aware reward** — reward function được thiết kế để align với *evolving user intent* qua turns, không chỉ reward ở final turn.

Model 3B và 7B. Tested trên 5 conversational QA datasets.

**Results:** Outperforms baselines trên F1, BERTScore, LLM-as-judge.

**Relevance: MEDIUM** — Chủ yếu cho QA generation, ít trực tiếp cho retrieval ranking. Nhưng intent-aware reward concept có thể apply cho rewriter training.

---

## P4 — AdaQR: Adaptive Query Rewriting (arXiv 2024)
**arXiv 2406.10991** (2024)  
**Authors:** Zhang, Li, Luo, Wu, Glass, Meng

**Core idea:** Train rewriter với limited annotation bằng DPO. Reward = marginal probability của correct answers conditioned on conversation query thông qua top-K retrieved passages.

Không cần explicit rewrite labels, chỉ cần QA pairs.

**Results:** Outperforms SOTA trên 2 benchmarks.

**Relevance: MEDIUM** — Approach dùng QA signal (có thể dùng nếu RETECO có answer data, nhưng track 2a chỉ có relevance judgments).

---

## Tóm tắt xu hướng 2025–2026

Phân kỳ thành 2 luồng:

**Luồng 1 (Pragmatic):** Multi-query + Hybrid + LLM reranking  
→ Không cần training, high performance, deployable  
→ Phù hợp SemEval competition

**Luồng 2 (Research frontier):** RL agents, intent-aware rewards  
→ Cần training, hướng dài hạn  
→ Potential breakthrough nhưng cần resource

RETECO nên focus **Luồng 1** trước, sau đó nếu có time thử **RL rewriter** (ConvSearch-R1 style) trên RETECO training data.
