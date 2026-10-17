# Direction 3: Hybrid Retrieval (Sparse + Dense Fusion)

Kết hợp BM25/SPLADE (sparse) và bi-encoder (dense) để tận dụng ưu điểm của cả hai.
Sparse tốt ở exact match và keyword-heavy queries; dense tốt ở semantic matching.
Trong conversational IR, hybrid thường vượt trội vì queries sau rewriting vẫn có keywords rõ.

---

## P1 — Sifei@SemEval-2026 Task 8: Hybrid + Rewriting (SemEval 2026 / ACL 2026)
**ACL 2026 (SemEval workshop)** | arXiv:2606.28352  
**Authors:** Meng, Ilvovsky

**Core idea:** Pipeline hoàn chỉnh cho multi-turn RAG:
1. Controlled query rewriting (LLM, với temperature thấp để ổn định)
2. Hybrid retrieval: dense (bi-encoder) + sparse (BM25) fused bằng RRF
3. Cross-encoder reranking
4. Generation

Training-free hoàn toàn.

**Results:** nDCG@5 = 0.5453, xếp hạng 3/38 teams tại SemEval-2026 Task 8 (multi-turn RAG).

**Relevance: VERY HIGH** — SemEval task khác nhưng setup rất giống RETECO. Và đây là training-free method. RRF hybrid là baseline mạnh cần implement.

---

## P2 — uva-irlab-conv@SemEval-2026 Task 8: Sparse + Listwise Reranking (SemEval 2026 / ACL 2026)
**ACL 2026 (SemEval workshop)** | arXiv:2606.11945  
**Authors:** Lupart, Mekonnen, Abbasiantaeb, Aliannejadi (IRLab Amsterdam)

**Core idea:** Pipeline cho SemEval-2026 Task 8:
1. LLM-based conversational query rewriting (lấy full history)
2. Learned Sparse Retrieval (SPLADE variant) làm primary retriever
3. LLM pointwise + **listwise reranking** (đánh giá cả list thay vì từng document)
4. Response generation

**Listwise reranking** là điểm đặc biệt — LLM nhận toàn bộ top-k candidates và rerank một lần (eficient hơn pointwise N calls).

**Results:** Không công bố số cụ thể trong abstract nhưng là top-tier system.

**Relevance: HIGH** — Listwise reranking LLM là improvement trực tiếp so với pointwise cross-encoder.

---

## P3 — CFDA & CLIP@iKAT 2025: RRF Fusion (iKAT 2025)
**TREC iKAT 2025** | arXiv:2509.15588  
**Authors:** Chang, Yeo, Eugene, Shih, Kuo, Yu, Hsu, Tsai, Wang (Taiwan)

**Core idea:** Best-of-N selection cho query rewriting (sinh nhiều rewrites, chọn tốt nhất theo criteria) + Reciprocal Rank Fusion để merge results từ nhiều queries/retrievers.

**Results:** Competitive iKAT 2025 system.

**Relevance: MEDIUM-HIGH** — RRF là simple nhưng effective. "Best-of-N" selection cho rewriting là interesting alternative cho top-1 rewrite.

---

## P4 — Multimodal Hybrid RAG cho Scientific Documents (arXiv 2026)
**arXiv 2026** | (unnamed system paper)  
**Authors:** Saucă, Rusnac

**Core idea:** HNSW-based semantic search (dense) + GIN-based lexical search (BM25-like) fused bằng RRF + cross-encoder reranking. Thêm Query Condenser module cho conversational coherence.

**Relevance: MEDIUM** — Kiến trúc hybrid tốt nhưng focus multimodal, ít directly applicable.

---

## Kết quả thực nghiệm: Hybrid vs Single-retriever

Từ các papers thu thập được, pattern rõ ràng:

| System | Sparse only | Dense only | Hybrid (RRF) |
|--------|-------------|-----------|--------------|
| Typical gain | baseline | ±0-5% | +5–15% trên Recall@100 |
| iKAT/CAsT best systems | | | hybrid phổ biến |
| SemEval-2026 top-3 | | | tất cả dùng hybrid |

**Vì sao hybrid tốt trong conversational IR:**
- Rewritten queries thường vẫn có strong keywords → BM25 bắt được
- Semantic drift và paraphrases → dense bắt được
- RRF ổn định, không cần hyperparameter tuning nhiều

---

## Implementation note cho RETECO

Công thức đơn giản nhất (không cần training):
```
final_rank = RRF(BM25_rank, dense_rank)  # k=60 (default)
```

Nâng cấp:
1. BM25 → SPLADE (learned sparse, vocab expansion)
2. Dense → fine-tuned e5 hoặc ContextualRetriever
3. RRF → Learned fusion (với labels từ RETECO train set)

**Ước tính gain:** +5–10% nDCG@10 so với best single-retriever, dựa trên literature.
