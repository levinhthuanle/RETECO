"""
embedder.py — Dense retrieval index for RETECO Track 2a.

Model: intfloat/e5-base-v2
- Passages prefixed with "passage: "
- Queries prefixed with "query: "
- Cosine similarity via faiss IndexFlatIP (on L2-normalized vectors = cosine)

Index is cached per domain as output/index/{domain}/embeddings.npy + doc_ids.json.
On re-run the cache is loaded, skipping re-encoding (~2 min per large domain).
"""

import json
import numpy as np
from pathlib import Path

import faiss  # must be imported before sentence_transformers to avoid libomp thread conflict
import torch
from sentence_transformers import SentenceTransformer


# ── model singleton ───────────────────────────────────────────────────────────

_models: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = "intfloat/e5-base-v2") -> SentenceTransformer:
    if model_name not in _models:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print(f"  [embedder] loading {model_name} on {device}")
        _models[model_name] = SentenceTransformer(model_name, device=device)
    return _models[model_name]


# ── corpus encoding ───────────────────────────────────────────────────────────

def load_corpus(domain_dir: Path) -> tuple[list[str], list[str]]:
    doc_ids, texts = [], []
    with open(domain_dir / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                doc_ids.append(d["doc_id"])
                texts.append("passage: " + d["content"])
    return doc_ids, texts


def _make_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def build_index(
    domain_dir: Path,
    cache_dir: Path,
    model_name: str = "intfloat/e5-base-v2",
    batch_size: int = 256,
) -> tuple[faiss.Index, list[str]]:
    """Build (or load from cache) a faiss index for the domain corpus."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "doc_ids.json"

    if emb_path.exists() and ids_path.exists():
        print(f"  [embedder] loading cached index for {domain_dir.name}")
        embeddings = np.load(str(emb_path))
        doc_ids = json.loads(ids_path.read_text())
    else:
        print(f"  [embedder] encoding {domain_dir.name} corpus...")
        doc_ids, texts = load_corpus(domain_dir)
        model = _get_model(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(str(emb_path), embeddings)
        ids_path.write_text(json.dumps(doc_ids))
        print(f"  [embedder] saved index: {embeddings.shape}")

    return _make_index(embeddings), doc_ids


# ── query search ──────────────────────────────────────────────────────────────

def search(
    index: faiss.Index,
    doc_ids: list[str],
    queries: dict[str, str],
    model_name: str = "intfloat/e5-base-v2",
    top_k: int = 100,
    batch_size: int = 256,
) -> dict[str, list[tuple[str, float]]]:
    """Encode queries and retrieve top_k docs. Returns {qid: [(docid, score)]}."""
    model = _get_model(model_name)

    qids = list(queries.keys())
    texts = ["query: " + q for q in queries.values()]

    q_embs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores_mat, idxs_mat = index.search(q_embs, top_k)

    results = {}
    for i, qid in enumerate(qids):
        ranked = [
            (doc_ids[idx], float(scores_mat[i, j]))
            for j, idx in enumerate(idxs_mat[i])
            if idx >= 0
        ]
        results[qid] = ranked
    return results
