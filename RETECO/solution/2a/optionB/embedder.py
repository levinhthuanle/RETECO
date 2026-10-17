"""
embedder.py — Dense retrieval index for RETECO Track 2a.

Supported models:
  intfloat/e5-base-v2          (default)  — prefix "passage:"/"query:"
  AQ-MedAI/Diver-Retriever-4B-1020       — SentenceTransformer prompt_name="query"
  reasonir/ReasonIR-8B                   — SentenceTransformer trust_remote_code=True

Index is cached per domain+model as output/index/{model_slug}/{domain}/embeddings.npy.
On re-run the cache is loaded, skipping re-encoding.
"""

import json
import numpy as np
import re
from pathlib import Path

import faiss  # must be imported before sentence_transformers to avoid libomp thread conflict
import torch
from sentence_transformers import SentenceTransformer


# ── model config ──────────────────────────────────────────────────────────────

# Models that use SentenceTransformer prompt_name="query" for query side
# (no manual prefix needed — the model's built-in prompt handles it)
_PROMPT_NAME_MODELS = {
    "AQ-MedAI/Diver-Retriever-4B-1020",
    "reasonir/ReasonIR-8B",
}

# Models that need trust_remote_code=True
_TRUST_REMOTE_MODELS = {
    "reasonir/ReasonIR-8B",
}

# Models that use "passage: " / "query: " prefixes (e5-style)
_E5_PREFIX_MODELS = {
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "intfloat/e5-mistral-7b-instruct",
}

# Max sequence length override for large models (default 8192 is too slow on MPS/CPU)
_MAX_SEQ_LENGTH: dict[str, int] = {
    "AQ-MedAI/Diver-Retriever-4B-1020": 512,
    "reasonir/ReasonIR-8B": 512,
}

def _model_slug(model_name: str) -> str:
    """Convert model name to filesystem-safe slug for cache paths."""
    return re.sub(r"[^\w\-]", "_", model_name)


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

        kwargs: dict = {"device": device}
        if model_name in _TRUST_REMOTE_MODELS:
            kwargs["trust_remote_code"] = True
        if model_name not in _E5_PREFIX_MODELS:
            # LLM-based models benefit from explicit dtype
            kwargs["model_kwargs"] = {"torch_dtype": "auto"}

        _models[model_name] = SentenceTransformer(model_name, **kwargs)
        if model_name in _PROMPT_NAME_MODELS:
            _models[model_name].set_pooling_include_prompt(include_prompt=False)
        if model_name in _MAX_SEQ_LENGTH:
            _models[model_name].max_seq_length = _MAX_SEQ_LENGTH[model_name]
            print(f"  [embedder] max_seq_length set to {_MAX_SEQ_LENGTH[model_name]}")
    return _models[model_name]


# ── corpus encoding ───────────────────────────────────────────────────────────

def load_corpus(domain_dir: Path) -> tuple[list[str], list[str]]:
    doc_ids, texts = [], []
    with open(domain_dir / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                doc_ids.append(d["doc_id"])
                # e5-style models need "passage: " prefix; others embed raw text
                texts.append(d["content"])
    return doc_ids, texts


def _prefix_docs(texts: list[str], model_name: str) -> list[str]:
    if model_name in _E5_PREFIX_MODELS:
        return ["passage: " + t for t in texts]
    return texts


def _prefix_queries(texts: list[str], model_name: str) -> list[str]:
    if model_name in _E5_PREFIX_MODELS:
        return ["query: " + t for t in texts]
    return texts


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
    # Cache keyed by model slug so different models don't collide
    model_cache_dir = cache_dir / _model_slug(model_name) / domain_dir.name
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = model_cache_dir / "embeddings.npy"
    ids_path = model_cache_dir / "doc_ids.json"

    if emb_path.exists() and ids_path.exists():
        print(f"  [embedder] loading cached index for {domain_dir.name} ({model_name})")
        embeddings = np.load(str(emb_path))
        doc_ids = json.loads(ids_path.read_text())
    else:
        print(f"  [embedder] encoding {domain_dir.name} corpus with {model_name}...")
        doc_ids, texts = load_corpus(domain_dir)
        model = _get_model(model_name)
        prefixed = _prefix_docs(texts, model_name)
        embeddings = model.encode(
            prefixed,
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
    batch_size: int = 64,
) -> dict[str, list[tuple[str, float]]]:
    """Encode queries and retrieve top_k docs. Returns {qid: [(docid, score)]}."""
    model = _get_model(model_name)

    qids = list(queries.keys())
    texts = list(queries.values())
    prefixed = _prefix_queries(texts, model_name)

    encode_kwargs: dict = dict(
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if model_name in _PROMPT_NAME_MODELS:
        encode_kwargs["prompt_name"] = "query"

    q_embs = model.encode(prefixed, **encode_kwargs).astype(np.float32)

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
