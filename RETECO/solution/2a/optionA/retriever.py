"""
retriever.py — BM25 retriever for RETECO Track 2a.

Uses rank_bm25 (pure Python, no JDK needed) with simple whitespace
tokenization. Matches the spirit of the official BM25 baseline (k1=0.9,
b=0.4) but is dependency-free.

For each domain: build index once, search with arbitrary query strings.
Output: TREC run file (qid Q0 docid rank score tag).
"""

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_corpus(domain_dir: Path) -> tuple[list[str], list[str]]:
    doc_ids, texts = [], []
    with open(domain_dir / "documents.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                doc_ids.append(d["doc_id"])
                texts.append(d["content"])
    return doc_ids, texts


def load_qrels(domain_dir: Path, split: str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    qrel_file = domain_dir / f"qrels_{split}.txt"
    if not qrel_file.exists():
        return qrels
    with open(qrel_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, docid, rel = parts
                if int(rel) > 0:
                    qrels.setdefault(qid, set()).add(docid)
    return qrels


class BM25Index:
    def __init__(self, domain_dir: Path):
        self.doc_ids, texts = load_corpus(domain_dir)
        tokenized = [_tokenize(t) for t in tqdm(
            texts, desc=f"  index {domain_dir.name}", leave=False, unit="doc"
        )]
        self.bm25 = BM25Okapi(tokenized, k1=0.9, b=0.4)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(self.doc_ids, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]


def write_run(
    run: dict[str, list[tuple[str, float]]],
    out_path: Path,
    tag: str = "bm25_rewrite",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, ranked in run.items():
            for rank, (docid, score) in enumerate(ranked, 1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n")


def run_retrieval(
    index: BM25Index,
    queries: dict[str, str],
    top_k: int = 100,
) -> dict[str, list[tuple[str, float]]]:
    results = {}
    for qid, query in tqdm(queries.items(), desc="  search", leave=False, unit="q"):
        results[qid] = index.search(query, top_k)
    return results
