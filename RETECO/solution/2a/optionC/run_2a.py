#!/usr/bin/env python3
"""
run_2a.py — RETECO Track 2a · Option C: Hybrid BM25 + Dense (RRF).

Combines BM25 and e5-base-v2 dense retrieval via Reciprocal Rank Fusion (RRF).
Supports split query strategies: different queries for BM25 vs dense legs.

RRF formula: score(doc) = 1/(k + rank_bm25) + 1/(k + rank_dense)  [k=60]

Usage:
    # Split strategy (best): BM25 uses history, dense uses H+R reasoning
    python run_2a.py --data ../../../../reteco_data --splits dev --bm25-strategy history --dense-strategy hr

    # Same strategy for both legs:
    python run_2a.py --data ../../../../reteco_data --splits dev --query-strategy hr

    # Smoke test single domain:
    python run_2a.py --data ../../../../reteco_data --splits dev --domains drones

Results:
    output/runs/run_hybrid_<bm25_strategy>+<dense_strategy>_<domain>_<split>.trec
    output/results/summary.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# faiss must be imported before sentence_transformers
sys.path.insert(0, str(Path(__file__).parent.parent / "optionB"))
from embedder import build_index, search as dense_search, _get_model  # noqa: E402

import pytrec_eval
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "optionA"))
from retriever import BM25Index, load_corpus, load_qrels, write_run  # noqa: E402


TRACK2_DOMAINS = [
    "biology", "drones", "earth_science", "economics", "hardware",
    "law", "medicalsciences", "politics", "psychology", "robotics",
    "sustainable_living",
]

OPTIONA_REWRITES = Path(__file__).parent.parent / "optionA" / "output" / "rewrites"


# ── query builders ─────────────────────────────────────────────────────────

def build_queries_history(domain_dir: Path, split: str) -> dict[str, str]:
    bench = json.loads((domain_dir / f"benchmark_{split}.json").read_text())
    out = {}
    for conv in bench:
        for turn in conv["turns"]:
            qid = f"{conv['id']}_turn_{turn['turn_id']}"
            hist = turn.get("conversation_history", "")
            if hist.lower().startswith("no previous"):
                hist = ""
            q = (turn["query"] + "\n\nConversation History:\n" + hist).strip() if hist else turn["query"]
            out[qid] = q
    return out


def build_queries_rewrite(domain_dir: Path, split: str, strategy: str) -> dict[str, str]:
    cache_dir = OPTIONA_REWRITES / strategy / domain_dir.name / split
    if not cache_dir.exists():
        print(f"  [WARN] no {strategy} cache for {domain_dir.name}/{split}, using history")
        return build_queries_history(domain_dir, split)

    bench = json.loads((domain_dir / f"benchmark_{split}.json").read_text())
    hist_queries = build_queries_history(domain_dir, split)
    out = {}
    for conv in bench:
        for turn in conv["turns"]:
            qid = f"{conv['id']}_turn_{turn['turn_id']}"
            fname = re.sub(r"[^\w\-]", "_", qid) + ".txt"
            cache_file = cache_dir / fname
            if cache_file.exists():
                out[qid] = cache_file.read_text(encoding="utf-8").strip()
            else:
                out[qid] = hist_queries.get(qid, turn["query"])
    return out


# ── RRF fusion ─────────────────────────────────────────────────────────────

def rrf_merge(
    runs: list[dict[str, list[tuple[str, float]]]],
    k: int = 60,
    top_k: int = 100,
) -> dict[str, list[tuple[str, float]]]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    all_qids = set()
    for run in runs:
        all_qids.update(run.keys())

    merged = {}
    for qid in all_qids:
        scores: dict[str, float] = {}
        for run in runs:
            ranked = run.get(qid, [])
            for rank, (docid, _) in enumerate(ranked, 1):
                scores[docid] = scores.get(docid, 0.0) + 1.0 / (k + rank)
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        merged[qid] = sorted_docs[:top_k]
    return merged


# ── scoring ────────────────────────────────────────────────────────────────

def score_run(run: dict, qrels: dict, k: int = 10) -> dict:
    if not run or not qrels:
        return {}
    qrels_d = {q: {d: 1 for d in ds} for q, ds in qrels.items()}
    run_d   = {q: {d: s for d, s in ranked} for q, ranked in run.items() if q in qrels_d}
    ev = pytrec_eval.RelevanceEvaluator(qrels_d, {f"ndcg_cut.{k}", f"map_cut.{k}", "recip_rank"})
    sc = ev.evaluate(run_d)
    if not sc:
        return {}
    n = len(sc)
    return {
        f"nDCG@{k}": round(sum(v[f"ndcg_cut_{k}"] for v in sc.values()) / n, 5),
        f"MAP@{k}":  round(sum(v[f"map_cut_{k}"]  for v in sc.values()) / n, 5),
        "MRR":       round(sum(v["recip_rank"]     for v in sc.values()) / n, 5),
        "num_topics": n,
    }


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--data",           default=str(here / ".." / ".." / ".." / ".." / "reteco_data"))
    ap.add_argument("--out",            default=str(here / "output"))
    ap.add_argument("--splits",         nargs="+", default=["dev"])
    ap.add_argument("--domains",        nargs="*", default=None)
    ap.add_argument("--query-strategy", default=None, choices=["history", "qr", "hr"],
                    help="Query strategy for BOTH legs (overrides --bm25-strategy/--dense-strategy)")
    ap.add_argument("--bm25-strategy", default="history", choices=["history", "qr", "hr"],
                    help="Query strategy for BM25 leg (default: history)")
    ap.add_argument("--dense-strategy", default="hr", choices=["history", "qr", "hr"],
                    help="Query strategy for dense leg (default: hr)")
    ap.add_argument("--dense-model",    default="intfloat/e5-base-v2",
                    help="Dense model to use (default: intfloat/e5-base-v2)")
    ap.add_argument("--rrf-k",          type=int, default=60,
                    help="RRF k parameter (default: 60)")
    ap.add_argument("--top-k",          type=int, default=100)
    ap.add_argument("--batch",          type=int, default=256)
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    out_dir  = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = args.domains or TRACK2_DOMAINS
    model_slug = re.sub(r"[^\w\-]", "_", args.dense_model.split("/")[-1])

    # Resolve strategies: --query-strategy overrides both legs
    if args.query_strategy is not None:
        bm25_strategy = args.query_strategy
        dense_strategy = args.query_strategy
    else:
        bm25_strategy = args.bm25_strategy
        dense_strategy = args.dense_strategy

    tag = f"hybrid_{model_slug}_{bm25_strategy}+{dense_strategy}_rrf{args.rrf_k}"

    # preload dense model once
    _get_model(args.dense_model)

    all_results: dict[str, dict] = defaultdict(dict)

    for domain in domains:
        domain_dir = data_dir / "track2_recor" / domain
        if not domain_dir.exists():
            print(f"[SKIP] {domain}")
            continue

        print(f"\n=== {domain} ===")

        # --- build queries per split (separate for each leg) ---
        queries_bm25_by_split: dict[str, dict[str, str]] = {}
        queries_dense_by_split: dict[str, dict[str, str]] = {}
        for split in args.splits:
            if bm25_strategy in ("qr", "hr"):
                queries_bm25_by_split[split] = build_queries_rewrite(domain_dir, split, bm25_strategy)
            else:
                queries_bm25_by_split[split] = build_queries_history(domain_dir, split)

            if dense_strategy == bm25_strategy:
                queries_dense_by_split[split] = queries_bm25_by_split[split]
            elif dense_strategy in ("qr", "hr"):
                queries_dense_by_split[split] = build_queries_rewrite(domain_dir, split, dense_strategy)
            else:
                queries_dense_by_split[split] = build_queries_history(domain_dir, split)

        # --- BM25 index (built once per domain) ---
        print(f"  building BM25 index...", flush=True)
        bm25_index = BM25Index(domain_dir)

        # --- Dense index ---
        # Reuse optionB cache if it exists, otherwise encode into optionC cache
        model_slug_str = re.sub(r"[^\w\-]", "_", args.dense_model)
        optionb_cache = Path(__file__).parent.parent / "optionB" / "output" / "index"
        optionc_cache = out_dir / "index"
        # build_index creates {cache_dir}/{model_slug}/{domain}/ — try optionB first
        optionb_hit = (optionb_cache / model_slug_str / domain / "embeddings.npy").exists()
        index_base = optionb_cache if optionb_hit else optionc_cache
        faiss_index, doc_ids = build_index(
            domain_dir, index_base,
            model_name=args.dense_model,
            batch_size=args.batch,
        )

        for split in args.splits:
            bench_file = domain_dir / f"benchmark_{split}.json"
            if not bench_file.exists():
                continue
            qrels = load_qrels(domain_dir, split)
            if not qrels:
                continue

            bm25_queries = {q: v for q, v in queries_bm25_by_split[split].items() if q in qrels}
            dense_queries = {q: v for q, v in queries_dense_by_split[split].items() if q in qrels}
            if not bm25_queries:
                continue

            # --- BM25 retrieval ---
            print(f"  [{split}] BM25 retrieval ({len(bm25_queries)} queries)...", flush=True)
            bm25_run: dict[str, list[tuple[str, float]]] = {}
            for qid, query in tqdm(bm25_queries.items(), desc="  BM25", leave=False):
                bm25_run[qid] = bm25_index.search(query, top_k=args.top_k)

            # --- Dense retrieval ---
            print(f"  [{split}] dense retrieval...", flush=True)
            dense_run = dense_search(
                faiss_index, doc_ids, dense_queries,
                model_name=args.dense_model,
                top_k=args.top_k,
                batch_size=args.batch,
            )

            # --- RRF fusion ---
            hybrid_run = rrf_merge([bm25_run, dense_run], k=args.rrf_k, top_k=args.top_k)

            run_path = out_dir / "runs" / f"run_{tag}_{domain}_{split}.trec"
            write_run(hybrid_run, run_path, tag)

            metrics = score_run(hybrid_run, qrels)
            all_results[domain][split] = metrics

            # also score individual legs for comparison
            bm25_m  = score_run(bm25_run, qrels)
            dense_m = score_run(dense_run, qrels)
            print(f"  [{split}] BM25:  nDCG@10 {bm25_m.get('nDCG@10',0):.4f}")
            print(f"  [{split}] Dense: nDCG@10 {dense_m.get('nDCG@10',0):.4f}")
            print(f"  [{split}] RRF:   nDCG@10 {metrics.get('nDCG@10',0):.4f}  "
                  f"MAP@10 {metrics.get('MAP@10',0):.4f}  "
                  f"MRR {metrics.get('MRR',0):.4f}  "
                  f"({metrics.get('num_topics',0)} topics)")

    # macro-average
    print("\n" + "=" * 60)
    print(f"MACRO-AVERAGE nDCG@10  [{tag}]")
    print("=" * 60)

    summary: dict = {"tag": tag, "per_domain": dict(all_results), "macro_average": {}}
    for split in args.splits:
        domain_scores = [all_results[d][split] for d in domains if split in all_results.get(d, {})]
        if not domain_scores:
            continue
        macro = {
            "nDCG@10": round(sum(s["nDCG@10"] for s in domain_scores) / len(domain_scores), 5),
            "MAP@10":  round(sum(s["MAP@10"]  for s in domain_scores) / len(domain_scores), 5),
            "MRR":     round(sum(s["MRR"]     for s in domain_scores) / len(domain_scores), 5),
            "num_domains": len(domain_scores),
            "num_topics":  sum(s["num_topics"] for s in domain_scores),
        }
        summary["macro_average"][split] = macro
        print(f"  {split:<6}  nDCG@10 {macro['nDCG@10']:.4f}  "
              f"MAP@10 {macro['MAP@10']:.4f}  MRR {macro['MRR']:.4f}  "
              f"({macro['num_domains']} domains, {macro['num_topics']} topics)")

    sp = out_dir / "results" / "summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {sp}")


if __name__ == "__main__":
    main()
