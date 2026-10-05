#!/usr/bin/env python3
"""
run_2a.py — RETECO Track 2a pipeline: LLM query rewriting + BM25 retrieval.

Usage:
    # Full run (all 11 domains, dev split):
    python run_2a.py --data ../../../../reteco_data --splits dev

    # Train + dev:
    python run_2a.py --data ../../../../reteco_data --splits train dev

    # Single domain (fast smoke test, no API cost):
    python run_2a.py --data ../../../../reteco_data --splits dev --domains drones

    # Skip rewriting, use raw query + history (reproduces BM25+hist baseline):
    python run_2a.py --data ../../../../reteco_data --splits dev --no-rewrite

Results written to:
    output/runs/run_<tag>_<domain>_<split>.trec   — TREC submission file
    output/results/summary.json                   — nDCG@10 per domain + macro-average

Requires:
    OPENAI_API_KEY env var (unless --no-rewrite)
    pip install -r requirements.txt
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import pytrec_eval
from tqdm import tqdm

from retriever import BM25Index, load_qrels, run_retrieval, write_run
from rewriter import rewrite_domain_sync


TRACK2_DOMAINS = [
    "biology", "drones", "earth_science", "economics", "hardware",
    "law", "medicalsciences", "politics", "psychology", "robotics",
    "sustainable_living",
]


# ── query builders ──────────────────────────────────────────────────────────

def build_queries_history(domain_dir: Path, split: str) -> dict[str, str]:
    """Baseline+: concatenate query + history (replicates official +hist baseline)."""
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


def build_queries_rewritten(
    domain_dir: Path, split: str, out_dir: Path, model: str
) -> dict[str, str]:
    """LLM rewriting: decontextualized self-contained queries."""
    return rewrite_domain_sync(domain_dir, split, out_dir, model=model)


# ── scoring ─────────────────────────────────────────────────────────────────

def score_run(
    run: dict[str, list[tuple[str, float]]],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> dict:
    if not run or not qrels:
        return {}

    qrels_dict = {qid: {d: 1 for d in docs} for qid, docs in qrels.items()}
    run_dict = {
        qid: {docid: score for docid, score in ranked}
        for qid, ranked in run.items()
        if qid in qrels_dict
    }

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_dict, {f"ndcg_cut.{k}", f"map_cut.{k}", "recip_rank"}
    )
    scores = evaluator.evaluate(run_dict)
    if not scores:
        return {}

    n = len(scores)
    ndcg = sum(v[f"ndcg_cut_{k}"] for v in scores.values()) / n
    mmap  = sum(v[f"map_cut_{k}"]  for v in scores.values()) / n
    mrr   = sum(v["recip_rank"]    for v in scores.values()) / n
    return {
        f"nDCG@{k}": round(ndcg, 5),
        f"MAP@{k}":  round(mmap, 5),
        "MRR":       round(mrr, 5),
        "num_topics": n,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--data",    default=str(here / ".." / ".." / ".." / ".." / "reteco_data"))
    ap.add_argument("--out",     default=str(here / "output"))
    ap.add_argument("--splits",  nargs="+", default=["dev"])
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--model",   default="gpt-4o-mini",
                    help="OpenAI model for rewriting (default: gpt-4o-mini)")
    ap.add_argument("--no-rewrite", action="store_true",
                    help="Use raw query+history instead of LLM rewriting")
    ap.add_argument("--top-k",   type=int, default=100)
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    out_dir  = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = args.domains or TRACK2_DOMAINS
    tag = "bm25_hist" if args.no_rewrite else f"bm25_rewrite_{args.model}"

    all_results: dict[str, dict] = defaultdict(dict)

    for domain in domains:
        domain_dir = data_dir / "track2_recor" / domain
        if not domain_dir.exists():
            print(f"[SKIP] {domain} — directory not found")
            continue

        print(f"\n=== {domain} ===")
        print(f"  building BM25 index...", flush=True)
        index = BM25Index(domain_dir)

        for split in args.splits:
            bench_file = domain_dir / f"benchmark_{split}.json"
            if not bench_file.exists():
                continue

            qrels = load_qrels(domain_dir, split)
            if not qrels:
                print(f"  [{split}] no qrels, skipping")
                continue

            if args.no_rewrite:
                queries = build_queries_history(domain_dir, split)
            else:
                print(f"  [{split}] rewriting queries with {args.model}...", flush=True)
                queries = build_queries_rewritten(domain_dir, split, out_dir, args.model)

            queries = {q: v for q, v in queries.items() if q in qrels}

            run = run_retrieval(index, queries, top_k=args.top_k)

            run_path = out_dir / "runs" / f"run_{tag}_{domain}_{split}.trec"
            write_run(run, run_path, tag=tag)

            metrics = score_run(run, qrels)
            all_results[domain][split] = metrics

            print(f"  [{split}] nDCG@10 {metrics.get('nDCG@10', 0):.4f}  "
                  f"MAP@10 {metrics.get('MAP@10', 0):.4f}  "
                  f"MRR {metrics.get('MRR', 0):.4f}  "
                  f"({metrics.get('num_topics', 0)} topics)")

    # ── macro-average ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"MACRO-AVERAGE nDCG@10  [{tag}]")
    print("=" * 60)

    summary: dict = {"tag": tag, "per_domain": dict(all_results), "macro_average": {}}

    for split in args.splits:
        domain_scores = [
            all_results[d][split]
            for d in domains
            if split in all_results.get(d, {})
        ]
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

    summary_path = out_dir / "results" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
