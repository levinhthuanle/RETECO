#!/usr/bin/env python3
"""
run_2a.py — RETECO Track 2a · Option B: Dense retrieval.

Models:
  intfloat/e5-base-v2             (default, 109M)
  AQ-MedAI/Diver-Retriever-4B-1020  (4B, BRIGHT SOTA retriever)
  reasonir/ReasonIR-8B            (8B, Meta AI reasoning retriever)

Query strategies (--query-strategy):
  current   current turn only
  history   raw query + conversation history concatenated
  qr        LLM QueryRewrite from optionA cache
  hr        LLM History+Reasoning from optionA cache (best per RECOR paper)

Usage:
    # DIVER + H+R (Phase 3 target):
    python run_2a.py --data ../../../../reteco_data --splits dev \\
        --model AQ-MedAI/Diver-Retriever-4B-1020 --query-strategy hr

    # e5-base-v2 + history (old default):
    python run_2a.py --data ../../../../reteco_data --splits dev \\
        --model intfloat/e5-base-v2 --query-strategy history

    # Smoke test single domain:
    python run_2a.py --data ../../../../reteco_data --splits dev \\
        --domains drones --model AQ-MedAI/Diver-Retriever-4B-1020 --query-strategy hr

Results:
    output/runs/run_<model_slug>_<query_strategy>_<domain>_<split>.trec
    output/results/summary.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from embedder import build_index, search  # faiss imported here; must precede pytrec_eval

import pytrec_eval
from tqdm import tqdm


TRACK2_DOMAINS = [
    "biology", "drones", "earth_science", "economics", "hardware",
    "law", "medicalsciences", "politics", "psychology", "robotics",
    "sustainable_living",
]

OPTIONA_REWRITES = Path(__file__).parent.parent / "optionA" / "output" / "rewrites"


# ── query builders ─────────────────────────────────────────────────────────

def build_queries_current(domain_dir: Path, split: str) -> dict[str, str]:
    bench = json.loads((domain_dir / f"benchmark_{split}.json").read_text())
    out = {}
    for conv in bench:
        for turn in conv["turns"]:
            qid = f"{conv['id']}_turn_{turn['turn_id']}"
            out[qid] = turn["query"]
    return out


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


def _load_rewrite_cache(domain_dir: Path, split: str, strategy: str) -> dict[str, str]:
    """Load LLM rewrites cached by optionA (strategy = 'qr' or 'hr')."""
    cache_dir = OPTIONA_REWRITES / strategy / domain_dir.name / split
    if not cache_dir.exists():
        # Try old-style path (pre-strategy subdirectory, only qr existed)
        legacy_dir = OPTIONA_REWRITES / domain_dir.name / split
        if legacy_dir.exists() and strategy == "qr":
            cache_dir = legacy_dir
        else:
            return {}

    bench = json.loads((domain_dir / f"benchmark_{split}.json").read_text())
    out = {}
    for conv in bench:
        for turn in conv["turns"]:
            qid = f"{conv['id']}_turn_{turn['turn_id']}"
            fname = re.sub(r"[^\w\-]", "_", qid) + ".txt"
            cache_file = cache_dir / fname
            if cache_file.exists():
                out[qid] = cache_file.read_text(encoding="utf-8").strip()
    return out


def build_queries_rewrite(domain_dir: Path, split: str, strategy: str) -> dict[str, str]:
    """Load LLM rewrites; fall back to history for missing turns."""
    cached = _load_rewrite_cache(domain_dir, split, strategy)
    if not cached:
        print(f"  [WARN] no {strategy} cache for {domain_dir.name}/{split}, falling back to history")
        return build_queries_history(domain_dir, split)

    # Fill any missing qids with history fallback
    hist_queries = build_queries_history(domain_dir, split)
    missing = [qid for qid in hist_queries if qid not in cached]
    if missing:
        print(f"  [WARN] {len(missing)} turns missing from {strategy} cache, using history fallback")
    for qid in missing:
        cached[qid] = hist_queries[qid]
    return cached


# ── scoring ────────────────────────────────────────────────────────────────

def load_qrels(domain_dir: Path, split: str) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    f = domain_dir / f"qrels_{split}.txt"
    if not f.exists():
        return qrels
    for line in f.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            qid, _, docid, rel = parts
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(docid)
    return qrels


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


def write_run(run: dict, path: Path, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, ranked in run.items():
            for rank, (docid, score) in enumerate(ranked, 1):
                f.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--data",           default=str(here / ".." / ".." / ".." / ".." / "reteco_data"))
    ap.add_argument("--out",            default=str(here / "output"))
    ap.add_argument("--splits",         nargs="+", default=["dev"])
    ap.add_argument("--domains",        nargs="*", default=None)
    ap.add_argument("--model",          default="intfloat/e5-base-v2",
                    help="Embedding model (e.g. AQ-MedAI/Diver-Retriever-4B-1020)")
    ap.add_argument("--query-strategy", default="hr",
                    choices=["current", "history", "qr", "hr"],
                    help="Query strategy: current/history/qr/hr (default: hr)")
    ap.add_argument("--top-k",          type=int, default=100)
    ap.add_argument("--batch",          type=int, default=32,
                    help="Encoding batch size (use 32 for 4B+ models, 256 for small models)")
    args = ap.parse_args()

    data_dir = Path(args.data).resolve()
    out_dir  = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = args.domains or TRACK2_DOMAINS
    model_slug = re.sub(r"[^\w\-]", "_", args.model.split("/")[-1])
    tag = f"dense_{model_slug}_{args.query_strategy}"

    # preload model once before any domain loop to avoid MPS re-init crash
    from embedder import _get_model
    _get_model(args.model)

    all_results: dict[str, dict] = defaultdict(dict)

    for domain in domains:
        domain_dir = data_dir / "track2_recor" / domain
        if not domain_dir.exists():
            print(f"[SKIP] {domain}")
            continue

        print(f"\n=== {domain} ===")
        index_cache = out_dir / "index"
        index, doc_ids = build_index(
            domain_dir, index_cache,
            model_name=args.model,
            batch_size=args.batch,
        )

        for split in args.splits:
            if not (domain_dir / f"benchmark_{split}.json").exists():
                continue
            qrels = load_qrels(domain_dir, split)
            if not qrels:
                continue

            qs = args.query_strategy
            if qs in ("qr", "hr"):
                queries = build_queries_rewrite(domain_dir, split, qs)
            elif qs == "history":
                queries = build_queries_history(domain_dir, split)
            else:
                queries = build_queries_current(domain_dir, split)

            queries = {q: v for q, v in queries.items() if q in qrels}

            print(f"  [{split}] searching {len(queries)} queries ({qs})...", flush=True)
            run = search(index, doc_ids, queries,
                         model_name=args.model, top_k=args.top_k, batch_size=args.batch)

            run_path = out_dir / "runs" / f"run_{tag}_{domain}_{split}.trec"
            write_run(run, run_path, tag)

            metrics = score_run(run, qrels)
            all_results[domain][split] = metrics
            print(f"  [{split}] nDCG@10 {metrics.get('nDCG@10',0):.4f}  "
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
