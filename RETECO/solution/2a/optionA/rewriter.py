"""
rewriter.py — LLM query rewriting for RETECO Track 2a.

Uses SAP AI Core Orchestration SDK (gen_ai_hub) instead of OpenAI directly.
Credentials are auto-loaded from the canary-eu10.yml file or from .env.

File-based cache: output/rewrites/{domain}/{split}/{qid}.txt — skipped on re-run.
Falls back to query+history concatenation if API call fails.

Supported strategies:
  "qr"  — QueryRewrite: rewrite current turn into standalone query (original)
  "hr"  — History+Reasoning: generate CoT reasoning + search query (best per RECOR paper)
"""

import asyncio
import os
import re
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

# ── credential loading ────────────────────────────────────────────────────────

_YML_CREDENTIAL_FILE = Path.home() / "Documents/Work/Multi-Model-Demo 1/environments/canary-eu10.yml"
_ENV_FILE = Path(__file__).parent.parent / ".env"

def _load_credentials() -> None:
    """Load AICORE_* env vars from yml file, then overlay with .env if present."""
    # 1. parse yml
    if _YML_CREDENTIAL_FILE.exists():
        with open(_YML_CREDENTIAL_FILE, encoding="utf-8") as f:
            yml = yaml.safe_load(f)
        mapping = {v["name"]: v.get("value", "") for v in yml.get("variables", []) if "value" in v}
        _set_if_missing("AICORE_BASE_URL",      mapping.get("AI_API_URL", ""))
        _set_if_missing("AICORE_AUTH_URL",       mapping.get("AUTH_URL", ""))
        _set_if_missing("AICORE_CLIENT_ID",      mapping.get("CLIENT_ID", ""))
        _set_if_missing("AICORE_CLIENT_SECRET",  mapping.get("CLIENT_SECRET", ""))
        _set_if_missing("AICORE_RESOURCE_GROUP", mapping.get("RESOURCE_GROUP", "default"))
        orch_url = mapping.get("ORCH_DEPLOYMENT_URL", "")
        if orch_url:
            _set_if_missing("AICORE_ORCHESTRATION_DEPLOYMENT_URL", orch_url)

    # 2. .env can override
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE, override=False)


def _set_if_missing(key: str, value: str) -> None:
    if not os.environ.get(key) and value:
        os.environ[key] = value


# Load credentials immediately on import
_load_credentials()


# ── prompts ───────────────────────────────────────────────────────────────────

# --- QueryRewrite (QR) strategy ---

QR_SYSTEM_PROMPT = """You are a search query optimizer for a conversational retrieval system.

Your task: given a multi-turn conversation and the CURRENT turn's question, rewrite the current question into a single, self-contained search query that:
1. Includes all necessary context from the conversation history
2. Resolves pronouns and references (e.g. "that", "those", "it", "they")
3. Is specific enough to retrieve the right documents
4. Is concise — aim for 1-2 sentences, max 60 words
5. Reads as a standalone question, NOT as part of a conversation

Output ONLY the rewritten query. No explanation, no preamble, no quotes."""

QR_USER_TEMPLATE = """Conversation history:
{history}

Current question: {query}

Rewrite the current question into a standalone search query:"""

# --- History+Reasoning (H+R) strategy (best per RECOR paper Table 3) ---

HR_SYSTEM_PROMPT = """You are an expert search assistant for a reasoning-intensive retrieval system.

Documents in this system are retrieved based on reasoning chains, not keyword matching. Your task: given a multi-turn conversation and the current question, generate a step-by-step reasoning chain about what information is needed, then formulate a search query.

Output format (use exactly these labels):
Reasoning: <2-4 sentences explaining what specific information or concepts are needed to answer the question, considering the conversation context>
Search query: <a concise standalone search query, 1-2 sentences, max 60 words>"""

HR_USER_TEMPLATE_WITH_HISTORY = """Conversation history:
{history}

Current question: {query}

First reason about what information is needed, then formulate a search query:"""

HR_USER_TEMPLATE_NO_HISTORY = """Current question: {query}

First reason about what specific information or concepts are needed to answer this question, then formulate a search query:"""


# ── cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(cache_dir: Path, qid: str) -> Path:
    safe = re.sub(r"[^\w\-]", "_", qid)
    return cache_dir / f"{safe}.txt"


def _fallback(query: str, history: str) -> str:
    if not history or history.lower().startswith("no previous"):
        return query
    return f"{query}\n\nConversation History:\n{history}"


# ── SAP AI Core call (sync, wrapped in thread for async) ──────────────────────

def _sap_rewrite_sync(query: str, history: str, model: str, strategy: str = "qr", max_tokens: int = 300) -> str:
    from gen_ai_hub.orchestration.service import OrchestrationService
    from gen_ai_hub.orchestration.models.config import OrchestrationConfig
    from gen_ai_hub.orchestration.models.llm import LLM
    from gen_ai_hub.orchestration.models.template import Template
    from gen_ai_hub.orchestration.models.message import SystemMessage, UserMessage

    has_history = bool(history) and not history.lower().startswith("no previous")

    if strategy == "hr":
        system_msg = HR_SYSTEM_PROMPT
        if has_history:
            user_msg = HR_USER_TEMPLATE_WITH_HISTORY.format(history=history.strip(), query=query.strip())
        else:
            user_msg = HR_USER_TEMPLATE_NO_HISTORY.format(query=query.strip())
    else:
        system_msg = QR_SYSTEM_PROMPT
        user_msg = QR_USER_TEMPLATE.format(history=history.strip(), query=query.strip())

    service = OrchestrationService()
    config = OrchestrationConfig(
        template=Template(messages=[
            SystemMessage(system_msg),
            UserMessage(user_msg),
        ]),
        llm=LLM(name=model, parameters={
            "max_completion_tokens": max_tokens,
            "temperature": 0.0,
        }),
    )
    resp = service.run(config=config)
    raw = resp.orchestration_result.choices[0].message.content.strip()

    if strategy == "hr":
        return _extract_search_query(raw)
    return raw


def _extract_search_query(text: str) -> str:
    """Extract the 'Search query:' line from H+R LLM output."""
    for line in text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("search query:"):
            return stripped[len("search query:"):].strip()
    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


# ── per-turn rewrite ──────────────────────────────────────────────────────────

async def rewrite_turn(
    query: str,
    history: str,
    qid: str,
    cache_dir: Path,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o-mini",
    strategy: str = "qr",
) -> str:
    cache_file = _cache_path(cache_dir, qid)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8").strip()

    has_history = bool(history) and not history.lower().startswith("no previous")

    # Turn 1 (no history) with QR strategy: LLM call not needed
    if not has_history and strategy == "qr":
        result = query
        cache_file.write_text(result, encoding="utf-8")
        return result

    async with semaphore:
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(_sap_rewrite_sync, query, history, model, strategy),
                timeout=60.0,
            )
        except Exception as e:
            print(f"  [WARN] rewrite failed for {qid}: {e}")
            result = _fallback(query, history)

    cache_file.write_text(result, encoding="utf-8")
    return result


# ── domain-level batch rewrite ────────────────────────────────────────────────

async def rewrite_domain(
    domain_dir: Path,
    split: str,
    out_dir: Path,
    concurrency: int = 4,
    model: str = "gpt-4o-mini",
    strategy: str = "qr",
) -> dict[str, str]:
    """Rewrite all turns for one domain/split. Returns {qid: rewritten_query}."""
    bench_file = domain_dir / f"benchmark_{split}.json"
    if not bench_file.exists():
        return {}

    cache_dir = out_dir / "rewrites" / strategy / domain_dir.name / split
    cache_dir.mkdir(parents=True, exist_ok=True)

    conversations = json.loads(bench_file.read_text(encoding="utf-8"))
    semaphore = asyncio.Semaphore(concurrency)

    tasks: dict[str, asyncio.coroutine] = {}
    for conv in conversations:
        for turn in conv["turns"]:
            qid = f"{conv['id']}_turn_{turn['turn_id']}"
            tasks[qid] = rewrite_turn(
                turn["query"],
                turn.get("conversation_history", ""),
                qid,
                cache_dir,
                semaphore,
                model=model,
                strategy=strategy,
            )

    qids = list(tasks.keys())
    rewrites = await asyncio.gather(*tasks.values())
    return dict(zip(qids, rewrites))


def rewrite_domain_sync(
    domain_dir: Path,
    split: str,
    out_dir: Path,
    concurrency: int = 4,
    model: str = "gpt-4o-mini",
    strategy: str = "qr",
) -> dict[str, str]:
    return asyncio.run(rewrite_domain(domain_dir, split, out_dir, concurrency, model, strategy))
