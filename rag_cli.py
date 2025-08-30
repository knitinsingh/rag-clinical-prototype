#!/usr/bin/env python3
"""
Minimal RAG CLI:
- Index .txt files from ./data into ChromaDB using Ollama embeddings.
- Retrieve top-k chunks and ask an LLM (OpenAI or Ollama) to summarize/answer.

Notes:
- Keep responses grounded in retrieved context.
- Exit with Ctrl-C or by typing 'exit' in the prompt.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

import chromadb
from chromadb.config import Settings

try:
    # pip install openai>=1.40.0
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # Only needed if --provider=openai

# ----------------------------- Logging & Console -----------------------------

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# --------------------------------- Config -----------------------------------

@dataclass(frozen=True)
class AppCfg:
    data_dir: Path = Path("data")
    persist_dir: Path = Path("chroma_db")
    collection: str = "mimic_notes"

    top_k: int = 6
    chunk_size: int = 1400
    chunk_overlap: int = 120
    min_chunk_len: int = 120
    snippet_chars: int = 400
    embed_workers: int = 4

    # HTTP defaults
    http_timeout_s: int = 60
    keep_alive: str = "5m"

    # Ollama
    ollama_chat_url: str = "http://127.0.0.1:11434/api/chat"
    ollama_model: str = "llama3.2"
    ollama_embed_url: str = "http://127.0.0.1:11434/api/embeddings"
    ollama_embed_model: str = "nomic-embed-text"

    # OpenAI
    openai_default_model: str = "gpt-4o-mini"


CFG = AppCfg()

# --------------------------------- Utils ------------------------------------

def read_txt_files(folder: Path) -> List[Tuple[str, str]]:
    """Read all .txt files from a folder as (filename, text)."""
    docs: List[Tuple[str, str]] = []
    for fp in sorted(folder.glob("*.txt")):
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore").replace("\x00", "")
            docs.append((fp.name, text))
        except Exception as e:  # pragma: no cover
            logging.warning("Failed reading %s: %s", fp, e)
    return docs


def chunk_text(text: str, size: int, overlap: int, min_len: int) -> List[str]:
    """Simple sliding-window chunking with overlap."""
    out, n, start = [], len(text), 0
    while start < n:
        end = min(start + size, n)
        ch = text[start:end].strip()
        if len(ch) >= min_len:
            out.append(ch)
        if end == n:
            break
        start = max(0, end - overlap)
    return out


def build_or_load_collection(rebuild: bool = False):
    """Create or load a Chroma collection."""
    client = chromadb.PersistentClient(
        path=str(CFG.persist_dir),
        settings=Settings(allow_reset=True),
    )
    if rebuild:
        try:
            client.delete_collection(CFG.collection)
        except Exception:
            pass
    try:
        return client.get_collection(CFG.collection)
    except Exception:
        return client.create_collection(CFG.collection, metadata={"hnsw:space": "cosine"})


# ---------------------------- HTTP / Embeddings -----------------------------

_session = requests.Session()
_session.headers.update({"User-Agent": "rag-cli/0.1"})

def _post_json(url: str, payload: dict, timeout_s: int) -> dict:
    r = _session.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def _embed_request_ollama(text: str, retries: int = 2, backoff: float = 0.8) -> List[float]:
    """
    Single embedding call with light retries.
    Tries both 'prompt' and 'input' payloads for compatibility across Ollama versions.
    """
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            # 'prompt' style
            data = _post_json(
                CFG.ollama_embed_url,
                {
                    "model": CFG.ollama_embed_model,
                    "prompt": text,
                    "options": {"keep_alive": CFG.keep_alive},
                },
                timeout_s=CFG.http_timeout_s,
            )
            vec = data.get("embedding")
            if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                return vec

            # fallback: 'input' style
            data = _post_json(
                CFG.ollama_embed_url,
                {
                    "model": CFG.ollama_embed_model,
                    "input": text,
                    "options": {"keep_alive": CFG.keep_alive},
                },
                timeout_s=CFG.http_timeout_s,
            )
            vec = data.get("embedding")
            if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                return vec

            raise ValueError("Embedding missing in response")
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    raise last_exc or RuntimeError("Embedding failed")


def ollama_embed(texts: List[str]) -> List[List[float]]:
    """Embed texts concurrently, preserving order."""
    if not texts:
        return []
    out: List[List[float] | None] = [None] * len(texts)

    def _task(i: int, t: str) -> None:
        out[i] = _embed_request_ollama(t)

    with ThreadPoolExecutor(max_workers=CFG.embed_workers) as ex:
        futures = {ex.submit(_task, i, t): i for i, t in enumerate(texts)}
        done = 0
        for f in as_completed(futures):
            f.result()  # surface exceptions promptly
            done += 1
            if done % 10 == 0 or done == len(futures):
                logging.info("Embedded %d/%d chunks", done, len(futures))

    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        raise RuntimeError(f"Embedding failed for indices: {missing}")
    return out  # type: ignore


# ---------------------------- Index / Retrieve ------------------------------

def index_notes(collection) -> None:
    if not CFG.data_dir.exists():
        console.print(f"[yellow]{CFG.data_dir} not found. Add .txt files and re-run with --rebuild.[/yellow]")
        return

    docs = read_txt_files(CFG.data_dir)
    if not docs:
        console.print("[yellow]No .txt files in ./data. Add at least 3 and re-run with --rebuild.[/yellow]")
        return

    ids, texts, metas = [], [], []
    idx = 0
    for fname, text in docs:
        for ch in chunk_text(text, CFG.chunk_size, CFG.chunk_overlap, CFG.min_chunk_len):
            ids.append(f"{fname}:{idx}")
            texts.append(ch)
            metas.append({"source": fname})
            idx += 1

    console.print(f"[cyan]Embedding {len(texts)} chunks via Ollama (workers={CFG.embed_workers})...[/cyan]")
    embeddings = ollama_embed(texts)

    if len(embeddings) != len(texts):
        raise RuntimeError("Mismatch: embeddings vs texts")

    collection.upsert(documents=texts, metadatas=metas, ids=ids, embeddings=embeddings)
    console.print(f"[green]Indexed {len(texts)} chunks from {len(docs)} files[/green]")


def retrieve(collection, query: str, k: int) -> List[Tuple[str, dict, float, float]]:
    qvec = ollama_embed([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]  # cosine distance = 1 - cosine_similarity
    sims = [max(0.0, 1.0 - d) for d in dists]
    return list(zip(docs, metas, dists, sims))


# ------------------------------ Prompt Builders -----------------------------

def build_summary_prompt(context_blocks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    instr = (
        "Summarize key facts from the context in 3–5 bullet points. "
        "If there is no clear information to summarize, reply: I don't know."
    )
    return f"{instr}\n\nContext:\n{context}\n\nSummary:"


def build_answer_prompt(context_blocks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    instr = (
        "Answer only using the context. "
        "If the answer is absent, reply exactly: I don't know. "
        "Keep it to one or two short sentences."
    )
    return f"{instr}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


# ------------------------------- LLM Callers --------------------------------

def call_ollama(prompt: str, model: str, temperature: float = 0.0, num_predict: int = 1024) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Stick to the provided context."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature, "num_predict": num_predict, "keep_alive": CFG.keep_alive},
        "stream": False,
    }
    data = _post_json(CFG.ollama_chat_url, payload, timeout_s=CFG.http_timeout_s)
    text = (data.get("message") or {}).get("content") or ""
    return text.strip() or "I don't know."


def call_openai(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 300) -> str:
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai>=1.40.0")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "Stick to the provided context."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    return text or "I don't know."


# ----------------------------------- CLI ------------------------------------

def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser(description="RAG CLI (Ollama embeddings + OpenAI/Ollama generation)")
    p.add_argument("--rebuild", action="store_true", help="(Re)index from ./data")
    p.add_argument("--model", default=CFG.ollama_model, help="Ollama model for local generation")
    p.add_argument("--provider", choices=["openai", "ollama"], default="openai", help="LLM provider (default: openai)")
    p.add_argument("--openai-model", default=CFG.openai_default_model, help=f"OpenAI model (default: {CFG.openai_default_model})")
    p.add_argument("--k", type=int, default=CFG.top_k, help=f"Top-k results (default {CFG.top_k})")
    args = p.parse_args()

    col = build_or_load_collection(rebuild=args.rebuild)
    if args.rebuild:
        index_notes(col)

    console.print(Panel.fit("RAG CLI — type 'exit' to quit", style="cyan"))

    while True:
        try:
            q = Prompt.ask("[bold]Your question[/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/dim]")
            return

        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        hits = retrieve(col, q, k=args.k)
        if not hits:
            console.print("[yellow]No results. Try --rebuild after adding notes to ./data[/yellow]")
            continue

        # Show sources + snippet + similarity
        for i, (doc, meta, dist, sim) in enumerate(hits, 1):
            snippet = (doc[:CFG.snippet_chars] + "...") if doc else ""
            console.print(f"[dim]#{i} source={meta.get('source')} | cosine_sim={sim:.3f} (dist={dist:.3f})[/dim]")
            console.print(f"[dim]{snippet}[/dim]\n")

        ctx_blocks = [doc for doc, _m, _d, _s in hits]

        # Summary
        try:
            sp = build_summary_prompt(ctx_blocks)
            if args.provider == "openai":
                summary = call_openai(sp, model=args.openai_model, temperature=0.0, max_tokens=220)
            else:
                summary = call_ollama(sp, model=args.model, temperature=0.0, num_predict=512)
        except Exception as e:
            summary = f"(LLM error) {e}"
        console.print(Panel(summary, title="Summary", style="blue"))

        # Direct answer
        try:
            ap = build_answer_prompt(ctx_blocks, q)
            if args.provider == "openai":
                ans = call_openai(ap, model=args.openai_model, temperature=0.0, max_tokens=300)
            else:
                ans = call_ollama(ap, model=args.model, temperature=0.0, num_predict=1024)
        except Exception as e:
            ans = f"(LLM error) {e}"
        console.print(Panel(ans, title="Answer", style="green"))


if __name__ == "__main__":
    main()
