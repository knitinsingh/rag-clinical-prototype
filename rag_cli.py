import os, glob, argparse, requests, time
from typing import List, Tuple
from dotenv import load_dotenv

import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- NEW: OpenAI SDK ----------
try:
    from openai import OpenAI  # pip install openai>=1.40.0
except Exception:
    OpenAI = None  # we'll error nicely if user selects provider=openai without SDK

# ---------------- Config ----------------
DATA_DIR        = "data"
PERSIST_DIR     = "chroma_db"
COLLECTION      = "mimic_notes"

TOP_K           = 6             # retrieve more hits
CHUNK_SIZE      = 1400          # larger chunks keep sentences intact
CHUNK_OVERLAP   = 120
MIN_CHUNK_LEN   = 120
SNIPPET_CHARS   = 400           # longer preview of each hit
EMBED_WORKERS   = 4             # parallel embedding requests (tune 4–6)

# Ollama (server on 127.0.0.1:11434)
OLLAMA_CHAT_URL    = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL       = "llama3.2"
OLLAMA_EMBED_URL   = "http://127.0.0.1:11434/api/embeddings"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# OpenAI defaults (used when --provider=openai)
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"

console = Console()

# ---------------- Utils ----------------
def read_txt_files(folder: str) -> List[Tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    docs = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read().replace("\x00", "")  # strip nulls if any
                docs.append((os.path.basename(fp), txt))
        except Exception as e:
            console.print(f"[red]Failed reading {fp}: {e}[/red]")
    return docs

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks, n, start = [], len(text), 0
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if len(chunk) >= MIN_CHUNK_LEN:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

def build_or_load_collection(rebuild: bool = False):
    # Use cosine metric so distances ~= (1 - cosine_sim)
    client = chromadb.PersistentClient(path=PERSIST_DIR, settings=Settings(allow_reset=True))
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    return col

# -------- Robust, parallel Ollama embeddings --------
def _embed_request(text: str, retries: int = 2, backoff: float = 0.8) -> List[float]:
    """
    Single embedding call with retries. Tries both 'prompt' and 'input'
    payload styles for compatibility with different Ollama versions.
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            # Try 'prompt' style first
            r = requests.post(
                OLLAMA_EMBED_URL,
                json={
                    "model": OLLAMA_EMBED_MODEL,
                    "prompt": text,
                    "options": {"keep_alive": "5m"}
                },
                timeout=300
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                return vec

            # Fallback: some Ollama builds accept 'input' instead of 'prompt'
            r = requests.post(
                OLLAMA_EMBED_URL,
                json={
                    "model": OLLAMA_EMBED_MODEL,
                    "input": text,
                    "options": {"keep_alive": "5m"}
                },
                timeout=300
            )
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec):
                return vec

            raise ValueError(f"Embedding missing/invalid in response: {data}")
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (attempt + 1))
    # After retries, raise the last exception
    raise last_exc if last_exc else RuntimeError("Unknown embedding error")

def ollama_embed(texts: List[str]) -> List[List[float]]:
    """Embed texts via Ollama in parallel, preserving order and surfacing errors."""
    if not texts:
        return []
    out: List[List[float] | None] = [None] * len(texts)

    def _task(i: int, t: str) -> None:
        out[i] = _embed_request(t)

    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as ex:
        futures = {ex.submit(_task, i, t): i for i, t in enumerate(texts)}
        done = 0
        for f in as_completed(futures):
            f.result()  # surface exceptions
            done += 1
            if done % 10 == 0 or done == len(futures):
                console.print(f"[dim]Embedded {done}/{len(futures)} chunks[/dim]")

    if any(v is None for v in out):
        missing = [i for i, v in enumerate(out) if v is None]
        raise RuntimeError(f"Embedding failed for indices: {missing}")
    return out  # type: ignore

def index_notes(collection):
    docs = read_txt_files(DATA_DIR)
    if not docs:
        console.print("[yellow]No .txt files in ./data. Add at least 3 and re-run with --rebuild.[/yellow]")
        return

    ids, texts, metas = [], [], []
    idx = 0
    for fname, text in docs:
        for ch in chunk_text(text):
            ids.append(f"{fname}:{idx}")
            texts.append(ch)
            metas.append({"source": fname})
            idx += 1

    console.print(f"[cyan]Embedding {len(texts)} chunks with Ollama (workers={EMBED_WORKERS})...[/cyan]")
    embeddings = ollama_embed(texts)

    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embeddings count {len(embeddings)} != texts count {len(texts)}")
    if not all(isinstance(vec, list) and all(isinstance(x, (int, float)) for x in vec) for vec in embeddings):
        raise RuntimeError("Embeddings are not lists of floats as expected.")

    collection.upsert(documents=texts, metadatas=metas, ids=ids, embeddings=embeddings)
    console.print(f"[green]Indexed {len(texts)} chunks from {len(docs)} files[/green]")

def retrieve(collection, query: str, k: int = TOP_K):
    qvec = ollama_embed([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]  # cosine distance = 1 - cosine_similarity
    sims = [max(0.0, 1.0 - d) for d in dists]
    return list(zip(docs, metas, dists, sims))

# ---------------- Prompts ----------------
def build_summary_prompt(context_blocks: List[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    instr = (
        "Summarize the key facts from the following clinical context in 3–5 concise bullet points. "
        "Do not invent information. If nothing material is present, answer: I don't know."
    )
    return f"{instr}\n\nContext:\n{context}\n\nSummary:"

def build_answer_prompt(context_blocks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    instr = (
        "You are a clinical assistant. Answer ONLY using the provided context. "
        "If the answer is not present, respond exactly with: I don't know. "
        "Answer in one or two concise sentences."
    )
    return f"{instr}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

# ---------------- LLM callers ----------------
def call_ollama(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.0, num_predict: int = 1024) -> str:
    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [
                {"role":"system","content":"You are a helpful clinical QA assistant. Answer only from the provided context."},
                {"role":"user","content":prompt}
            ],
            "options": {"temperature": temperature, "num_predict": num_predict, "keep_alive": "5m"},
            "stream": False
        },
        timeout=300
    )
    r.raise_for_status()
    ans = r.json().get("message", {}).get("content", "")
    return (ans.strip() if ans and ans.strip() else "I don't know.")

def call_openai(prompt: str, model: str = OPENAI_DEFAULT_MODEL, temperature: float = 0.0, max_tokens: int = 300) -> str:
    """
    Calls OpenAI Chat Completions. Requires OPENAI_API_KEY in env or .env.
    """
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai>=1.40.0")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. export OPENAI_API_KEY=sk-... or use a .env file.")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role":"system","content":"You are a helpful clinical QA assistant. Answer only from the provided context."},
            {"role":"user","content":prompt},
        ],
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    return text.strip() if text.strip() else "I don't know."

# ---------------- CLI ----------------
def main():
    load_dotenv()  # allows .env to provide OPENAI_API_KEY
    parser = argparse.ArgumentParser(description="Minimal RAG (Ollama embeddings + OpenAI/Ollama generation)")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from ./data")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name for local generation (e.g., llama3.2)")
    parser.add_argument("--provider", choices=["openai","ollama"], default="openai",
                        help="LLM provider for answer synthesis (default: openai)")
    parser.add_argument("--openai-model", default=OPENAI_DEFAULT_MODEL,
                        help=f"OpenAI model for generation (default: {OPENAI_DEFAULT_MODEL})")
    parser.add_argument("--k", type=int, default=TOP_K, help="Top-k results (default 6)")
    args = parser.parse_args()

    top_k = args.k

    col = build_or_load_collection(rebuild=args.rebuild)
    if args.rebuild:
        index_notes(col)

    console.print(Panel.fit("RAG CLI • type 'exit' to quit", style="cyan"))
    while True:
        q = Prompt.ask("[bold]Your question[/bold]")
        if q.strip().lower() in {"exit","quit"}:
            break

        hits = retrieve(col, q, k=top_k)
        if not hits:
            console.print("[yellow]No results. Did you run with --rebuild after adding notes?[/yellow]")
            continue

        # Show sources + longer snippet + similarity
        for i, (doc, meta, dist, sim) in enumerate(hits, 1):
            snippet = (doc[:SNIPPET_CHARS] + "...") if doc else ""
            console.print(f"[dim]#{i} source={meta.get('source')} | cosine_sim={sim:.3f} (dist={dist:.3f})[/dim]")
            console.print(f"[dim]{snippet}[/dim]\n")

        # Context blocks for LLM
        ctx_blocks = [doc for doc, _m, _d, _s in hits]

        # 1) Summary panel
        summary_prompt = build_summary_prompt(ctx_blocks)
        try:
            if args.provider == "openai":
                summary = call_openai(summary_prompt, model=args.openai_model, temperature=0.0, max_tokens=220)
            else:
                summary = call_ollama(summary_prompt, model=args.model, temperature=0.0, num_predict=512)
        except Exception as e:
            summary = f"(LLM error) {e}"
        console.print(Panel(summary, title="Summary of retrieved context", style="blue"))

        # 2) Direct answer panel
        answer_prompt = build_answer_prompt(ctx_blocks, q)
        try:
            if args.provider == "openai":
                ans = call_openai(answer_prompt, model=args.openai_model, temperature=0.0, max_tokens=300)
            else:
                ans = call_ollama(answer_prompt, model=args.model, temperature=0.0, num_predict=1024)
        except Exception as e:
            ans = f"(LLM error) {e}"
        console.print(Panel(ans, title="Answer", style="green"))

if __name__ == "__main__":
    main()
