# app.py — Gradio UI for GitaSphere (CPU-friendly, llama.cpp + FAISS RAG)
import os, time, re, logging, pickle, asyncio, inspect
from typing import List, Dict, Optional

import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from googletrans import Translator

# =========================
# Config via environment
# =========================
FAISS_PATH = os.environ.get("RAG_FAISS_PATH", "gita_knowledge_base.faiss")
DOCS_PATH  = os.environ.get("RAG_DOCS_PATH",  "gita_documents.pkl")
GGUF_PATH  = os.environ.get("GGUF_PATH",      "checkpoints/gita-model-q4_K_M.gguf")

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANK_MODEL     = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Reasonable CPU fanout
INITIAL_K  = int(os.environ.get("RAG_INITIAL_K", "20"))
RERANK_TOP = int(os.environ.get("RERANK_TOP",   "20"))
RETURN_K   = int(os.environ.get("RAG_RETURN_K", "5"))

# llama.cpp generation params (CPU-tuned)
N_THREADS    = int(os.environ.get("LLAMA_THREADS", str(os.cpu_count() or 4)))
N_CTX        = int(os.environ.get("LLAMA_CTX", "4096"))
N_BATCH      = int(os.environ.get("LLAMA_BATCH", "256"))
TEMPERATURE  = float(os.environ.get("LLAMA_TEMP", "0.6"))
TOP_P        = float(os.environ.get("LLAMA_TOP_P", "0.8"))
MIROSTAT     = int(os.environ.get("LLAMA_MIROSTAT", "0"))   # keep 0 unless your llama-cpp version supports it
MIROSTAT_TAU = float(os.environ.get("LLAMA_MIROSTAT_TAU", "5.0"))
MIROSTAT_ETA = float(os.environ.get("LLAMA_MIROSTAT_ETA", "0.1"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gitasphere-app")

# =========================
# Globals loaded once
# =========================
index: Optional[faiss.Index] = None
documents: List[Dict] = []
embedder: Optional[SentenceTransformer] = None
reranker: Optional[CrossEncoder] = None
llm: Optional[Llama] = None
translator: Optional[Translator] = None
SYSTEM_MSG: Optional[str] = None

# =========================
# Utilities
# =========================
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


GGUF_REPO_ID = "Harsh0304/gitasphere-llama3-gguf"
GGUF_FILENAME = "gita-model-q4_K_M.gguf"



def is_sanskrit(text: str) -> bool:
    """Heuristic: >50% non-space chars are Devanagari."""
    if not text:
        return False
    dev, tot = 0, 0
    for ch in text:
        if ch.strip():
            tot += 1
            if '\u0900' <= ch <= '\u097F':
                dev += 1
    return (tot > 0) and (dev / tot > 0.5)

def translate_to_en(text: str) -> Optional[str]:
    try:
        if not text or not text.strip():
            return None
        res = translator.translate(text, src="auto", dest="en")
        # Some environments wrap an async coroutine — guard for it:
        if inspect.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            res = loop.run_until_complete(res)
        return getattr(res, "text", None)
    except Exception as e:
        log.warning(f"Google Translate failed: {e}")
        return None

def load_kb():
    idx = faiss.read_index(FAISS_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)
    return idx, docs

def embed(texts: List[str]) -> np.ndarray:
    v = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")

def build_short_tag(meta: Dict) -> str:
    # e.g., "Śaṅkara/Gītā Bhāṣya" or "Rāmānuja/Gītā Bhāṣya"
    auth = (meta.get("author") or "").lower()
    src  = (meta.get("source_file") or "").lower()
    if "sankara" in src or "shank" in auth:
        return "Śaṅkara/Gītā Bhāṣya"
    if "ramanuja" in src or "ramanuj" in auth:
        return "Rāmānuja/Gītā Bhāṣya"
    school = meta.get("school")
    return f"{school}" if school else "Commentary"

def rag_search(query: str) -> List[Dict]:
    # 1) ANN
    qv = embed([query])
    _, I = index.search(qv, INITIAL_K)
    cands = [{"idx": i, "doc": documents[i]} for i in I[0] if i >= 0]
    if not cands:
        return []

    # 2) Rerank
    pairs = [[query, c["doc"]["content"]] for c in cands[:RERANK_TOP]]
    scores = reranker.predict(pairs, batch_size=32) if pairs else []
    for j, s in enumerate(scores):
        cands[j]["score"] = float(s)
    reranked = sorted(cands[:RERANK_TOP], key=lambda x: x.get("score", -1e9), reverse=True)

    # 3) Promote diversity by source_file
    diverse, seen = [], set()
    for c in reranked:
        src = c["doc"]["metadata"].get("source_file")
        if src not in seen:
            diverse.append(c)
            seen.add(src)
            if len(diverse) >= RETURN_K:
                break
    if len(diverse) < RETURN_K:
        for c in reranked:
            if c not in diverse:
                diverse.append(c)
                if len(diverse) >= RETURN_K:
                    break
    return diverse[:RETURN_K]

def persona_instructions(persona: str) -> str:
    p = (persona or "").lower().strip()
    if p == "school":
        return ("Audience: a 14-year-old. Short, simple, encouraging. "
                "One daily-life example. Use max 2 citations [n].")
    if p == "masters":
        return ("Audience: graduate student. Define key Sanskrit terms. "
                "Discuss 1–2 schools (e.g., Advaita, Viśiṣṭādvaita). Use 3–4 citations [n].")
    if p == "phd":
        return ("Audience: doctoral scholar. Contrast Śaṅkara vs Rāmānuja vs (optionally) Madhva; "
                "use technical Sanskrit with brief glosses. Use 5+ citations [n].")
    if p == "bhakta":
        return ("Audience: devotee. Devotional framing; examples from saints and lived practice; "
                "actionable sādhana steps. Use 2–4 citations [n].")
    return "Default graduate-level tone with 3–4 citations."

def build_system_message() -> str:
    return (
        "You are an expert Bhagavad Gītā teacher.\n"
        "Always structure the answer in exactly four sections:\n"
        "1) Importance\n"
        "2) Philosophical Interpretation (cite schools where relevant)\n"
        "3) Modern Relevance\n"
        "4) Practical Implementation\n\n"
        "Citations: When you use the provided context, cite with bracketed numbers [1]..[k] "
        "referring to the numbered context chunks. Do not invent sources. "
        "If context is insufficient, say so briefly.\n"
        "Conclusion: Invite the user to ask follow-ups about any part."
    )

def format_context_for_prompt(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["doc"]["metadata"]
        tag  = build_short_tag(meta)
        content = re.sub(r"\s+", " ", c["doc"]["content"]).strip()
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"[{i}] {tag} — {content}")
    return "\n".join(lines)

def build_prompt(query_en: str, persona: str, chunks: List[Dict], history_pairs: List) -> List[Dict]:
    ctx = format_context_for_prompt(chunks)
    user_prompt = (
        f"Persona instructions: {persona_instructions(persona)}\n\n"
        f"Question/Verse (English): {query_en}\n\n"
        f"Context chunks (use [1]..[{len(chunks)}] for citations):\n{ctx}\n\n"
        "Write the answer now."
    )
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for u, a in history_pairs:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_prompt})
    return messages

def run_llama_stream(messages: List[Dict], max_tokens: int):
    kwargs = dict(
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
        stop=["</s>", "<|eot_id|>"],
        stream=True,
    )
    if MIROSTAT in (1, 2):
        kwargs.update({"mirostat": MIROSTAT, "mirostat_tau": MIROSTAT_TAU, "mirostat_eta": MIROSTAT_ETA})

    for chunk in llm.create_chat_completion(**kwargs):
        delta = chunk["choices"][0].get("delta", {})
        if "content" in delta:
            yield delta["content"]

# =========================
# One-time startup
# =========================
def _startup_once():
    global index, documents, embedder, reranker, llm, translator, SYSTEM_MSG

    log.info("Loading FAISS & docs…")
    idx, docs = load_kb()
    index, documents = idx, docs
    log.info(f"FAISS vectors: {index.ntotal} | docs: {len(documents)}")

    log.info(f"Loading embedder… {EMBED_MODEL_NAME}")
    globals()["embedder"] = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    log.info(f"Loading reranker… {RERANK_MODEL}")
    globals()["reranker"] = CrossEncoder(RERANK_MODEL, device="cpu")

    log.info(f"Downloading llama.cpp model from Hub: {GGUF_REPO_ID}")
    
    # --- THIS IS THE KEY CHANGE ---
    # Download the model from the Hub. It will be cached locally.
    model_path = hf_hub_download(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME
    )
    
    log.info(f"Loading llama.cpp from downloaded path: {model_path}")
    os.environ.setdefault("GGML_N_THREADS", str(N_THREADS))
    
    # Load the model from the downloaded path
    globals()["llm"] = Llama(
        model_path=model_path, 
        n_ctx=N_CTX, 
        n_threads=N_THREADS, 
        n_batch=N_BATCH, 
        embedding=False, 
        verbose=False
    )

    log.info("Loading Google Translator client…")
    globals()["translator"] = Translator(service_urls=["translate.googleapis.com"])

    globals()["SYSTEM_MSG"] = build_system_message()
    log.info("Startup complete.")

_startup_once()

# =========================
# Gradio callbacks (Blocks)
# =========================
def messages_to_pairs(chat_messages: List[Dict]) -> List:
    """Convert Chatbot(type='messages') history to [(user, assistant), ...]."""
    pairs, curr_user = [], None
    for m in chat_messages or []:
        role, content = m.get("role"), m.get("content", "")
        if role == "user":
            curr_user = content
        elif role == "assistant" and curr_user is not None:
            pairs.append((curr_user, content))
            curr_user = None
    return pairs

def respond(user_msg, persona, max_tokens, chat_history):
    """
    Streaming generator that updates Chatbot(type='messages') contents.
    chat_history is a list[{"role":..., "content":...}, ...]
    """
    t0 = time.time()
    user_msg = (user_msg or "").strip()
    if not user_msg:
        yield chat_history, ""
        return

    # Append the user message
    chat_history = (chat_history or []) + [{"role": "user", "content": user_msg}]

    # Translate if needed
    translation = translate_to_en(user_msg) if is_sanskrit(user_msg) else None
    query_en = translation or user_msg

    # RAG & prompt
    chunks = rag_search(query_en)
    history_pairs = messages_to_pairs(chat_history)
    messages = build_prompt(query_en, persona, chunks, history_pairs)

    # Prepare citations
    refs_lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["doc"]["metadata"]
        tag = build_short_tag(meta)
        fname = meta.get("source_file") or "doc"
        refs_lines.append(f"- **[{i}]** {tag} — `{fname}`")
    citations_md = "\n".join(refs_lines) if refs_lines else "_No context found._"

    # Start assistant message with optional translation header
    header = f"**Translation:** *{translation}*\n\n---\n" if translation else ""
    assistant_accum = ""

    # Add empty assistant message first so UI shows streaming
    chat_history.append({"role": "assistant", "content": header})
    yield chat_history, f"⌛ Generating…"

    # Stream tokens
    for tok in run_llama_stream(messages, max_tokens):
        assistant_accum += tok
        chat_history[-1]["content"] = header + assistant_accum
        yield chat_history, f"{time.time() - t0:.2f}s"

    # Finalize with references + elapsed time
    chat_history[-1]["content"] = (
        header + assistant_accum +
        "\n\n---\n### References\n" + citations_md +
        f"\n\n*Response generated in {time.time() - t0:.2f}s.*"
    )
    yield chat_history, f"{time.time() - t0:.2f}s"

def clear_chat():
    return [], ""

# =========================
# Gradio UI (no ChatInterface)
# =========================
with gr.Blocks(title="GitaSphere AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "## 🕉️ GitaSphere AI\n"
        "Paste a Sanskrit verse or ask an English question. Choose a persona.\n"
        "The app will (1) translate if Sanskrit (Google), (2) retrieve diverse commentary (RAG),\n"
        "and (3) generate a structured answer with inline citations [n]."
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            persona = gr.Dropdown(
                choices=["school", "masters", "phd", "bhakta"],
                value="masters",
                label="Persona / Depth",
            )
            max_tokens = gr.Slider(
                minimum=128, maximum=1024, value=512, step=32,
                label="Max Response Length"
            )
            clear_btn = gr.Button("🧹 Clear History")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="GitaSphere Conversation",
                show_label=False,
                height=600,
                type="messages",      # <- modern format to silence the warning
            )
            user_box = gr.Textbox(
                placeholder="Paste a verse or ask a question…",
                lines=4,
                label="Your message",
            )
            submit_btn = gr.Button("✨ Ask GitaSphere", variant="primary")

    # Hidden state not needed when using type='messages'; Chatbot stores messages.
    # But we’ll return elapsed time for convenience:
    elapsed = gr.Label(label="Elapsed", value="")

    # Wire events
    submit_btn.click(
        fn=respond,
        inputs=[user_box, persona, max_tokens, chatbot],
        outputs=[chatbot, elapsed],
        queue=True,
    )

    # Also allow pressing Enter in the textbox
    user_box.submit(
        fn=respond,
        inputs=[user_box, persona, max_tokens, chatbot],
        outputs=[chatbot, elapsed],
        queue=True,
    )

    clear_btn.click(fn=clear_chat, inputs=[], outputs=[chatbot, elapsed])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0",share=True, server_port=int(os.environ.get("PORT", "7860")))

