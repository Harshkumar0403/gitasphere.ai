# app.py — Gradio UI for GitaSphere (CPU/GPU, transformers + FAISS RAG)
import os, time, re, logging, pickle, asyncio, inspect
from typing import List, Dict, Optional
from threading import Thread # --- NEW ---

import numpy as np
import faiss
import gradio as gr
import torch # --- NEW ---
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download
# from llama_cpp import Llama # --- REMOVED ---
from transformers import pipeline, AutoTokenizer, TextIteratorStreamer # --- NEW ---
from googletrans import Translator

# =========================
# Config via environment
# =========================
# --- NEW: Point to a standard transformers model repository ---
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "Harsh0304/llama-3.2-3b-gita-sft")
KB_REPO_ID = os.getenv("KB_REPO_ID", "Harsh0304/gitasphere-ai-knowledge-base")

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANK_MODEL     = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- RAG params are unchanged ---
INITIAL_K  = int(os.environ.get("RAG_INITIAL_K", "20"))
RERANK_TOP = int(os.environ.get("RERANK_TOP",    "20"))
RETURN_K   = int(os.environ.get("RAG_RETURN_K", "5"))

# --- MODIFIED: Generation params for transformers ---
DEVICE       = os.environ.get("DEVICE", "cpu") # "cpu" or "cuda"
TORCH_DTYPE  = torch.bfloat16 if DEVICE == "cuda" and torch.cuda.is_bf16_supported() else "auto"
TEMPERATURE  = float(os.environ.get("LLAMA_TEMP", "0.6"))
TOP_P        = float(os.environ.get("LLAMA_TOP_P", "0.8"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gitasphere-app")

# =========================
# Globals loaded once
# =========================
index: Optional[faiss.Index] = None
documents: List[Dict] = []
embedder: Optional[SentenceTransformer] = None
reranker: Optional[CrossEncoder] = None
translator: Optional[Translator] = None
SYSTEM_MSG: Optional[str] = None

# --- MODIFIED: Globals for transformers model ---
llm_pipeline: Optional[pipeline] = None
tokenizer: Optional[AutoTokenizer] = None
streamer: Optional[TextIteratorStreamer] = None

# =========================
# Utilities (largely unchanged)
# =========================
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')

def is_sanskrit(text: str) -> bool:
    if not text: return False
    dev, tot = 0, 0
    for ch in text:
        if ch.strip():
            tot += 1
            if '\u0900' <= ch <= '\u097F':
                dev += 1
    return (tot > 0) and (dev / tot > 0.5)

def translate_to_en(text: str) -> Optional[str]:
    try:
        if not text or not text.strip(): return None
        res = translator.translate(text, src="auto", dest="en")
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

def load_kb_from_hub():
    """Downloads and loads the FAISS index and documents from the HF Hub."""
    log.info(f"Downloading knowledge base from Hub repo: {KB_REPO_ID}")
    faiss_path = hf_hub_download(repo_id=KB_REPO_ID, filename="gita_knowledge_base.faiss", repo_type="dataset")
    docs_path = hf_hub_download(repo_id=KB_REPO_ID, filename="gita_documents.pkl", repo_type="dataset")
    
    idx = faiss.read_index(faiss_path)
    with open(docs_path, "rb") as f:
        docs = pickle.load(f)
    return idx, docs

def embed(texts: List[str]) -> np.ndarray:
    v = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return v.astype("float32")

def build_short_tag(meta: Dict) -> str:
    auth = (meta.get("author") or "").lower()
    src  = (meta.get("source_file") or "").lower()
    if "sankara" in src or "shank" in auth: return "Śaṅkara/Gītā Bhāṣya"
    if "ramanuja" in src or "ramanuj" in auth: return "Rāmānuja/Gītā Bhāṣya"
    school = meta.get("school")
    return f"{school}" if school else "Commentary"

def rag_search(query: str) -> List[Dict]:
    qv = embed([query])
    _, I = index.search(qv, INITIAL_K)
    cands = [{"idx": i, "doc": documents[i]} for i in I[0] if i >= 0]
    if not cands: return []
    pairs = [[query, c["doc"]["content"]] for c in cands[:RERANK_TOP]]
    scores = reranker.predict(pairs, batch_size=32) if pairs else []
    for j, s in enumerate(scores):
        cands[j]["score"] = float(s)
    reranked = sorted(cands[:RERANK_TOP], key=lambda x: x.get("score", -1e9), reverse=True)
    diverse, seen = [], set()
    for c in reranked:
        src = c["doc"]["metadata"].get("source_file")
        if src not in seen:
            diverse.append(c)
            seen.add(src)
            if len(diverse) >= RETURN_K: break
    if len(diverse) < RETURN_K:
        for c in reranked:
            if c not in diverse:
                diverse.append(c)
                if len(diverse) >= RETURN_K: break
    return diverse[:RETURN_K]

def persona_instructions(persona: str) -> str:
    p = (persona or "").lower().strip()
    if p == "school": return "Audience: a 14-year-old. Short, simple, encouraging. One daily-life example. Use max 2 citations [n]."
    if p == "masters": return "Audience: graduate student. Define key Sanskrit terms. Discuss 1–2 schools (e.g., Advaita, Viśiṣṭādvaita). Use 3–4 citations [n]."
    if p == "phd": return "Audience: doctoral scholar. Contrast Śaṅkara vs Rāmānuja vs (optionally) Madhva; use technical Sanskrit with brief glosses. Use 5+ citations [n]."
    if p == "bhakta": return "Audience: devotee. Devotional framing; examples from saints and lived practice; actionable sādhanā steps. Use 2–4 citations [n]."
    return "Default graduate-level tone with 3–4 citations."

def build_system_message() -> str:
    return ("You are an expert Bhagavad Gītā teacher.\n"
            "Always structure the answer in exactly four sections:\n"
            "1) Importance\n2) Philosophical Interpretation (cite schools where relevant)\n3) Modern Relevance\n4) Practical Implementation\n\n"
            "Citations: When you use the provided context, cite with bracketed numbers [1]..[k] "
            "referring to the numbered context chunks. Do not invent sources. "
            "If context is insufficient, say so briefly.\n"
            "Conclusion: Invite the user to ask follow-ups about any part.")

def format_context_for_prompt(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["doc"]["metadata"]
        tag = build_short_tag(meta)
        content = re.sub(r"\s+", " ", c["doc"]["content"]).strip()
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"[{i}] {tag} — {content}")
    return "\n".join(lines)

def build_prompt(query_en: str, persona: str, chunks: List[Dict], history_pairs: List) -> List[Dict]:
    ctx = format_context_for_prompt(chunks)
    user_prompt = (f"Persona instructions: {persona_instructions(persona)}\n\n"
                   f"Question/Verse (English): {query_en}\n\n"
                   f"Context chunks (use [1]..[{len(chunks)}] for citations):\n{ctx}\n\n"
                   "Write the answer now.")
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for u, a in history_pairs:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user_prompt})
    return messages

# --- MODIFIED: Replaced run_llama_stream with a new function for transformers ---
def run_model_stream(messages: List[Dict], max_tokens: int):
    """Uses a threaded pipeline to stream tokens."""
    generation_kwargs = dict(
        max_new_tokens=max_tokens,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    # The pipeline call must be run in a thread so we can iterate on the streamer
    thread = Thread(target=llm_pipeline, args=(messages,), kwargs=generation_kwargs)
    thread.start()
    # Yield the tokens as they become available
    for new_text in streamer:
        yield new_text

# =========================
# One-time startup
# =========================
def _startup_once():
    global index, documents, embedder, reranker, translator, SYSTEM_MSG
    global llm_pipeline, tokenizer, streamer # --- MODIFIED ---
    
    log.info("Loading FAISS & docs from Hub…")
    idx, docs = load_kb_from_hub()
    index, documents = idx, docs
    log.info(f"FAISS vectors: {index.ntotal} | docs: {len(documents)}")

    log.info(f"Loading embedder… {EMBED_MODEL_NAME}")
    globals()["embedder"] = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    log.info(f"Loading reranker… {RERANK_MODEL}")
    globals()["reranker"] = CrossEncoder(RERANK_MODEL, device="cpu")

    # --- MODIFIED: Load transformers pipeline instead of llama.cpp ---
    log.info(f"Loading transformers model from Hub: {MODEL_REPO_ID}")
    globals()["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_REPO_ID)
    globals()["llm_pipeline"] = pipeline(
        "text-generation",
        model=MODEL_REPO_ID,
        tokenizer=tokenizer,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE
    )
    globals()["streamer"] = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    log.info(f"Model loaded on device: {DEVICE} with dtype: {TORCH_DTYPE}")
    
    log.info("Loading Google Translator client…")
    globals()["translator"] = Translator(service_urls=["translate.googleapis.com"])
    globals()["SYSTEM_MSG"] = build_system_message()
    log.info("Startup complete.")

_startup_once()

# =========================
# Gradio callbacks (Blocks)
# =========================
def messages_to_pairs(chat_messages: List[Dict]) -> List:
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
    t0 = time.time()
    user_msg = (user_msg or "").strip()
    if not user_msg:
        yield chat_history, ""
        return
    
    # We now use Gradio's native message dict format
    chat_history = (chat_history or []) + [{"role": "user", "content": user_msg}]

    translation = translate_to_en(user_msg) if is_sanskrit(user_msg) else None
    query_en = translation or user_msg
    chunks = rag_search(query_en)
    
    # --- MODIFIED: Get history from Gradio's dict format ---
    history_pairs = messages_to_pairs(chat_history[:-1]) # Exclude the current user message
    messages = build_prompt(query_en, persona, chunks, history_pairs)

    refs_lines = [f"- **[{i}]** {build_short_tag(c['doc']['metadata'])} — `{c['doc']['metadata'].get('source_file') or 'doc'}`" for i, c in enumerate(chunks, start=1)]
    citations_md = "\n".join(refs_lines) if refs_lines else "_No context found._"
    header = f"**Translation:** *{translation}*\n\n---\n" if translation else ""
    
    assistant_accum = ""
    chat_history.append({"role": "assistant", "content": header})
    yield chat_history, f"⌛ Generating…"
    
    # --- MODIFIED: Loop over the new streaming function ---
    for tok in run_model_stream(messages, max_tokens):
        assistant_accum += tok
        chat_history[-1]["content"] = header + assistant_accum
        yield chat_history, f"{time.time() - t0:.2f}s"
        
    final_response = (header + assistant_accum + "\n\n---\n### References\n" + citations_md + f"\n\n*Response generated in {time.time() - t0:.2f}s.*")
    chat_history[-1]["content"] = final_response
    yield chat_history, f"{time.time() - t0:.2f}s"

def clear_chat():
    return [], ""

# =========================
# Gradio UI (no ChatInterface)
# =========================
with gr.Blocks(title="GitaSphere AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🕉️ GitaSphere AI\nAsk questions or paste verses from the Bhagavad Gītā. The AI will use a knowledge base of classical commentaries to provide a structured answer.")
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            persona = gr.Dropdown(choices=["school", "masters", "phd", "bhakta"], value="masters", label="Persona / Depth")
            max_tokens = gr.Slider(minimum=256, maximum=2048, value=768, step=64, label="Max Response Length") # Increased max
            clear_btn = gr.Button("🧹 Clear History")
        with gr.Column(scale=3):
            # --- MODIFIED: Use Gradio's native message format for chatbot ---
            chatbot = gr.Chatbot(label="GitaSphere Conversation", show_label=False, height=600, render=False)
            gr.ChatInterface(
                fn=respond,
                chatbot=chatbot,
                additional_inputs=[persona, max_tokens],
                textbox=gr.Textbox(placeholder="Paste a verse or ask a question…", container=False, scale=7),
                submit_btn="✨ Ask GitaSphere",
                clear_btn=None, # Disable default clear, we have our own
            )
    
    # The ChatInterface handles the event logic, so manual .click is not needed
    # We connect our custom clear button to the chatbot and the textbox (which is part of ChatInterface)
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, chatbot.parent.parent.children[0]])


if __name__ == "__main__":
    # Note: `share=True` creates a public link. Be mindful of this.
    demo.queue().launch(server_name="0.0.0.0", share=True, server_port=int(os.environ.get("PORT", "8080")))
