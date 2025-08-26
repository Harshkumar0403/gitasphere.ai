# scripts/server.py
import os, time, re, logging, pickle
from typing import List, Dict, Optional
import unicodedata
import numpy as np
import faiss
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer, CrossEncoder
from llama_cpp import Llama
from googletrans import Translator

# ---------- Config via env ----------
FAISS_PATH = os.environ.get("RAG_FAISS_PATH", "gita_knowledge_base.faiss")
DOCS_PATH  = os.environ.get("RAG_DOCS_PATH",  "gita_documents.pkl")
GGUF_PATH  = os.environ.get("GGUF_PATH",      "checkpoints/gita-model-q4_K_M.gguf")

EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
RERANK_MODEL     = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Smaller fanout for CPU responsiveness
INITIAL_K        = int(os.environ.get("RAG_INITIAL_K", "12"))
RERANK_TOP       = int(os.environ.get("RERANK_TOP", "12"))
RETURN_K         = int(os.environ.get("RAG_RETURN_K", "5"))

# llama.cpp generation params (CPU-tuned)
N_THREADS   = int(os.environ.get("LLAMA_THREADS", str(os.cpu_count() or 4)))
N_CTX       = int(os.environ.get("LLAMA_CTX", "4096"))
N_BATCH     = int(os.environ.get("LLAMA_BATCH", "256"))
TEMPERATURE = float(os.environ.get("LLAMA_TEMP", "0.6"))
TOP_P       = float(os.environ.get("LLAMA_TOP_P", "0.8"))
MIROSTAT    = int(os.environ.get("LLAMA_MIROSTAT", "0"))
MIROSTAT_TAU= float(os.environ.get("LLAMA_MIROSTAT_TAU", "5.0"))
MIROSTAT_ETA= float(os.environ.get("LLAMA_MIROSTAT_ETA", "0.1"))

logging.basicConfig(level=logging.INFO)

# ---------- App ----------
app = FastAPI(title="GitaSphere AI API", version="0.2")

# ---------- Globals (loaded once) ----------
index = None
documents: List[Dict] = []
embedder: Optional[SentenceTransformer] = None
reranker: Optional[CrossEncoder] = None
llm: Optional[Llama] = None
translator: Optional[Translator] = None
SYSTEM_MSG: Optional[str] = None

# ---------- Models ----------
class GenerateRequest(BaseModel):
    query: str
    persona: str = Field(default="masters", description="school | masters | phd | bhakta")
    max_tokens: int = Field(default=512, ge=64, le=1024)

class Citation(BaseModel):
    num: int
    source: str
    file: Optional[str] = None

class GenerateResponse(BaseModel):
    translation: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    answer: str

# ---------- Utils ----------
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')  # robust Devanagari detector
def is_sanskrit(text: str) -> bool:
    if not text:
        return False
    
    # Count Devanagari characters
    devanagari_chars = 0
    total_chars = 0
    
    for char in text:
        if char.strip():  # Skip whitespace
            total_chars += 1
            if '\u0900' <= char <= '\u097F':  # Devanagari range
                devanagari_chars += 1
    
    # If more than 50% are Devanagari characters
    if total_chars > 0:
        return (devanagari_chars / total_chars) > 0.5
    
    return False

def translate_to_en(text: str) -> Optional[str]:
    try:
        if not text or not text.strip():
            return None
        res = translator.translate(text, src='auto', dest='en')
        return res.text
    except Exception as e:
        logging.warning(f"Google Translate failed: {e}")
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

def rag_search(query: str):
    # 1) ANN fetch
    qv = embed([query])
    D, I = index.search(qv, INITIAL_K)
    I = I[0].tolist()
    D = D[0].tolist()
    # If index is IP: larger is better; treat distance as similarity score directly
    candidates = [{"idx": i, "score": float(d), "doc": documents[i]}
                  for i, d in zip(I, D) if i >= 0]

    # 2) Rerank top-K with cross-encoder
    if not candidates:
        return []
    pairs = [[query, c["doc"]["content"]] for c in candidates[:RERANK_TOP]]
    scores = reranker.predict(pairs, batch_size=32) if pairs else []
    for j, s in enumerate(scores):
        candidates[j]["score"] = float(s)
    candidates = sorted(candidates[:RERANK_TOP], key=lambda x: x["score"], reverse=True)
    return candidates[:RETURN_K]

def persona_instructions(persona: str) -> str:
    persona = (persona or "").lower().strip()
    if persona == "school":
        return ("Audience: a 14-year-old. Short, simple, encouraging. "
                "One daily-life example. Use max 2 citations [n].")
    if persona == "masters":
        return ("Audience: graduate student. Define key Sanskrit terms. "
                "Discuss 1–2 schools (e.g., Advaita, Viśiṣṭādvaita). Use 3–4 citations [n].")
    if persona == "phd":
        return ("Audience: doctoral scholar. Contrast Śaṅkara vs Rāmānuja vs (optionally) Madhva; "
                "use technical Sanskrit with brief glosses. Use 5+ citations [n].")
    if persona == "bhakta":
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
        "If context is insufficient, say so briefly."
    )

def format_context_for_prompt(chunks: List[Dict]) -> str:
    # Deterministic, numbered, short-tagged context for reliable [n] citations
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["doc"]["metadata"]
        tag  = build_short_tag(meta)
        content = c["doc"]["content"].replace("\n", " ").strip()
        if len(content) > 600:
            content = content[:600] + "..."
        lines.append(f"[{i}] {tag} — {content}")
    return "\n".join(lines)

def build_prompt(query_en: str, persona: str, chunks: List[Dict]) -> List[Dict]:
    ctx = format_context_for_prompt(chunks)
    sys = SYSTEM_MSG
    user = (
        f"Persona instructions: {persona_instructions(persona)}\n\n"
        f"Question/Verse (English): {query_en}\n\n"
        f"Context chunks (use [1]..[{len(chunks)}] for citations):\n{ctx}\n\n"
        "Write the answer now."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

def run_llama(messages: List[Dict], max_tokens: int) -> str:
    # llama.cpp chat completion; prefer modern args names
    kwargs = dict(
        messages=messages,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=max_tokens,
        stop=["</s>"],
    )
    # enable mirostat if requested
    if MIROSTAT in (1, 2):
        kwargs["mirostat"] = MIROSTAT
        kwargs["mirostat_tau"] = MIROSTAT_TAU
        kwargs["mirostat_eta"] = MIROSTAT_ETA

    out = llm.create_chat_completion(**kwargs)
    return out["choices"][0]["message"]["content"].strip()

# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    global index, documents, embedder, reranker, llm, translator, SYSTEM_MSG

    print("Loading FAISS & docs...")
    index, documents = load_kb()
    print(f"FAISS vectors: {index.ntotal}  | docs: {len(documents)}")

    print("Loading embedder...", EMBED_MODEL_NAME)
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    print("Loading reranker...", RERANK_MODEL)
    reranker = CrossEncoder(RERANK_MODEL, device="cpu")

    print("Loading llama.cpp...", GGUF_PATH)
    llm = Llama(
        model_path=GGUF_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_batch=N_BATCH,
        embedding=False,
        verbose=False,
    )

    print("Loading Google Translator client...")
    translator = Translator(service_urls=['translate.googleapis.com'])

    SYSTEM_MSG = build_system_message()
    print("Startup complete.")

# ---------- Routes ----------
@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    t0 = time.time()
    q = (req.query or "").strip()
    translation = None

    # Translate if Sanskrit (auto)
    if is_sanskrit(q):
        try:
            tr = translator.translate(q, src="auto", dest="en")
            translation = tr.text
            print(translation)
            query_en = translation
        except Exception:
            translation = None
            query_en = q  # fallback
    else:
        query_en = q

    # RAG
    chunks = rag_search(query_en)
    messages = build_prompt(query_en, req.persona, chunks)
    answer = run_llama(messages, max_tokens=req.max_tokens)

    # Minimal citation record
    cites = []
    for i, c in enumerate(chunks, 1):
        meta = c["doc"]["metadata"]
        cites.append(Citation(num=i, source=build_short_tag(meta), file=meta.get("source_file")))

    print(f"Served in {time.time() - t0:.2f}s")
    return GenerateResponse(translation=translation, citations=cites, answer=answer)

