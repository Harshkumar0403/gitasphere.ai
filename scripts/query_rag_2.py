#!/usr/bin/env python3
# query_rag.py — FAISS retrieval + cosine re-score + Gita boost + Cross-Encoder rerank

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import time
import re
from typing import List, Tuple, Optional

# ====== CONFIG ======
FAISS_INDEX_PATH      = 'gita_knowledge_base.faiss'
DOCUMENTS_PKL_PATH    = 'gita_documents.pkl'
EMBED_MODEL_NAME      = 'all-mpnet-base-v2'     # must match index build
DEVICE                = 'cuda' if torch.cuda.is_available() else 'cpu'

# Retrieval knobs
TOPK_RETURN           = 5        # final results to show
TOPK_INITIAL          = 50       # first-stage hits (>= TOPK_RETURN)
NORMALIZE_QUERY       = True     # L2-normalize query before FAISS search
ENABLE_RESCORING      = True     # cosine re-score using reconstructed doc vecs

# Bhagavad-Gita small boost (helps verse queries)
GITA_BOOST            = 0.10
GITA_PAT              = re.compile(r'g(?:i|ī)ta|g(?:i|ī)tā', re.IGNORECASE)

# Cross-encoder reranker
RERANKER_ENABLED      = True
RERANKER_MODEL        = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOPN           = 50       # how many FAISS hits to rerank
RERANK_BATCH_SIZE     = 64
RERANK_BLEND_ALPHA    = 0.80     # final_score = alpha*ce_score + (1-alpha)*base_score

# Truncation for reranker input (char-based safeguard; tokenization still applies)
DOC_TRUNC_CHARS       = 1200

# ====================

def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n

def load_knowledge_base():
    print("Loading the knowledge base...")
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(DOCUMENTS_PKL_PATH, 'rb') as f:
            documents = pickle.load(f)
        print(f"Successfully loaded FAISS index with {index.ntotal} vectors.")
        print(f"Successfully loaded {len(documents)} document chunks.")
        return index, documents
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        print("Please ensure you have built the KB (build_rag_database.py).")
        return None, None

def safe_get_text(doc) -> str:
    if isinstance(doc, dict):  # expected shape: {'content': str, 'metadata': {...}}
        return doc.get('content', '')
    if isinstance(doc, (list, tuple)) and len(doc) >= 1:
        return doc[0]
    return ""

def safe_get_meta(doc) -> dict:
    if isinstance(doc, dict):
        return doc.get('metadata', {})
    if isinstance(doc, (list, tuple)) and len(doc) >= 2 and isinstance(doc[1], dict):
        return doc[1]
    return {}

def doc_title_for_boost(meta: dict) -> str:
    fields = [
        meta.get('source_file', ''),
        meta.get('book_title', ''),
        meta.get('title', ''),
        meta.get('path', ''),
    ]
    return " ".join(str(x) for x in fields if x)

def cosine_rescore(index, query_vec: np.ndarray, indices: np.ndarray):
    """
    Reconstruct document vectors, L2-normalize, and compute cosine similarity to the (already
    normalized) query vector. Returns (cos_scores, valid_ids) or (None, None) if reconstruction fails.
    """
    q = query_vec.copy().astype('float32')
    if NORMALIZE_QUERY:
        q = l2norm(q)

    doc_vecs = []
    valid_ids = []
    for idx in indices:
        if idx < 0:
            continue
        try:
            v = index.reconstruct(int(idx))  # may fail for IVF/HNSW without reconstruct
            doc_vecs.append(v)
            valid_ids.append(int(idx))
        except Exception:
            return None, None

    if not doc_vecs:
        return None, None

    doc_vecs = np.stack(doc_vecs).astype('float32')
    doc_vecs = l2norm(doc_vecs)
    scores = (doc_vecs @ q[0].T).reshape(-1)  # cosine since both are unit
    return scores, np.array(valid_ids, dtype=np.int64)

# ---------- Cross-Encoder loader(s) ----------
class CrossEncoderWrapper:
    """Abstracts over sentence-transformers CrossEncoder and transformers-based rerankers."""
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.backend = None  # "st" or "hf"
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self):
        if self.model_name.startswith("cross-encoder/"):
            # sentence-transformers CrossEncoder
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            self.backend = "st"
        else:
            # Try transformers AutoModelForSequenceClassification (e.g., BAAI/bge-reranker-v2-m3)
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.backend = "hf"

    @torch.no_grad()
    def score(self, pairs: List[Tuple[str, str]], batch_size: int = 64) -> List[float]:
        if self.backend == "st":
            # sentence-transformers CrossEncoder returns numpy array of scores
            scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
            return [float(s) for s in scores]
        # transformers path
        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i+batch_size]
            texts_a = [a for a, _ in chunk]
            texts_b = [b for _, b in chunk]
            # Many rerankers expect pair encoded as 'text'/'text_pair'
            inputs = self.tokenizer(
                texts_a,
                texts_b,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits  # shape [bsz, 1] or [bsz]
            s = logits.view(-1).float().tolist()
            scores.extend(s)
        return scores

# ---------- Main retrieval ----------
def search_knowledge_base(query_text, index, documents, embedding_model, top_k=TOPK_RETURN):
    if index is None or documents is None:
        print("Knowledge base is not loaded. Cannot perform search.")
        return

    print(f"\nSearching for: '{query_text}'")
    print("-" * 30)

    # 1) Encode query
    t0 = time.time()
    q = embedding_model.encode([query_text], convert_to_numpy=True)  # shape (1, D)
    if NORMALIZE_QUERY:
        q = l2norm(q.astype('float32'))
    t1 = time.time()
    print(f"Query encoded in {t1 - t0:.4f} seconds.")

    # 2) FAISS search
    k1 = max(top_k, TOPK_INITIAL, RERANK_TOPN if RERANKER_ENABLED else 0)
    t0 = time.time()
    faiss_scores, faiss_ids = index.search(q.astype('float32'), k1)
    t1 = time.time()
    print(f"FAISS search completed in {t1 - t0:.4f} seconds.")
    faiss_ids = faiss_ids[0]
    faiss_scores = faiss_scores[0]

    # 3) Cosine re-score (if possible)
    rescored = []
    cos_scores, valid_ids = (None, None)
    if ENABLE_RESCORING:
        cos_scores, valid_ids = cosine_rescore(index, q, faiss_ids)
    if cos_scores is not None:
        id2cos = {int(i): float(s) for i, s in zip(valid_ids, cos_scores)}
        for idx, ip_sc in zip(faiss_ids, faiss_scores):
            if idx < 0: 
                continue
            base = id2cos.get(int(idx), float(ip_sc))
            meta = safe_get_meta(documents[idx])
            if GITA_PAT.search(doc_title_for_boost(meta)):
                base += GITA_BOOST
            rescored.append((base, int(idx)))
    else:
        for idx, ip_sc in zip(faiss_ids, faiss_scores):
            if idx < 0: 
                continue
            base = float(ip_sc)
            meta = safe_get_meta(documents[idx])
            if GITA_PAT.search(doc_title_for_boost(meta)):
                base += GITA_BOOST
            rescored.append((base, int(idx)))

    # 4) Cross-encoder rerank (optional)
    final_candidates = rescored
    if RERANKER_ENABLED and len(rescored) > 0:
        ce = CrossEncoderWrapper(RERANKER_MODEL, DEVICE)
        # Take top-N by current score, then rerank with CE
        rescored.sort(key=lambda x: x[0], reverse=True)
        pool = rescored[:min(RERANK_TOPN, len(rescored))]
        pairs = []
        for _, idx in pool:
            txt = safe_get_text(documents[idx])[:DOC_TRUNC_CHARS]
            pairs.append((query_text, txt if txt else ""))
        t0 = time.time()
        ce_scores = ce.score(pairs, batch_size=RERANK_BATCH_SIZE)
        t1 = time.time()
        print(f"Cross-encoder rerank ({RERANKER_MODEL}) on {len(pairs)} pairs took {t1 - t0:.2f}s.")
        # Normalize CE scores (z-score) to blend robustly
        ce_arr = np.array(ce_scores, dtype=np.float32)
        if len(ce_arr) > 1:
            ce_norm = (ce_arr - ce_arr.mean()) / (ce_arr.std() + 1e-6)
        else:
            ce_norm = ce_arr
        # Blend: final_score = alpha * CE + (1-alpha) * base
        blended = []
        for (base, idx), ce_s in zip(pool, ce_norm):
            blended.append((RERANK_BLEND_ALPHA * float(ce_s) + (1.0 - RERANK_BLEND_ALPHA) * float(base), idx))
        final_candidates = blended

    # 5) Final sort & display
    final_candidates.sort(key=lambda x: x[0], reverse=True)
    final = final_candidates[:top_k]

    print(f"\n--- Top {top_k} Relevant Chunks (final) ---")
    for rank, (score, idx) in enumerate(final, 1):
        doc = documents[idx]
        content = safe_get_text(doc)
        meta = safe_get_meta(doc)
        source = meta.get('source_file', meta.get('path', 'N/A'))
        author = meta.get('author', 'N/A')
        school = meta.get('school', 'N/A')
        chapter = meta.get('chapter', None)
        verse = meta.get('verse', None)

        print(f"\nResult {rank} (Score: {score:.4f})")
        print(f"Source: {source}")
        print(f"Author: {author}, School: {school}, ch={chapter}, v={verse}")
        print("-" * 20)
        snippet = content[:800] + "..." if len(content) > 800 else content
        print(snippet if snippet.strip() else "(empty text)")
        print("-" * 20)

def main():
    index, documents = load_knowledge_base()
    if index is None:
        return

    print(f"Loading embedding model ({EMBED_MODEL_NAME}) on {DEVICE}...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)
    print("Embedder ready.")

    # Try a representative verse-shaped query
    test_query = "You have a right to perform your prescribed duties, but not to the fruits of action."
    search_knowledge_base(test_query, index, documents, embedder, top_k=TOPK_RETURN)

if __name__ == '__main__':
    main()

