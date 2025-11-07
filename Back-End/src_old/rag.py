import os
from typing import List, Tuple
from rank_bm25 import BM25Okapi

def load_kb() -> List[Tuple[str, str]]:
    kb_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kb")
    texts = []
    for fn in os.listdir(kb_dir):
        if fn.endswith(".md"):
            path = os.path.join(kb_dir, fn)
            with open(path, "r", encoding="utf-8") as f:
                texts.append((fn, f.read()))
    return texts

def search_kb(query: str, k: int = 4) -> List[Tuple[str, str]]:
    docs = load_kb()
    if not docs: 
        return []
    corpus = [t[1] for t in docs]
    tokenized = [d.split() for d in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:k]]
