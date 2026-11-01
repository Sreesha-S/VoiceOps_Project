"""Searching the knowledge base.

Two search methods, then we merge the rankings:
  * KEYWORD (BM25)   - great for exact things: "ELE-0012", "COD", "RET-8".
  * MEANING (embeddings, optional) - finds "send it back" when the document says "return".
                       Gemini turns each chunk into a list of numbers; similar meaning = similar numbers.
Merging uses Reciprocal Rank Fusion (RRF): every method votes 1/(60+rank) for each chunk.

Extra rules:
  * PRIORS: the knowledge base says policy > agent guide > help > FAQ > marketing, and superseded
    documents last. We multiply scores by a small weight so a vague FAQ cannot beat the real policy.
  * PRODUCT LANE: product rows are searched separately and capped at 2, otherwise a question naming a
    product would fill all results with product rows and push the policy out.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .kb import Chunk

_STOP = set("a an the is are was were be been am i me my we our you your it its this that of to in on at for from with by and or not no "
            "do does did can could will would should may might have has had how what when where which who why if then than so as about "
            "into please kya hai hain ho ka ki ke ko se mein pe par aur ya hi bhi".split())
_TOKEN = re.compile(r"[a-z0-9\u0900-\u097f]+")


def tokenize(text: str) -> list[str]:
    out = []
    for t in _TOKEN.findall(text.lower()):
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(d)) for d in docs]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = float(self.len.mean()) if len(docs) else 1.0
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q, out = tokenize(query), np.zeros(len(self.tf))
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
            out[i] = sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if tf.get(t))
        return out


class APIEmbedder:
    """Embeddings from any OpenAI-compatible endpoint (we use Gemini's)."""

    def __init__(self, settings):
        from openai import OpenAI
        from .llm import with_retries

        self.name = settings.embedding_model
        self._retry = with_retries
        self.client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url, timeout=60, max_retries=0)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), 32):
            batch = texts[i:i + 32]
            resp = self._retry(lambda: self.client.embeddings.create(model=self.name, input=batch))
            vecs.extend(d.embedding for d in resp.data)
        arr = np.asarray(vecs, dtype=float)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True)


PRIORS = {"policy": 1.0, "catalog": 1.0, "agent_guide": 0.95, "meta": 0.9, "help": 0.9, "faq": 0.8, "marketing": 0.6}
SUPERSEDED = 0.7
_SKU = re.compile(r"\b([A-Za-z]{3})[-\s]?(\d{4})\b")


@dataclass
class Retriever:
    chunks: list
    embedder: object = None
    cache_dir: object = None
    by_id: dict = field(init=False)
    bm25: BM25 = field(init=False)
    vectors: object = field(init=False, default=None)
    mode: str = field(init=False, default="bm25")

    def __post_init__(self):
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25([c.for_index() for c in self.chunks])
        if self.embedder is not None:
            texts = [c.for_index() for c in self.chunks]
            key = hashlib.sha1((self.embedder.name + "\x00" + "\x01".join(texts)).encode()).hexdigest()[:16]
            path = (self.cache_dir / f"emb_{key}.npy") if self.cache_dir else None
            if path and path.exists():
                self.vectors = np.load(path)
            else:
                self.vectors = self.embedder.embed(texts)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, self.vectors)
            self.mode = "bm25+embeddings"

    @staticmethod
    def _ranks(scores: np.ndarray) -> dict:
        return {int(i): r for r, i in enumerate(np.argsort(-scores)) if scores[i] > 0}

    def _prior(self, c: Chunk) -> float:
        return PRIORS.get(c.doc_type, 0.9) * (SUPERSEDED if c.status == "superseded" else 1.0)

    def _product_hits(self, queries: list[str], cap: int = 2) -> list[Chunk]:
        found: dict = {}
        for q in queries:
            for pre, num in _SKU.findall(q or ""):
                cid = f"products.csv:{pre.upper()}-{num}"
                if cid in self.by_id:
                    found[cid] = 10.0
        qtok = set(tokenize(" ".join(queries)))
        for c in self.chunks:
            if c.doc_type == "catalog" and c.chunk_id not in found:
                ttok = set(tokenize(c.text.split(" (SKU")[0]))
                inter = len(ttok & qtok)
                if inter >= 2 and inter / max(len(ttok), 1) >= 0.5:
                    found[c.chunk_id] = inter / len(ttok)
        return [self.by_id[cid] for cid, _ in sorted(found.items(), key=lambda kv: -kv[1])[:cap]]

    def search(self, queries: list[str], k: int = 5, dense_query: str | None = None) -> list[Chunk]:
        queries = [q for q in queries if q and q.strip()]
        products = self._product_hits(queries)
        # Product-aware search: the catalog says which return category a product is in (innerwear, beauty, ...),
        # and that category decides which return rule applies. Search the rules for that category too.
        kb_queries = queries + [f"{cat} return rules" for cat in dict.fromkeys(
            m.group(1) for c in products for m in [re.search(r"return_category: ([\w-]+)", c.text)] if m)]
        fused: dict = {}

        def vote(ranks):
            for idx, r in ranks.items():
                if self.chunks[idx].doc_type != "catalog":
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + r)

        for q in kb_queries:
            vote(self._ranks(self.bm25.scores(q)))
        if self.vectors is not None and (dense_query or queries):
            try:
                qv = self.embedder.embed([dense_query or queries[-1]])[0]
                vote(self._ranks(self.vectors @ qv))
            except Exception as exc:  # noqa: BLE001 - never let search fail because embeddings failed
                print(f"  (embedding search skipped: {str(exc)[:80]})", file=sys.stderr)
        ranked = sorted(((s * self._prior(self.chunks[i]), i) for i, s in fused.items()), reverse=True)
        hits = [self.chunks[i] for _, i in ranked[:k]]
        seen = {c.chunk_id for c in hits}
        for c in products:
            if c.chunk_id not in seen:
                hits.append(c)
                seen.add(c.chunk_id)
        # Linked rule: Assured warranty never applies to marketplace-seller items (MKS-3).
        if re.search(r"assur|warrant|वारंटी", " ".join(queries).lower()) and "MKS-3" not in seen:
            if any(c.doc_type == "catalog" and "fulfilled_by: ZentroMart." not in c.text for c in hits):
                hits.append(self.by_id["MKS-3"])
        return hits


def build_retriever(settings, verbose: bool = True) -> Retriever:
    from .kb import load_all_chunks

    chunks = load_all_chunks(settings.dataset_dir)
    embedder = None
    if settings.embedding_backend in ("auto", "api") and settings.llm_api_key:
        try:
            embedder = APIEmbedder(settings)
            r = Retriever(chunks, embedder, settings.cache_dir)
            if verbose:
                print(f"[search] keyword + {settings.embedding_model} embeddings", file=sys.stderr)
            return r
        except Exception as exc:  # noqa: BLE001
            if settings.embedding_backend == "api":
                raise
            print(f"[search] embeddings unavailable ({str(exc)[:80]}); using keyword search only", file=sys.stderr)
    return Retriever(chunks)
