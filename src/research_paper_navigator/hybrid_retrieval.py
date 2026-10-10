"""Small retrieval-only experiment: fixed vs per-subquery hybrid weights.

Semantic search embeds the first 1,400 characters of each paper's first PDF
page, not all 6,105 pages. BM25 scores every PDF page. Semantic hits nominate
papers; BM25 picks their most relevant page. This is a cheap exploratory test,
not a full page-level semantic index. The same decompositions and five-page
budget are used in both arms. The original capstone solution is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi

from .advanced_retrieval import (
    BASELINE, CP31, DOCS, DOC_BY_ID, FINAL_K, FIRST_PAGE_BY_SOURCE,
    PaperIndex, TITLE_BY_SOURCE,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
CACHE_PATH = PROJECT_ROOT / ".cache" / "research_papers_abstract_embeddings.json"
RESULT_PATH = PROJECT_ROOT / "outputs" / "hybrid_retrieval_results.json"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
SOURCE_ORDER = sorted(FIRST_PAGE_BY_SOURCE)
MIXED_FACETS = {
    3: [
        "The Sketching Approach to Program Synthesis how programmer creates sketches",
        "Learning to Infer Program Sketches how neural model generates sketches",
        "manual program sketches versus automatically inferred program sketches",
    ],
    5: [
        "Learning to Infer Program Sketches evaluation results",
        "benefits of automatically inferred sketches compared with no sketches",
        "limitations failure cases of neural sketch inference on programming tasks",
    ],
}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def embedding_client() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=BASELINE.check_api_key(),
        base_url=BASELINE.OPENROUTER_BASE_URL,
        check_embedding_ctx_length=False,
    )


def paper_embeddings(client: OpenAIEmbeddings) -> np.ndarray:
    passages = [FIRST_PAGE_BY_SOURCE[source]["text"][:1400] for source in SOURCE_ORDER]
    signature = hashlib.sha256(json.dumps(passages, ensure_ascii=False).encode("utf-8")).hexdigest()
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if cache.get("signature") == signature and cache.get("model") == EMBEDDING_MODEL:
            print(f"Using cached embeddings for {len(SOURCE_ORDER)} paper first pages")
            return np.asarray(cache["vectors"], dtype=np.float32)
    print(f"Embedding {len(passages)} paper first pages (not full PDFs)...", flush=True)
    vectors = client.embed_documents(passages)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps({
        "model": EMBEDDING_MODEL, "signature": signature,
        "sources": SOURCE_ORDER, "vectors": vectors,
    }), encoding="utf-8")
    return np.asarray(vectors, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a @ b.T) / (np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :] + 1e-12)


def bm25_weight(subquery: str) -> float:
    """Favor BM25 for a named title; semantic search for paraphrased concepts."""
    for title in set(TITLE_BY_SOURCE.values()):
        if len(title) >= 24 and title in subquery.casefold():
            return 0.75
    return 0.25


class HybridPaperIndex:
    def __init__(self, vectors: np.ndarray) -> None:
        self.paper_index = PaperIndex()
        self.docs = DOCS
        self.bm25 = BM25Okapi([tokens(doc["text"]) for doc in DOCS])
        self.vectors = vectors
        self.source_position = {source: pos for pos, source in enumerate(SOURCE_ORDER)}

    def candidates(self, query: str, query_vector: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        raw_bm = self.bm25.get_scores(tokens(query))
        bm_order = np.argsort(-raw_bm, kind="stable")[:40]
        paper_sim = cosine(query_vector[None, :], self.vectors)[0]
        source_order = np.argsort(-paper_sim, kind="stable")[:12]
        candidate_ids = {self.docs[int(pos)]["id"] for pos in bm_order if raw_bm[pos] > 0}
        for source_pos in source_order:
            source = SOURCE_ORDER[int(source_pos)]
            pages = self.paper_index.pages_by_paper[source]
            best_page = max(pages, key=lambda p: raw_bm[self.doc_position[p["id"]]])
            candidate_ids.add(best_page["id"])
            candidate_ids.add(FIRST_PAGE_BY_SOURCE[source]["id"])
        bm = {doc_id: float(raw_bm[self.doc_position[doc_id]]) for doc_id in candidate_ids}
        vec = {
            doc_id: float(paper_sim[self.source_position[DOC_BY_ID[doc_id]["source"]]])
            for doc_id in candidate_ids
        }
        return normalize(bm), normalize(vec)

    @property
    def doc_position(self) -> dict[str, int]:
        if not hasattr(self, "_doc_position"):
            self._doc_position = {doc["id"]: pos for pos, doc in enumerate(self.docs)}
        return self._doc_position

    def retrieve(self, subqueries: list[str], query_vectors: np.ndarray, dynamic: bool) -> tuple[list[str], list[float]]:
        totals: dict[str, float] = defaultdict(float)
        facet_best: list[str] = []
        weights: list[float] = []
        for subquery, vector in zip(subqueries, query_vectors):
            weight = bm25_weight(subquery) if dynamic else 0.5
            weights.append(weight)
            bm, semantic = self.candidates(subquery, vector)
            fused = {doc_id: weight * bm.get(doc_id, 0) + (1 - weight) * semantic.get(doc_id, 0)
                     for doc_id in bm.keys() | semantic.keys()}
            ranked = sorted(fused, key=fused.get, reverse=True)
            ranked = self.paper_index.unique_pages(ranked, 12)
            if ranked:
                facet_best.append(ranked[0])
            # Per-facet rescaling prevents one subquery's numerical range from
            # dominating the merged ranking merely because its weights changed.
            scaled = normalize({doc_id: fused[doc_id] for doc_id in ranked})
            for doc_id, score in scaled.items():
                totals[doc_id] += score
        merged = list(dict.fromkeys(facet_best)) + sorted(totals, key=totals.get, reverse=True)
        return self.paper_index.unique_pages(merged, FINAL_K), weights


def main() -> None:
    previous = json.loads((HERE / "checkpoint_4_1_real_papers_results_v2.json").read_text(encoding="utf-8"))
    client = embedding_client()
    index = HybridPaperIndex(paper_embeddings(client))
    all_subqueries = list(dict.fromkeys(
        s for number, case in enumerate(previous["cases"], 1)
        for s in MIXED_FACETS.get(number, case["subqueries"])
    ))
    query_vectors = client.embed_documents(all_subqueries)
    vector_by_query = {query: np.asarray(vector, dtype=np.float32)
                       for query, vector in zip(all_subqueries, query_vectors)}
    results = []
    for number, case in enumerate(previous["cases"], 1):
        subs = MIXED_FACETS.get(number, case["subqueries"])
        vectors = np.asarray([vector_by_query[sub] for sub in subs])
        record = {"question": case["question"], "subqueries": subs,
                  "manually_mixed_facets": number in MIXED_FACETS, "modes": {}}
        for mode, dynamic in (("fixed_50_50", False), ("dynamic", True)):
            pages, weights = index.retrieve(subs, vectors, dynamic)
            hits = [{"id": doc_id, "paper": DOC_BY_ID[doc_id]["source"],
                     "pdf_position": DOC_BY_ID[doc_id]["page"]} for doc_id in pages]
            record["modes"][mode] = {"bm25_weights": weights, "hits": hits}
            print(f"Q{number} {mode} weights={weights}: " + ", ".join(
                f"{hit['paper']}#{hit['pdf_position']}" for hit in hits
            ), flush=True)
        results.append(record)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({"design": "153 first-page semantic embeddings + 6105-page BM25",
                                       "cases": results}, indent=2), encoding="utf-8")
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
