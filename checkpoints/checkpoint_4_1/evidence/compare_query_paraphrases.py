"""Retrieval-only comparison of original and manually paraphrased subqueries.

Uses the same 50/50 hybrid retriever, paper-embedding cache, facet merge,
and five-page budget in both arms. Q4 and Q5 are the two weak cases from the
Checkpoint 4.1 research-paper test. No answer LLM or judge is called here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compare_dynamic_subquery_weights import (
    HybridPaperIndex, embedding_client, paper_embeddings,
)
from capstone_checkpoint_4_1_real_papers_eval import DOC_BY_ID

HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "query_paraphrase_retrieval_results.json"
PARAPHRASES = {
    4: [
        "human writes a partial program with blanks that a solver fills in",
        "synthesizer completes holes in incomplete source code while meeting constraints",
        "developer supplies a program template and an automated tool searches for a valid completion",
    ],
    5: [
        "learned intermediate program templates help code generation from examples",
        "experiments comparing search success with and without predicted partial programs",
        "failure cases and limits of predicting partial code structures from task specifications",
    ],
}
EXPECTED_PAPER = {
    4: "DBLP_conf_sigsoft_JeonQFS15.pdf",  # JSKETCH, a directly relevant method.
    5: "DBLP_conf_icml_NyeHTS19.pdf",     # Learning to Infer Program Sketches.
}
EQUIVALENT_PAPER = {
    5: "DBLP_journals_corr_abs-1902-06349.pdf",  # Preprint of the same paper.
}


def main() -> None:
    previous = json.loads((HERE / "checkpoint_4_1_real_papers_results_v2.json").read_text(encoding="utf-8"))
    client = embedding_client()
    index = HybridPaperIndex(paper_embeddings(client))
    queries = list(dict.fromkeys(
        subquery
        for number in PARAPHRASES
        for subquery in previous["cases"][number - 1]["subqueries"] + PARAPHRASES[number]
    ))
    vectors = client.embed_documents(queries)
    vector_by_query = {query: np.asarray(vector, dtype=np.float32)
                       for query, vector in zip(queries, vectors)}
    results = []
    for number in PARAPHRASES:
        case = previous["cases"][number - 1]
        record = {"question": case["question"], "expected_paper": EXPECTED_PAPER[number], "modes": {}}
        for mode, subqueries in (("original", case["subqueries"]),
                                 ("manual_paraphrase", PARAPHRASES[number])):
            matrix = np.asarray([vector_by_query[query] for query in subqueries])
            pages, weights = index.retrieve(subqueries, matrix, dynamic=False)
            hits = [{"id": doc_id, "paper": DOC_BY_ID[doc_id]["source"],
                     "pdf_position": DOC_BY_ID[doc_id]["page"]} for doc_id in pages]
            relevant_sources = {EXPECTED_PAPER[number], EQUIVALENT_PAPER.get(number)}
            relevant_rank = next((pos for pos, hit in enumerate(hits, 1)
                                  if hit["paper"] in relevant_sources), None)
            record["modes"][mode] = {
                "subqueries": subqueries, "bm25_weights": weights,
                "first_relevant_rank": relevant_rank, "hits": hits,
            }
            print(f"Q{number} {mode}: first relevant rank={relevant_rank}; " + ", ".join(
                f"{hit['paper']}#{hit['pdf_position']}" for hit in hits
            ), flush=True)
        results.append(record)
    RESULT_PATH.write_text(json.dumps({
        "design": "Retrieval-only; same 50/50 weights and five-page budget; manual paraphrases",
        "cases": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved {RESULT_PATH}")


if __name__ == "__main__":
    main()
