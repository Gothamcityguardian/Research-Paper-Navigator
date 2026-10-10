"""Compare four retrieval modes on the real Research Paper Navigator corpus.

Modes: the Checkpoint 2.1 page-level keyword baseline, decomposition-only,
graph-only, and decomposition followed by graph expansion. Each mode gives the
answer model exactly five PDF pages. The five Checkpoint 3.1 grading notes are
reused without alteration. This is an experiment, not a tuned final retriever.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from langchain_core.messages import HumanMessage, SystemMessage

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULT_PATH = OUTPUT_DIR / "advanced_retrieval_results.json"
FINAL_K = 5
GRAPH_SEEDS = 3
DECOMP_SEEDS = 3


from . import evaluation as CP31
BASELINE = CP31.BASELINE
DOCS = BASELINE.SAMPLE_DOCS
DOC_BY_ID = BASELINE.DOC_BY_ID
FIRST_PAGE_BY_SOURCE = {}
for _doc in DOCS:
    _source = _doc["source"]
    if _source not in FIRST_PAGE_BY_SOURCE or _doc["page"] < FIRST_PAGE_BY_SOURCE[_source]["page"]:
        FIRST_PAGE_BY_SOURCE[_source] = _doc
TITLE_BY_SOURCE = {
    source: re.sub(r"\s+", " ", page["text"].splitlines()[0]).strip().casefold()
    for source, page in FIRST_PAGE_BY_SOURCE.items()
    if page["text"].splitlines()
}

STOP = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of", "on",
    "or", "the", "to", "using", "via", "with", "towards", "toward", "approach",
    "study", "paper", "research", "new", "learning", "learn", "based",
}


def topic_terms(text: str) -> set[str]:
    terms = set()
    for raw in re.findall(r"[a-z]+", text.lower()):
        if raw.startswith("sketch"):
            term = "sketch"
        elif raw.endswith("ies") and len(raw) > 5:
            term = raw[:-3] + "y"
        elif raw.endswith("s") and len(raw) > 5:
            term = raw[:-1]
        else:
            term = raw
        if len(term) >= 4 and term not in STOP:
            terms.add(term)
    return terms


class PaperIndex:
    def __init__(self) -> None:
        self.page_tokens = {d["id"]: BASELINE._tokens(d["text"]) for d in DOCS}
        self.pages_by_paper: dict[str, list[dict]] = defaultdict(list)
        for doc in DOCS:
            self.pages_by_paper[doc["source"]].append(doc)
        # Conference and preprint PDFs can be copies of the same paper.
        self.title_by_source = TITLE_BY_SOURCE

        self.graph = nx.DiGraph()
        self.topic_papers: dict[str, set[str]] = defaultdict(set)
        for source, pages in self.pages_by_paper.items():
            paper_node = f"paper:{source}"
            self.graph.add_node(paper_node, node_type="paper")
            for page in pages:
                self.graph.add_node(page["id"], node_type="page")
                self.graph.add_edge(page["id"], paper_node, edge_type="contains_page")

            first_page = min(pages, key=lambda p: p["page"])["text"]
            title_line = first_page.splitlines()[0] if first_page else ""
            for term in topic_terms(title_line):
                topic_node = f"topic:{term}"
                self.graph.add_node(topic_node, node_type="topic")
                self.graph.add_edge(paper_node, topic_node, edge_type="about_topic")
                self.topic_papers[term].add(source)

    def score(self, query: str, doc_id: str) -> float:
        return float(len(BASELINE._tokens(query) & self.page_tokens[doc_id]))

    def keyword(self, query: str, k: int = FINAL_K) -> list[str]:
        # Same set-overlap score and stable corpus-order tie break as Checkpoint 2.1.
        query_tokens = BASELINE._tokens(query)
        scored = [
            (d["id"], len(query_tokens & self.page_tokens[d["id"]]))
            for d in DOCS
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [doc_id for doc_id, score in scored[:k] if score > 0]

    def unique_pages(self, pages: list[str], k: int) -> list[str]:
        """Allow distinct pages, but not the same PDF position of a duplicate paper."""
        selected, seen = [], set()
        for doc_id in pages:
            doc = DOC_BY_ID[doc_id]
            key = (self.title_by_source[doc["source"]], doc["page"])
            if key not in seen:
                selected.append(doc_id)
                seen.add(key)
            if len(selected) == k:
                break
        return selected

    def graph_expand(self, query: str, seeds: list[str], k: int = FINAL_K) -> list[str]:
        """Keep seed pages, then add adjacent pages or pages from title-linked papers."""
        selected = self.unique_pages(list(dict.fromkeys(seeds)), k)
        candidates: dict[str, float] = {}

        # A page-to-paper-to-page hop gives local section context.
        for seed_id in selected:
            seed = DOC_BY_ID[seed_id]
            for page in self.pages_by_paper[seed["source"]]:
                distance = abs(page["page"] - seed["page"])
                if 0 < distance <= 2:
                    candidates[page["id"]] = max(
                        candidates.get(page["id"], 0.0),
                        self.score(query, page["id"]) + 0.5 / distance,
                    )

        # A query-to-topic-to-paper-to-page hop can discover a named paper even
        # when no page from that paper appeared in the keyword baseline top five.
        for term in topic_terms(query):
            sources = self.topic_papers.get(term, set())
            if not sources or len(sources) > 20:
                continue  # Very common title terms create noisy links.
            bonus = 2.0 / len(sources)
            for source in sources:
                best_pages = sorted(
                    self.pages_by_paper[source],
                    key=lambda page: self.score(query, page["id"]),
                    reverse=True,
                )[:2]
                for page in best_pages:
                    candidates[page["id"]] = max(
                        candidates.get(page["id"], 0.0),
                        self.score(query, page["id"]) + bonus,
                    )

        ranked = sorted(candidates, key=lambda doc_id: candidates[doc_id], reverse=True)
        return self.unique_pages(selected + ranked, k)


DECOMPOSE_SYSTEM = (
    "Split the research-paper question into 2-4 focused search queries covering "
    "its distinct evidence needs. Use the user's terms; do not invent paper titles "
    "or facts. Return only a JSON array of strings."
)


def decompose(llm, query: str) -> list[str]:
    response = llm.invoke(
        [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    )
    raw = response.content.strip()
    if raw.startswith("```") and "[" in raw and "]" in raw:
        raw = raw[raw.find("["): raw.rfind("]") + 1]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed and all(isinstance(s, str) and s.strip() for s in parsed):
            return parsed[:4]
    except (ValueError, TypeError):
        pass
    return [query]


def decomposed_pages(index: PaperIndex, subqueries: list[str], k: int = FINAL_K) -> list[str]:
    """Reserve one best page per facet, then fill remaining slots by summed score."""
    first_for_facet: list[str] = []
    scores: dict[str, float] = defaultdict(float)
    for subquery in subqueries:
        hits = index.keyword(subquery, 3 * k)
        if hits:
            first_for_facet.append(hits[0])
        for doc_id in hits:
            scores[doc_id] += index.score(subquery, doc_id)
    return index.unique_pages(
        list(dict.fromkeys(first_for_facet)) + sorted(scores, key=scores.get, reverse=True), k
    )


def answer_from_pages(llm, query: str, pages: list[str], primary_count: int) -> str:
    context = "\n\n".join(
        (
            f"[{doc_id}] {'PRIMARY' if pos < primary_count else 'RELATED CONTEXT'} "
            f"Paper: {DOC_BY_ID[doc_id]['source']}, "
            f"PDF page: {DOC_BY_ID[doc_id]['page']}\n"
            f"{DOC_BY_ID[doc_id]['text']}"
        )
        for pos, doc_id in enumerate(pages)
    )
    messages = [
        SystemMessage(content=BASELINE.ANSWER_SYSTEM + " Cite each substantive claim using the exact [document ID] of the supporting page. A citation must support that claim, not merely mention the topic. Do not attribute one paper's limitations to another. If a requested part is unsupported, say so."),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


SOURCE_JUDGE_SYSTEM = (
    "Check source support, independently of whether the answer addresses the question. "
    "Each substantive factual claim must have a cited [document ID] whose supplied page "
    "supports that exact claim. Reject uncited claims, citations to absent pages, "
    "attributing a different paper's limitations to the named method, and invented quotes. "
    "When a question names specific papers, a claim about one of those papers must cite "
    "that paper's own page, not a secondary paper that merely describes it. "
    "Return JSON only: {\"verdict\": \"pass\" or \"fail\", \"reason\": \"brief reason\"}."
)


def check_source_support(llm, query: str, answer: str, pages: list[str]) -> dict[str, str]:
    # Match only our document-ID citation format. Broad square-bracket matching
    # incorrectly treated quoted editorial spellings such as "learn[s]" as a
    # citation named "s".
    cited = re.findall(r"\[(DBLP_[^\[\]\s]+_page_\d+)\]", answer)
    valid = set(pages)
    if not cited or any(doc_id not in valid for doc_id in cited):
        return {"verdict": "fail", "reason": "Missing citation or citation not in retrieved pages."}
    # The LLM judge may accept a secondary paper's description of a named paper.
    # For explicit title comparisons, require a cited page from each named paper.
    named_titles = {title for title in TITLE_BY_SOURCE.values() if len(title) >= 24 and title in query.casefold()}
    cited_sources = {DOC_BY_ID[doc_id]["source"] for doc_id in cited}
    cited_titles = {TITLE_BY_SOURCE[source] for source in cited_sources if source in TITLE_BY_SOURCE}
    if len(named_titles) >= 2 and not named_titles.issubset(cited_titles):
        missing = sorted(named_titles - cited_titles)
        return {"verdict": "fail", "reason": f"No direct citation to named paper(s): {missing}"}
    sources = "\n\n".join(
        f"[{doc_id}] PDF: {DOC_BY_ID[doc_id]['source']}\n{DOC_BY_ID[doc_id]['text']}"
        for doc_id in dict.fromkeys(cited)
    )
    raw = llm.invoke([
        SystemMessage(content=SOURCE_JUDGE_SYSTEM),
        HumanMessage(content=f"QUESTION:\n{query}\n\nANSWER:\n{answer}\n\nCITED PAGES:\n{sources}"),
    ]).content.strip()
    if raw.startswith("```"):
        raw = raw[raw.find("{"):raw.rfind("}") + 1]
    try:
        result = json.loads(raw)
        if result.get("verdict") in {"pass", "fail"}:
            return {"verdict": result["verdict"], "reason": str(result.get("reason", ""))}
    except (ValueError, TypeError, AttributeError):
        pass
    return {"verdict": "fail", "reason": f"Could not parse source check: {raw[:200]}"}


def hit_records(index: PaperIndex, query: str, pages: list[str]) -> list[dict]:
    return [
        {
            "id": doc_id,
            "paper": DOC_BY_ID[doc_id]["source"],
            "pdf_position": DOC_BY_ID[doc_id]["page"],
            "keyword_score": index.score(query, doc_id),
        }
        for doc_id in pages
    ]


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    llm = BASELINE.make_llm()
    index = PaperIndex()
    cases = CP31.my_eval_set()
    print(f"Corpus: {len(index.pages_by_paper)} PDFs, {len(DOCS)} pages")
    print(f"Graph: {index.graph.number_of_nodes()} nodes, {index.graph.number_of_edges()} edges")
    print(f"Final context budget: {FINAL_K} pages per mode\n")
    results = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "corpus_pdfs": len(index.pages_by_paper),
        "corpus_pages": len(DOCS),
        "final_k": FINAL_K,
        "cases": [],
    }

    for number, case in enumerate(cases, start=1):
        query = case["question"]
        subqueries = decompose(llm, query)
        baseline = index.keyword(query, FINAL_K)
        decomposition = decomposed_pages(index, subqueries, FINAL_K)
        graph_seeds = index.unique_pages(index.keyword(query, 3 * FINAL_K), GRAPH_SEEDS)
        graph = index.graph_expand(query, graph_seeds, FINAL_K)
        both = index.graph_expand(query, decomposition[:DECOMP_SEEDS], FINAL_K)
        modes = {
            "baseline": (baseline, len(baseline)),
            "decomposition": (decomposition, len(decomposition)),
            "graph": (graph, min(GRAPH_SEEDS, len(graph))),
            "both": (both, min(DECOMP_SEEDS, len(both))),
        }
        case_result = {"question": query, "grading_notes": case["grading_notes"],
                       "subqueries": subqueries, "modes": {}}
        print(f"Q{number}: {query}")
        print(f"  subqueries: {subqueries}")
        for mode, (pages, primary_count) in modes.items():
            hits = hit_records(index, query, pages)
            hit_labels = [f"{hit['paper']}#{hit['pdf_position']}" for hit in hits]
            print(f"  {mode}: {hit_labels}")
            answer = answer_from_pages(llm, query, pages, primary_count)
            verdict = CP31.judge(llm, answer, case["grading_notes"])
            source_check = check_source_support(llm, query, answer, pages)
            print(f"    answer={verdict.upper()}, source={source_check['verdict'].upper()}: {source_check['reason']}")
            case_result["modes"][mode] = {
                "hits": hits,
                "answer": answer,
                "verdict": verdict,
                "source_support": source_check,
            }
        results["cases"].append(case_result)
        RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nPass counts:")
    for mode in ("baseline", "decomposition", "graph", "both"):
        passes = sum(case["modes"][mode]["verdict"] == "pass" for case in results["cases"])
        supported = sum(case["modes"][mode]["source_support"]["verdict"] == "pass" for case in results["cases"])
        print(f"  {mode}: answer {passes}/{len(cases)}, source support {supported}/{len(cases)}")
    print(f"Saved detailed results: {RESULT_PATH}")


def recheck_saved_results() -> None:
    """Recheck the same generated answers so only the source rubric changes."""
    prior_path = HERE / "checkpoint_4_1_real_papers_results_v2.json"
    results = json.loads(prior_path.read_text(encoding="utf-8"))
    llm = BASELINE.make_llm()
    for number, case in enumerate(results["cases"], start=1):
        for mode, record in case["modes"].items():
            pages = [hit["id"] for hit in record["hits"]]
            record["source_support"] = check_source_support(
                llm, case["question"], record["answer"], pages
            )
            print(f"Q{number} {mode}: answer={record['verdict']}, source={record['source_support']['verdict']}")
        RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved rechecked results: {RESULT_PATH}")


if __name__ == "__main__":
    if "--recheck" in sys.argv:
        recheck_saved_results()
    else:
        run()
