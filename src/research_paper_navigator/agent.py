r"""Agentic RAG experiment for the Research Paper Navigator capstone.

The fixed baseline retrieves five keyword-matched PDF pages once. The agent can
instead choose title-aware retrieval, hybrid retrieval, nearby-page expansion,
or answering. A refined policy requires direct evidence for every named paper
and every part of the question before stopping.

Run the final organized application from the repository root:
    python -m research_paper_navigator --chat
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

from .advanced_retrieval import (
    BASELINE,
    CP31,
    DOCS,
    DOC_BY_ID,
    FIRST_PAGE_BY_SOURCE,
    PaperIndex,
    TITLE_BY_SOURCE,
    answer_from_pages,
    check_source_support,
)
from .hybrid_retrieval import (
    HybridPaperIndex,
    embedding_client,
    paper_embeddings,
)

RESULT_PATH = OUTPUT_DIR / "agent_results.json"
MAX_ITERATIONS = 3
MAX_CONTEXT_PAGES = 8

INITIAL_PLANNER = """You control retrieval for a research-paper question.
Choose one action and return JSON only:
{"action":"title_search|year_search|hybrid_search|expand_pages|answer",
 "query":"focused retrieval query", "start_year":2018, "end_year":2020,
 "reasoning":"brief reason"}

Use title_search when the question names a paper. Use hybrid_search for conceptual
or paraphrased needs. Use year_search when publication years or a year range are
requested. Use expand_pages when a useful page may need surrounding context. Answer
when the collected pages appear sufficient. Do not repeat actions."""

REFINED_PLANNER = """You control retrieval for a research-paper question.
Choose one action and return JSON only:
{"action":"title_search|year_search|hybrid_search|expand_pages|answer",
 "query":"focused retrieval query", "start_year":2018, "end_year":2020,
 "reasoning":"brief reason"}

Evidence policy:
- Break the question into evidence needs, such as each named paper, benefits,
  limitations, methods, results, or comparisons.
- A named paper is covered only by pages from that paper itself; a secondary
  paper describing it is not direct evidence.
- Do not answer while any requested part lacks direct supporting evidence.
- Use title_search for named papers, year_search for an explicit publication-year
  constraint, hybrid_search for a missing concept, and expand_pages when the right
  paper was found but the needed section was not.
- Never repeat an executed action/query. If the retrieval limit is reached, the
  system will answer only supported parts and disclose missing evidence."""


@dataclass
class StepRecord:
    iteration: int
    action: str
    query: str
    reasoning: str
    added_pages: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    question: str
    policy: str
    answer: str
    pages: list[str]
    steps: list[StepRecord]
    elapsed_seconds: float
    answer_verdict: str = ""
    source_verdict: str = ""
    source_reason: str = ""


class ResearchPaperAgent:
    def __init__(self, refined: bool, max_iterations: int = MAX_ITERATIONS) -> None:
        self.refined = refined
        self.max_iterations = max_iterations
        self.llm = BASELINE.make_llm()
        self.embedder = embedding_client()
        self.paper_index = PaperIndex()
        self.hybrid = HybridPaperIndex(paper_embeddings(self.embedder))
        self.doc_position = {doc["id"]: pos for pos, doc in enumerate(DOCS)}

    @property
    def planner_prompt(self) -> str:
        return REFINED_PLANNER if self.refined else INITIAL_PLANNER

    def _page_summary(self, pages: list[str]) -> str:
        if not pages:
            return "(none)"
        return "\n\n---\n\n".join(
            f"[{doc_id}] PDF={DOC_BY_ID[doc_id]['source']} "
            f"position={DOC_BY_ID[doc_id]['page']}\n"
            f"{DOC_BY_ID[doc_id]['text'][:1800]}"
            for doc_id in pages[:MAX_CONTEXT_PAGES]
        )

    def _plan(self, question: str, pages: list[str], steps: list[StepRecord]) -> dict:
        executed = "\n".join(
            f"- {step.action}: {step.query}" for step in steps
        ) or "(none)"
        prompt = (
            f"Question: {question}\n\n"
            f"Actions already executed:\n{executed}\n\n"
            f"Collected pages ({len(pages)}):\n{self._page_summary(pages)}"
        )
        raw = self.llm.invoke([
            SystemMessage(content=self.planner_prompt),
            HumanMessage(content=prompt),
        ]).content.strip()
        # Some models wrap valid JSON in a code fence or a short explanation.
        # Extract the outer JSON object before parsing either form.
        json_start = raw.find("{")
        json_end = raw.rfind("}")
        if json_start >= 0 and json_end > json_start:
            raw = raw[json_start:json_end + 1]
        try:
            result = json.loads(raw)
            action = result.get("action", "answer")
            if action not in {"title_search", "year_search", "hybrid_search", "expand_pages", "answer"}:
                action = "answer"
            return {
                "action": action,
                "query": str(result.get("query", "") or question),
                "start_year": result.get("start_year"),
                "end_year": result.get("end_year"),
                "reasoning": str(result.get("reasoning", "")),
            }
        except (ValueError, TypeError, AttributeError):
            return {"action": "answer", "query": question,
                    "start_year": None, "end_year": None,
                    "reasoning": "Planner response could not be parsed; stopping safely."}

    def _named_sources(self, text: str) -> list[str]:
        lowered = text.casefold()
        matches = [source for source, title in TITLE_BY_SOURCE.items()
                   if len(title) >= 20 and title in lowered]
        # Conference and preprint copies can have the same title. Retain one copy.
        seen_titles, unique_sources = set(), []
        for source in matches:
            title = TITLE_BY_SOURCE[source]
            if title not in seen_titles:
                seen_titles.add(title)
                unique_sources.append(source)
        return unique_sources

    @staticmethod
    def _publication_year(source: str) -> int | None:
        """Read the DBLP/arXiv year encoded in the corpus filename."""
        arxiv = re.search(r"_abs-(\d{2})\d{2}-", source)
        if arxiv:
            return 2000 + int(arxiv.group(1))
        suffix = re.search(r"(\d{2})\.pdf$", source)
        if suffix:
            short_year = int(suffix.group(1))
            return 2000 + short_year if short_year <= 30 else 1900 + short_year
        return None

    def _title_search(self, original_question: str, query: str, k: int = 5) -> list[str]:
        sources = self._named_sources(query) or self._named_sources(original_question)
        if not sources:
            return []
        raw_scores = self.hybrid.bm25.get_scores(
            re.findall(r"[a-z0-9]+", query.casefold())
        )
        results: list[str] = []
        for source in sources:
            pages = self.paper_index.pages_by_paper[source]
            ranked = sorted(
                pages,
                key=lambda page: raw_scores[self.doc_position[page["id"]]],
                reverse=True,
            )
            # Preserve the title/abstract page plus the best evidence page.
            results.append(FIRST_PAGE_BY_SOURCE[source]["id"])
            results.extend(page["id"] for page in ranked[:2])
        return self.paper_index.unique_pages(results, k)

    def _hybrid_search(self, query: str, k: int = 5) -> list[str]:
        vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        pages, _ = self.hybrid.retrieve([query], vector[None, :], dynamic=False)
        return pages[:k]

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        low, high = min(values.values()), max(values.values())
        if high == low:
            return {key: 0.5 for key in values}
        return {key: (value - low) / (high - low) for key, value in values.items()}

    def _year_search(self, query: str, start_year: int, end_year: int,
                     k: int = 5) -> list[str]:
        """Hybrid page search restricted by structured publication year."""
        if end_year < start_year:
            start_year, end_year = end_year, start_year
        eligible_sources = {
            source for source in self.paper_index.pages_by_paper
            if (year := self._publication_year(source)) is not None
            and start_year <= year <= end_year
        }
        if not eligible_sources:
            return []

        query_tokens = re.findall(r"[a-z0-9]+", query.casefold())
        raw_bm25 = self.hybrid.bm25.get_scores(query_tokens)
        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        paper_vectors = self.hybrid.vectors
        semantic = (paper_vectors @ query_vector) / (
            np.linalg.norm(paper_vectors, axis=1) * np.linalg.norm(query_vector) + 1e-12
        )

        candidates = [
            doc for doc in DOCS if doc["source"] in eligible_sources
        ]
        bm25_scores = {
            doc["id"]: float(raw_bm25[self.doc_position[doc["id"]]])
            for doc in candidates
        }
        semantic_scores = {
            doc["id"]: float(semantic[self.hybrid.source_position[doc["source"]]])
            for doc in candidates
        }
        bm25_norm = self._normalize(bm25_scores)
        semantic_norm = self._normalize(semantic_scores)
        fused = {
            doc["id"]: 0.5 * bm25_norm[doc["id"]] + 0.5 * semantic_norm[doc["id"]]
            for doc in candidates
        }
        ranked = sorted(fused, key=fused.get, reverse=True)
        return self.paper_index.unique_pages(ranked, k)

    def _expand_pages(self, question: str, pages: list[str], k: int = 5) -> list[str]:
        candidates: dict[str, float] = {}
        for doc_id in pages:
            seed = DOC_BY_ID[doc_id]
            for page in self.paper_index.pages_by_paper[seed["source"]]:
                distance = abs(page["page"] - seed["page"])
                if 0 < distance <= 2:
                    candidates[page["id"]] = max(
                        candidates.get(page["id"], 0.0),
                        self.paper_index.score(question, page["id"]) + 1.0 / distance,
                    )
        ranked = sorted(candidates, key=candidates.get, reverse=True)
        return [doc_id for doc_id in ranked if doc_id not in pages][:k]

    def _answer(self, question: str, pages: list[str]) -> str:
        if not self.refined:
            return answer_from_pages(self.llm, question, pages, len(pages))
        context = "\n\n".join(
            f"[{doc_id}] Paper: {DOC_BY_ID[doc_id]['source']}, "
            f"PDF position: {DOC_BY_ID[doc_id]['page']}\n{DOC_BY_ID[doc_id]['text']}"
            for doc_id in pages
        )
        system = (
            BASELINE.ANSWER_SYSTEM
            + " Cite every substantive claim with the exact [document ID]. "
            "Answer each requested part separately. When asked for limitations of a "
            "named method, do not present limitations of prior systems or comparison "
            "baselines as limitations of that method. Distinguish direct reported "
            "evidence from your inference. If the pages do not establish a requested "
            "limitation, say that evidence is missing rather than inventing one."
        )
        return self.llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}"),
        ]).content

    def run(self, question: str) -> AgentResult:
        started = time.perf_counter()
        pages: list[str] = []
        steps: list[StepRecord] = []

        for iteration in range(1, self.max_iterations + 1):
            decision = self._plan(question, pages, steps)
            action = decision["action"]
            query = decision["query"]
            explicit_years = [
                int(year) for year in re.findall(r"(?:19|20)\d{2}", question)
            ]
            # An explicit date constraint is objective metadata, so enforce it
            # before semantic retrieval instead of relying on planner wording.
            if self.refined and iteration == 1 and len(explicit_years) >= 2:
                action = "year_search"
                decision["start_year"] = min(explicit_years)
                decision["end_year"] = max(explicit_years)
                decision["reasoning"] = (
                    "The question contains an explicit publication-year range; "
                    "apply the structured year filter before semantic ranking."
                )
            if self.refined and action == "answer" and iteration <= self.max_iterations:
                reasoning = decision["reasoning"].casefold()
                missing_signals = ("missing", "do not yet", "not yet", "still need", "need the")
                if any(signal in reasoning for signal in missing_signals):
                    action = "expand_pages" if pages else "hybrid_search"
                    decision["reasoning"] += " [Guard changed premature answer to retrieval.]"
            if action == "answer":
                steps.append(StepRecord(iteration, action, query, decision["reasoning"], []))
                break
            if action == "title_search":
                found = self._title_search(question, query)
            elif action == "year_search":
                years = [int(year) for year in re.findall(r"(?:19|20)\d{2}", question)]
                start_year = decision.get("start_year")
                end_year = decision.get("end_year")
                try:
                    start_year = int(start_year) if start_year is not None else min(years)
                    end_year = int(end_year) if end_year is not None else max(years)
                    found = self._year_search(query, start_year, end_year)
                    query = f"{query} [years {start_year}-{end_year}]"
                except (TypeError, ValueError):
                    found = []
            elif action == "hybrid_search":
                found = self._hybrid_search(query)
            else:
                found = self._expand_pages(query or question, pages)

            before = set(pages)
            pages = self.paper_index.unique_pages(pages + found, MAX_CONTEXT_PAGES)
            added = [doc_id for doc_id in pages if doc_id not in before]
            steps.append(StepRecord(iteration, action, query, decision["reasoning"], added))
            print(f"  {iteration}. {action}: added {len(added)} page(s) — {decision['reasoning']}")

        if not pages:
            # A planner that answers before searching must not receive an empty context.
            pages = self.paper_index.keyword(question, 5)

        answer = self._answer(question, pages)
        return AgentResult(
            question=question,
            policy="refined" if self.refined else "initial",
            answer=answer,
            pages=pages,
            steps=steps,
            elapsed_seconds=round(time.perf_counter() - started, 2),
        )


def baseline_result(llm, index: PaperIndex, question: str) -> AgentResult:
    started = time.perf_counter()
    pages = index.keyword(question, 5)
    answer = answer_from_pages(llm, question, pages, len(pages))
    return AgentResult(question, "baseline", answer, pages, [],
                       round(time.perf_counter() - started, 2))


def evaluate_result(llm, result: AgentResult, grading_notes: str) -> None:
    result.answer_verdict = CP31.judge(llm, result.answer, grading_notes)
    support = check_source_support(llm, result.question, result.answer, result.pages)
    result.source_verdict = support["verdict"]
    result.source_reason = support["reason"]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = CP31.my_eval_set()
    explicit_q5 = dict(cases[4])
    explicit_q5["question"] = (
        "What evidence does Learning to Infer Program Sketches provide about "
        "the benefits and limitations of automatically inferred sketches?"
    )
    year_case = {
        "question": (
            "Find papers published between 2018 and 2020 about automatically inferred "
            "program sketches, then summarize their reported benefits and limitations."
        ),
        "grading_notes": (
            "Identifies a paper from 2018 through 2020 that directly concerns automatically "
            "inferred program sketches, and summarizes supported benefits and limitations "
            "or clearly states when the retrieved paper does not report a direct limitation."
        ),
    }
    if "--q5-only" in sys.argv:
        selected = [explicit_q5]
    elif "--year-only" in sys.argv:
        selected = [year_case]
    else:
        selected = [cases[2], explicit_q5, year_case]
    llm = BASELINE.make_llm()
    index = PaperIndex()
    initial = ResearchPaperAgent(refined=False)
    refined = ResearchPaperAgent(refined=True)
    output = {"timestamp": datetime.now().isoformat(timespec="seconds"), "cases": []}

    for case in selected:
        question = case["question"]
        print(f"\nQUESTION: {question}")
        records = [
            baseline_result(llm, index, question),
            initial.run(question),
            refined.run(question),
        ]
        for record in records:
            evaluate_result(llm, record, case["grading_notes"])
            print(
                f"  {record.policy}: answer={record.answer_verdict}, "
                f"source={record.source_verdict}, pages={len(record.pages)}, "
                f"time={record.elapsed_seconds}s"
            )
        output["cases"].append({
            "question": question,
            "grading_notes": case["grading_notes"],
            "results": [asdict(record) for record in records],
        })
        RESULT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved detailed results to {RESULT_PATH}")


if __name__ == "__main__":
    main()
