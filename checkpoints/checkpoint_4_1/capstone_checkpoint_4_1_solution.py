r"""Capstone Checkpoint 4.1 — Advanced Retrieval Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 4.1 — Advanced Retrieval Implementation
# **MO-LLM Module 4/Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This checkpoint advances your capstone system from a *single-pass* retriever to an
# **advanced retriever**. You will use the baseline retrieval system you built in
# Checkpoint 2.1 and evaluated in Checkpoint 3.1 to identify where single-pass retrieval
# falls short and apply the Module 4 techniques to fix it:
#
# - **Lab 4.1 — multistep retrieval (query decomposition):** Rewrite a complex
#   question into focused sub-queries, retrieve for each, and merge the results.
# - **Lab 4.2 — graph-based retrieval:** Organize your corpus as a graph of
#   relationships (citations, shared topics, links, same author/entity) and pull in
#   *related* documents a keyword/vector search would miss.
#
# You then **measure the improvement** against your 3.1 baseline. The graded
# deliverable is a **500-750 word written report** (final section). This script is a
# runnable demonstration of both techniques on a tiny sample corpus so you can see
# them work before adapting them to your real system.
#
# **Learning outcomes (Module 4):**
# 1. Identify common retrieval failures — e.g., complex multistep questions and too
#    much loosely-related context degrading the answer.
# 2. Explain *why* those failures occur with single-pass retrieval.
# 3. Implement an advanced retrieval strategy (query decomposition and/or graph
#    traversal) that addresses them.
# 4. Measure the improvement against your baseline using your 3.1 evaluation set.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Natural relationships to exploit in a graph |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | citations, shared authors, shared topics/keywords, "published-before" |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | hyperlinks between articles, shared categories, mentioned entities |
#
# Multistep decomposition shines on cross-document questions ("Compare X and Y,"
# "how did an idea evolve"); graph retrieval shines when the answer needs documents
# that are *related* to a hit but don't themselves match the query terms.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv networkx`
# 3. Use the OpenRouter API key provided for this course. This checkpoint uses
#   the `openai/gpt-5.4-mini` model, with usage covered by course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This script runs on a tiny built-in sample corpus, so you do **not** need your full
# dataset indexed to complete it. You *will* refer to your real 2.1 system and 3.1
# evaluation results when you write the report.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # Latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
LOG_PATH = Path.cwd() / "checkpoint_4_1_responses.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECOMPOSE_SYSTEM = (
    "You are a query decomposition assistant for a document retrieval system. "
    "Break the user's question into 2-4 focused sub-queries that together cover "
    "everything needed to answer it. Each sub-query should target a distinct aspect. "
    'Return ONLY a JSON array of strings, e.g., ["sub-query 1", "sub-query 2"].'
)

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at "
            "https://openrouter.ai/keys, put it in a .env file next to this "
            "script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )


def log_response(label: str, prompt: str, response: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    entry = (
        f"[{ts}]  {label}  SCENARIO={SCENARIO}  MODEL={LLM_MODEL}\n"
        f"PROMPT:   {prompt}\n"
        f"RESPONSE: {response}\n"
        f"{'-' * 72}\n"
    )
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(entry)


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# Six short "documents" with relationships so both techniques are demonstrable
# without indexing your full corpus. Each has an ID, text, a `topics` list, and
# `links` to related documents (i.e., think citations for papers, or hyperlinks for
# Wikipedia). Your real system would derive these from your actual data.

# %%
SAMPLE_DOCS = [
    {"id": "d1", "text": "Paper on retrieval-augmented generation: Grounding LLM answers in retrieved documents reduces hallucination.",
     "topics": ["RAG", "hallucination"], "links": ["d2", "d3"]},
    {"id": "d2", "text": "Study of hallucination in language models: Models fabricate specifics when they lack grounding.",
     "topics": ["hallucination"], "links": ["d1"]},
    {"id": "d3", "text": "Hybrid retrieval combines keyword and vector search to improve recall over either alone.",
     "topics": ["RAG", "retrieval"], "links": ["d1", "d4"]},
    {"id": "d4", "text": "Query decomposition breaks a complex question into sub-queries, improving multi-aspect retrieval.",
     "topics": ["retrieval", "decomposition"], "links": ["d3"]},
    {"id": "d5", "text": "Graph-based retrieval traverses relationships between documents to add related context.",
     "topics": ["retrieval", "graph"], "links": ["d4", "d6"]},
    {"id": "d6", "text": "Evaluation of RAG systems uses an LLM judge to score answers against grading notes.",
     "topics": ["evaluation", "RAG"], "links": ["d5"]},
]
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def keyword_score(query: str, text: str) -> float:
    """A tiny, dependency-free relevance score: shared word count. This stands in for
    your real hybrid retriever, so the retrieval step needs no extra dependencies. The
    demo still calls the LLM, which requires an OpenRouter API key."""
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    t = set(re.findall(r"[a-z0-9]+", text.lower()))
    return float(len(q & t))


def baseline_retrieve(query: str, k: int = 3) -> list[tuple[str, float]]:
    """Single-pass retrieval: Score every doc once, take the top k. (id, score)."""
    scored = [(d["id"], keyword_score(query, d["text"])) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in scored[:k] if s > 0]


# %% [markdown]
# ## Step 2 — Multistep retrieval (provided, adapted from Lab 4.1)
#
# `decompose_query` asks the LLM to split the question into sub-queries. For each
# sub-query, we retrieve the top documents and **sum** each document's score across
# the sub-queries. A document relevant to several parts of the question rises to the
# top. (In your real system, replace `baseline_retrieve` with your 2.1 retriever.)

# %%
def decompose_query(llm: ChatOpenAI, query: str) -> list[str]:
    messages = [SystemMessage(content=DECOMPOSE_SYSTEM), HumanMessage(content=query)]
    raw = llm.invoke(messages).content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("["): raw.rfind("]") + 1] if "[" in raw else raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed) and parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return [query]


def multistep_retrieve(llm: ChatOpenAI, query: str, k: int = 3) -> list[str]:
    sub_queries = decompose_query(llm, query)
    print(f"  decomposed into: {sub_queries}")
    score_map: dict[str, float] = {}
    for sq in sub_queries:
        for doc_id, score in baseline_retrieve(sq, 3 * k):
            score_map[doc_id] = score_map.get(doc_id, 0.0) + score
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k]]


# %% [markdown]
# ## Step 3 — Graph-based retrieval (provided, adapted from Lab 4.2)
#
# Build a graph whose nodes are documents and topics, with edges for "links to" and
# "relates to topic." Retrieval seeds with the baseline hits, then pulls in each
# seed's linked documents and topic-siblings as extra context — deduplicated.

# %%
def build_graph(docs: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for d in docs:
        G.add_node(f"doc:{d['id']}", node_type="doc")
        for other in d["links"]:
            G.add_edge(f"doc:{d['id']}", f"doc:{other}", edge_type="links_to")
        for topic in d["topics"]:
            G.add_node(f"topic:{topic}", node_type="topic")
            G.add_edge(f"doc:{d['id']}", f"topic:{topic}", edge_type="relates_to")
    return G


def graph_retrieve(
    graph: nx.DiGraph,
    query: str,
    k: int = 3,
    seed_ids: list[str] | None = None,
) -> dict[str, str]:
    """Expand baseline seeds, or supplied decomposition seeds, through the graph."""
    union: dict[str, str] = {}
    if seed_ids is None:
        seed_ids = [doc_id for doc_id, _ in baseline_retrieve(query, k)]
    for doc_id in seed_ids:
        union[doc_id] = "seed"
    for seed in list(union):
        node = f"doc:{seed}"
        if not graph.has_node(node):
            continue
        for _, target, ed in graph.out_edges(node, data=True):
            if ed.get("edge_type") == "links_to":
                union.setdefault(target.removeprefix("doc:"), "linked")
            elif ed.get("edge_type") == "relates_to":
                for sib, _, ed2 in graph.in_edges(target, data=True):  # Source docs on this topic
                    if ed2.get("edge_type") == "relates_to":
                        union.setdefault(sib.removeprefix("doc:"), "topic")
    return union


# %% [markdown]
# ## Step 4 — Your advanced-retrieval plan (TODO)
#
# Design how you will apply these techniques to **your** capstone corpus. Return a
# dictionary with the keys below. This is the plan you will implement in your real system
# and describe in the report. Keep it concrete and specific to your scenario.

# %%
def my_advanced_plan() -> dict[str, Any]:
    """Return YOUR advanced-retrieval plan for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "technique":   "decomposition", "graph", or "both"
      - "node_types":  list[str]  — for graph: the kinds of nodes in your corpus
                       (e.g., ["paper", "author", "topic"] or ["article", "category"])
      - "edge_types":  list[str]  — the relationships you will surface
                       (e.g., ["cites", "same_author", "shares_topic"])
      - "test_queries": list[str] — 2-3 questions from your 3.1 evaluation set that
                       single-pass retrieval handled poorly and that should improve.
      - "rationale":   str — one or two sentences on why these choices fit your data.

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        "technique": "both",
        "comparison_modes": ["decomposition", "graph", "both"],
        "node_types": ["paper", "page", "topic"],
        "edge_types": ["contains_page", "about_topic"],
        "test_queries": [
            "What evidence do the papers provide about the benefits and limitations of automatically inferred program sketches?",
            "What role does the programmer play in the sketching approach to program synthesis?",
            "How do The Sketching Approach to Program Synthesis and Learning to Infer Program Sketches differ in how program sketches are produced?",
        ],
        "rationale": (
            "Compare decomposition-only, graph-only, and combined retrieval against "
            "the same page-level keyword baseline and evaluation questions. Decomposition "
            "may separate benefit, limitation, and cross-paper evidence needs; a graph "
            "linking pages to papers and papers to topics may add relevant context that "
            "a single keyword query misses. Keep the final page budget comparable to "
            "the baseline and check whether each cited page supports its claim."
        ),
    }


# %% [markdown]
# ## Step 5 — Run baseline vs. advanced and capture the evidence
#
# This demonstration compares single-pass, decomposition-only, graph-only, and
# decomposition-plus-graph retrieval on the six sample documents. The graph may
# expand to most or all of this tiny corpus; these are mechanics, not capstone scores.

# %%
def answer_from_docs(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


def run_demo() -> None:
    llm = make_llm()
    graph = build_graph(SAMPLE_DOCS)
    query = "How does breaking a question into parts and following document links help retrieval?"

    print("=" * 72)
    print(f"Checkpoint 4.1 demo — scenario: {SCENARIO}\nQuery: {query}\n")

    base = [i for i, _ in baseline_retrieve(query, 3)]
    print(f"BASELINE (single-pass) retrieved: {base}")

    multi = multistep_retrieve(llm, query, 3)
    print(f"MULTI-STEP retrieved:             {multi}")

    g = graph_retrieve(graph, query, 3)
    print(f"GRAPH retrieved (with sources):   {g}")

    combined = graph_retrieve(graph, query, 3, seed_ids=multi)
    print(f"COMBINED retrieved (with sources): {combined}")

    for label, doc_ids in (
        ("BASELINE", base),
        ("DECOMPOSITION", multi),
        ("GRAPH", list(g)),
        ("BOTH", list(combined)),
    ):
        answer = answer_from_docs(llm, query, doc_ids)
        print(f"\n{label} answer (documents: {doc_ids}):\n{answer}\n")
        log_response(label, query, f"documents={doc_ids}\n{answer}")

    # Show your own plan was filled in.
    try:
        plan = my_advanced_plan()
        print("Your advanced-retrieval plan:")
        print(json.dumps(plan, indent=2))
    except NotImplementedError as e:
        print(f"[my_advanced_plan not done yet] {e}")

    print("=" * 72)
    print("Done. Now reproduce this comparison in YOUR real system (baseline vs "
          "advanced, matched document count) and use the numbers in your report.")


run_demo()

# %% [markdown]
# ## Step 6
#
# The checkpoint deliverable is your completed Capstone Checkpoint 4.1 worksheet, not
# code. Using evidence from your real system, cover:
#
# 1. **System overview** State your scenario and your 2.1 baseline retriever.
# 2. **Failure diagnosis** Describe the meaningful retrieval failure or limitation identified through
#    your Checkpoint 3.1 evaluation and explain why it occurs.
# 3. **Data redesign** Describe the node/edge schema you defined to expose relationships in
#    your corpus (citations/authors/topics, or links/categories/entities), and why you chose it. 
# 4. **Advanced retrieval implemented** Provide a query decomposition and/or graph traversal,
#    with code or log evidence. Label primary vs. context documents in the prompt.
# 5. **Comparative evaluation** Re-run your 3.1 evaluation on baseline vs advanced,
#    **comparing a similar number of retrieved documents**, and report before/after.
# 6. **Analysis & reflection** Discuss where it helped, where it hurt (redundant/over-broad
#    context), the cost/latency trade-off, and remaining limitations.
#
