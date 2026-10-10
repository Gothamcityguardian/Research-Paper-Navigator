r"""Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code/PyCharm/Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 5.1 — Designing and Evaluating an Agent-Based RAG System
# **MO-LLM Module 5 -  Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# This is the final build step of your capstone. You will turn the retrieval system you've
# developed into an **agent-based RAG system**. Instead of a fixed pipeline, an agent
# decides at each step whether it has enough information, what to retrieve next, which
# tool to use, or whether to answer. This activity allows you to apply the Module 5 labs (Lab 5.1's agentic
# retriever and Lab 5.2's tool-using agent) to your capstone scenario.
#
# The graded deliverable is your completed Capstone Checkpoint 5.1 worksheet. This script is a
# runnable demonstration of an agentic loop on a tiny sample corpus so you can see the
# decision-making before adapting it to your full system.
#
# **Learning outcomes (Module 5):**
# 1. Build an agent-based system that integrates retrieval and external tools.
# 2. Use system prompts to guide agent behavior and decision-making.
# 3. Design workflows that coordinate retrieval, reasoning, and tool use in a RAG system.
# 4. Evaluate an agentic workflow against a fixed retrieval pipeline.
# 5. Design workflows that coordinate retrieval, reasoning, and tool use within a RAG system.
# 6. Build an agent-based system that integrates retrieval and external tools to complete user tasks.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1 and have built on since.
#
# | Scenario | Corpus | Useful agent tools/actions |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs | semantic search; "find papers by author/year"; "follow citations"; ask a clarifying question |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles | semantic search; "find by category"; "follow links"; ask a clarifying question |
#
# An agent shines when one query isn't enough — it can search, look at what it found,
# and decide to search again (or use a different tool) before answering.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Get a free OpenRouter key at <https://openrouter.ai/keys>. The labs use the paid
#    `openai/gpt-5.4-mini` model, which is covered by the course credits.
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). The built-in corpus is deliberately minimal: It only demonstrates
# the agent control flow, not retrieval quality, and it is unrelated to your capstone:
# Howerver, your design should target your own capstone corpus. Your full agent is what you describe
# in the written submission.

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

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"
TEMPERATURE = 0.2
MAX_STEPS = 3
LOG_PATH = Path.cwd() / "checkpoint_5_1_agent.log"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

DECIDE_SYSTEM = (
    "You are an agent retrieving from a small document collection. Given the question, "
    "the queries already run, and the documents found so far, decide what to do next. "
    'Respond with ONLY a JSON object: {"done": true|false, "new_queries": ["..."], '
    '"reasoning": "..."}. Set done=true when you have enough to answer; otherwise give '
    "1-2 new_queries targeting what is still missing (do not repeat past queries)."
)
ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided documents, "
    "quoting where you can. If they do not contain the answer, say so."
)


# %%
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Grab a free key at https://openrouter.ai/keys, "
            "put it in a .env file next to this script, and rerun."
        )
    return key


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE,
                      api_key=check_api_key(), base_url=OPENROUTER_BASE_URL)


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus + a keyword retriever (provided)

# %%
SAMPLE_DOCS = [
    {"id": "d1", "text": "Program synthesis generates programs from a specification, such as input-output examples."},
    {"id": "d2", "text": "The sketching approach lets a programmer leave holes in a program for a synthesizer to fill."},
    {"id": "d3", "text": "Retrieval-augmented generation grounds a model's answers in retrieved documents to reduce hallucination."},
    {"id": "d4", "text": "An agentic retriever decides at each step whether it has enough information or should search again."},
    {"id": "d5", "text": "A tool-using agent chooses among actions — search, look up by date, ask the user — to complete a task."},
    {"id": "d6", "text": "Evaluating an agent compares its answers and cost against a fixed single-pass pipeline."},
]
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


def retrieve(query: str, k: int = 2) -> list[str]:
    q = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = [(d["id"], len(q & set(re.findall(r"[a-z0-9]+", d["text"].lower())))) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, s in scored[:k] if s > 0]


def decide(llm: ChatOpenAI, question: str, collected: dict[str, str], executed: list[str]) -> dict:
    docs = "\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in collected) or "(none yet)"
    user = f"Question: {question}\n\nQueries run: {executed or '(none)'}\n\nDocuments so far:\n{docs}"
    raw = llm.invoke([SystemMessage(content=DECIDE_SYSTEM), HumanMessage(content=user)]).content.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        d = json.loads(raw)
        return {"done": bool(d.get("done", True)), "new_queries": d.get("new_queries", []) or [],
                "reasoning": d.get("reasoning", "")}
    except (json.JSONDecodeError, ValueError):
        return {"done": True, "new_queries": [], "reasoning": "parse-fail -> stop"}


def agentic_answer(llm: ChatOpenAI, question: str) -> str:
    """Provided: a minimal agentic loop — retrieve, decide whether to continue, repeat."""
    collected: dict[str, str] = {}
    executed: list[str] = []
    pending = [question]
    for step in range(MAX_STEPS):
        for q in pending:
            for doc_id in retrieve(q):
                collected[doc_id] = DOC_BY_ID[doc_id]["text"]
            executed.append(q)
        d = decide(llm, question, collected, executed)
        print(f"  step {step + 1}: have {sorted(collected)}  -> done={d['done']}  ({d['reasoning'][:60]})")
        if d["done"] or not d["new_queries"]:
            break
        pending = d["new_queries"]
    context = "\n\n".join(f"[{i}] {collected[i]}" for i in collected)
    return llm.invoke([SystemMessage(content=ANSWER_SYSTEM),
                       HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {question}")]).content


# %% [markdown]
# ## Step 2 — Your agent design (TODO)
#
# Design the agent you will build for **your** scenario. Return a dictionary with the keys
# below — this is the plan you implement in your real system and describe in the report.

# %%
def my_agent_plan() -> dict[str, Any]:
    """Return your agent design for your chosen scenario.

    TODO — your turn. Return a dictionary with these keys:
      - "tools": list[str]      — the actions/tools your agent can take (e.g.,
                                  ["semantic_search", "find_by_author", "follow_citation",
                                   "clarify", "answer"]).
      - "stop_condition": str   — how the agent decides it has enough to answer.
      - "system_prompt_idea": str — one or two sentences on how you'll instruct the agent
                                  to choose actions (this is what "guiding agent behavior
                                  with system prompts" means).
      - "test_tasks": list[str] — 2-3 questions for your scenario that need more than one
                                  retrieval step (so the agentic loop earns its keep).

    Example (illustrative — replace with your own scenario):
        return {
            "tools": ["semantic_search", "find_by_author", "clarify", "answer"],
            "stop_condition": "the retrieved passages cover every part of the question",
            "system_prompt_idea": "Search first; clarify only if the request is ambiguous; "
                                  "answer once the retrieved text supports a grounded reply.",
            "test_tasks": [
                "What changed between the v1 and v2 proposals, and who approved it?",
                "Summarize the budget decisions discussed across Q1 and Q2.",
            ],
        }

    Delete the raise NotImplementedError line once your code works.
    """
    return {
        "tools": [
            "clarify",
            "search_by_title",
            "search_by_year",
            "hybrid_page_search",
            "expand_adjacent_pages",
            "check_evidence_coverage",
            "answer",
        ],
        "stop_condition": (
            "Answer when every part of the question has supporting evidence, every "
            "explicitly named paper is represented by its own pages, and the response "
            "can cite the paper and PDF position. After the maximum retrieval rounds, "
            "answer only the supported parts and identify what evidence is still missing."
        ),
        "system_prompt_idea": (
            "Clarify only when the request is ambiguous. Choose title search for named "
            "papers and hybrid search for conceptual or paraphrased questions. After "
            "retrieval, check coverage of every question part before answering; if evidence "
            "is missing, search again or inspect nearby pages without repeating a query."
        ),
        "test_tasks": [
            (
                "How do The Sketching Approach to Program Synthesis and Learning to "
                "Infer Program Sketches differ in how program sketches are produced?"
            ),
            (
                "What evidence does Learning to Infer Program Sketches provide about "
                "the benefits and limitations of automatically inferred sketches?"
            ),
            (
                "Find papers published between 2018 and 2020 about automatically "
                "inferred program sketches, then summarize their reported benefits "
                "and limitations."
            ),
        ],
    }


# %% [markdown]
# ## Step 3 — Run the agentic loop and capture the evidence
#
# Runs the provided agentic loop on a multistep question (watch it retrieve, decide,
# and retrieve again), then prints your plan. Reproduce this in your real system for the
# report and compare it against your fixed Checkpoint 2.1 retriever on the same task.

# %%
def run() -> None:
    llm = make_llm()
    question = "How does an agentic retriever differ from a fixed pipeline, and how is it evaluated?"
    print(f"Checkpoint 5.1 — agentic RAG demo  |  scenario: {SCENARIO}")
    print(f"Question: {question}\n")
    answer = agentic_answer(llm, question)
    print(f"\nAgent answer:\n{answer}\n")
    log("AGENTIC", f"Q: {question}\nA: {answer}")
    try:
        print("Your agent plan:")
        print(json.dumps(my_agent_plan(), indent=2))
    except NotImplementedError as e:
        print(f"[my_agent_plan not done yet] {e}")
    print("=" * 72)
    print("Done. Build this agent for your real system and compare it to your 2.1 baseline.")


run()

# %% [markdown]
# ## Step 4 — Your written submission (the graded deliverable)
#
# The deliverable is a **written submission** (suggested length: 500-750 words). Cover the following:
#
# 1. **System overview**: State your scenario and the retrieval system you built (2.1–4.1).
# 2. **Agent design**: Describe the actions/tools your agent can take, and how a **system prompt**
#    guides it to choose among them (retrieve, use a tool, clarify, or answer).
# 3. **Workflow**: Explain how retrieval, reasoning, and tool use are coordinated across steps,
#    and how the agent decides it has enough information to answer.
# 4. **Evaluation**: Compare the agent-based system against your fixed Checkpoint 2.1/3.1
#    pipeline on a few representative tasks. Did multi-step decision-making improve the
#    answers? At what cost (extra model calls, latency)?
# 5. **Reflection**: When an agentic approach adds value vs. when a simpler pipeline is
#    better, and the limitations/trade-offs of increasingly autonomous systems.
#
# Include evidence (sample tasks, the agent's step-by-step decisions from a log, before/
# after comparison).
