r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import gzip
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# %%
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 5
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "baseline_retrieval.log"
load_dotenv(PROJECT_ROOT / ".env")

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "research_papers"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)


# %%
def check_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
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


def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## Research-paper corpus
#
# The baseline loads every PDF in the ResearchPapers folder. Each readable PDF
# page becomes one retrievable document so results retain useful paper and page
# metadata. Page-level indexing is intentionally simple and gives us a baseline
# that later chunking, vector, and hybrid retrieval can improve upon.


# %%
PAPERS_DIR = Path(
    os.getenv("RAG_PAPERS_DIR", str(PROJECT_ROOT / "data" / "research_papers"))
)
CACHE_PATH = PROJECT_ROOT / ".cache" / "research_papers_page_cache.json.gz"
CACHE_VERSION = 1


def _paper_signature(pdf_paths: list[Path]) -> list[dict]:
    """Return inexpensive metadata used to detect additions or PDF changes."""
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "modified_ns": path.stat().st_mtime_ns,
        }
        for path in pdf_paths
    ]


def load_research_papers(
    papers_dir: Path,
    cache_path: Path = CACHE_PATH,
) -> list[dict]:
    """Return one retrieval document per page, using a validated local cache."""
    if not papers_dir.is_dir():
        raise FileNotFoundError(f"Research-paper folder not found: {papers_dir}")

    pdf_paths = sorted(papers_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {papers_dir}")
    signature = _paper_signature(pdf_paths)

    if cache_path.is_file():
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
            documents = cached.get("documents", [])
            if (
                cached.get("version") == CACHE_VERSION
                and cached.get("paper_signature") == signature
                and documents
            ):
                print(
                    f"Loaded {len(documents)} cached pages from "
                    f"{len(pdf_paths)} research-paper PDFs."
                )
                return documents
            print("PDF collection changed; rebuilding the local page cache.")
        except (OSError, ValueError, TypeError) as exc:
            print(f"Warning: could not use page cache ({exc}); rebuilding it.")

    documents = []
    for pdf_path in pdf_paths:
        try:
            with pymupdf.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue
                    documents.append(
                        {
                            "id": f"{pdf_path.stem}_page_{page_number}",
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_number,
                        }
                    )
        except Exception as exc:
            print(f"Warning: could not read {pdf_path.name}: {exc}")

    if not documents:
        raise RuntimeError(f"No readable PDF pages found in: {papers_dir}")

    cache_payload = {
        "version": CACHE_VERSION,
        "paper_signature": signature,
        "documents": documents,
    }
    temporary_cache = cache_path.with_suffix(cache_path.suffix + ".tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(temporary_cache, "wt", encoding="utf-8") as cache_file:
            json.dump(cache_payload, cache_file, ensure_ascii=False)
        temporary_cache.replace(cache_path)
        print(f"Saved extracted page text to local cache: {cache_path.name}")
    except OSError as exc:
        print(f"Warning: could not save page cache: {exc}")

    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} research-paper PDFs.")
    return documents


SAMPLE_DOCS = load_research_papers(PAPERS_DIR)
DOC_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}


# %% [markdown]
# ## Step 2 — The baseline retriever (provided)
#
# A simple keyword-overlap retriever scores each document by how many query words
# it shares, and returns the top-k. This is the smallest possible baseline (a stand-in
# for the BM25 / vector / hybrid retriever you built in the Module 2 labs). The
# `answer` function then asks the LLM using only the retrieved documents.

# %%
def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """Baseline keyword retrieval: Score each doc by shared-word count, return top-k."""
    q = _tokens(query)
    scored = [(d["id"], float(len(q & _tokens(d["text"])))) for d in SAMPLE_DOCS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for doc_id, score in scored[:k] if score > 0]


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(
        (
            f"[{i}] Paper: {DOC_BY_ID[i]['source']}, "
            f"PDF page: {DOC_BY_ID[i]['page']}\n{DOC_BY_ID[i]['text']}"
        )
        for i in doc_ids
        if i in DOC_BY_ID
    )
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    """Return 3-5 representative queries for YOUR chosen scenario.

    TODO — your turn. See the guidance above. Each item is a query string. Pick
    queries that a real user of your system would ask and that require different
    retrieval behaviors (single-doc, multi-doc, paraphrased).

    Delete the raise NotImplementedError line once your code works.
    """
    return [
        "What role does the programmer play in the sketching approach to program synthesis?",
        "What is program synthesis, and how does the sketching approach change how a program is created?",
        "How do The Sketching Approach to Program Synthesis and Learning to Infer Program Sketches differ in how program sketches are produced?",
        "Which method lets a developer provide an incomplete program that an automated system completes under constraints?",
        "What evidence do the papers provide about the benefits and limitations of automatically inferred program sketches?",
    ]



# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
def run() -> None:
    llm = make_llm()
    queries = my_representative_queries()
    print(f"Checkpoint 2.1 — baseline retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(queries, 1):
        hits = retrieve(query, TOP_K)
        print("=" * 72)
        print(f"QUERY {i}: {query}")
        print(f"  retrieved: {hits}")
        if not hits:
            print("  (nothing matched — note this in your writeup)")
            continue
        ans = answer(llm, query, [doc_id for doc_id, _ in hits])
        print(f"  answer: {ans}\n")
        log(f"QUERY {i}: {query}", f"retrieved={hits}\nanswer={ans}")
    print("=" * 72)
    print("Done. Use the retrieved document results above as evidence in your writeup, and "
          "describe your REAL baseline (over your full corpus) in the submission.")


if __name__ == "__main__":
    run()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
