# Research Paper Navigator

Research Paper Navigator is an agentic retrieval-augmented generation system for
finding, explaining, and comparing evidence across technical research papers.
It was developed as the final capstone for the MIT xPRO Retrieval-Augmented LLM
program.

The project addresses a problem observed in the no-retrieval baseline: a general
LLM may know a paper's main idea while still inventing quotations, supplying an
unsupported page number, or citing the whole paper instead of the passage that
supports a claim. This system retrieves PDF pages locally and asks the model to
answer only from the retrieved evidence.

## Main capabilities

- Extract and cache readable text from a local PDF collection.
- Preserve the PDF filename and page position for traceability.
- Combine BM25 keyword retrieval with semantic retrieval.
- Route simple single-paper questions through a lower-cost direct path.
- Use a multistep agent for comparisons and broader evidence needs.
- Decompose questions, filter by publication year, and expand adjacent pages.
- Return grounded answers with retrieved page identifiers.
- Reuse equivalent, self-contained answers through a high-threshold semantic
  cache.
- Apply input sanitization and document/user trust boundaries.

See [Architecture](docs/ARCHITECTURE.md) and
[Evaluation](docs/EVALUATION.md) for the design and measured results.

## Repository structure

```text
src/research_paper_navigator/   Final runnable implementation
data/research_papers/           Local PDF location; PDFs are not committed
evidence/                       Selected evaluation logs and JSON results
checkpoints/                    Completed checkpoint solutions and worksheets
docs/presentation/              Design-review presentation and preparation notes
```

## Requirements

- Python 3.11 or 3.12
- An OpenRouter API key with access to the configured chat and embedding models
- The course research-paper PDF corpus, or another local folder of PDFs

Model calls and embeddings use OpenRouter and may incur charges. PDF extraction,
BM25 indexing, metadata handling, and caches run locally.

## Installation

From PowerShell:

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
Copy-Item .env.example .env
```

Edit `.env` and set:

```text
OPENROUTER_API_KEY=your_key_here
```

Place the PDF corpus under `data/research_papers/`, or set an external location:

```text
RAG_PAPERS_DIR=C:\path\to\ResearchPapers
```

The real `.env`, PDF corpus, extracted-page cache, embeddings, Chroma data, and
runtime outputs are excluded from Git.

## Run

Ask one question:

```powershell
python -m research_paper_navigator --question "How do The Sketching Approach to Program Synthesis and Learning to Infer Program Sketches differ in how program sketches are produced?"
```

Start interactive mode:

```powershell
python -m research_paper_navigator --chat
```

Run the representative production demonstration:

```powershell
python -m research_paper_navigator --demo
```

Run the targeted security probes:

```powershell
python -m research_paper_navigator --security-tests
```

Generated caches are written to `.cache/`, and generated results are written to
`outputs/`. Both directories are ignored by Git.

## Evaluation highlights

- The direct route used 1 call, 1,761 chat tokens, and 2.24 seconds for a tested
  single-paper question.
- The full agent used 3 calls, 3,533 chat tokens, and 3.87 seconds for the same
  question while retrieving the same pages.
- Exact and equivalent cached questions used no chat-model calls and returned in
  approximately 0.24-0.27 seconds.
- Direct, forged-source, and retrieved-document injection probes did not control
  the tested responses.

These measurements are observations from specific runs, not universal service
benchmarks. Network conditions and model-provider behavior will affect results.

## Known limitations

- A page ID does not by itself prove that every claim is supported; claim-level
  citation verification remains future work.
- Semantic retrieval currently embeds paper first pages while BM25 scores all
  extracted pages. A full page-level vector index may improve recall at higher
  embedding and storage cost.
- The cache needs expiration and stronger access controls for multi-user use.
- Deterministic routing is consistent but can misclassify unusual questions.
- The security controls reduce common prompt-injection risks but are not a formal
  security boundary.

## Public-repository safety

No API keys, `.env` file, virtual environment, generated cache, or research-paper
PDF is included. The repository includes selected outputs needed to review the
evaluation claims. Do not commit credentials or proprietary/copyrighted source
documents.
