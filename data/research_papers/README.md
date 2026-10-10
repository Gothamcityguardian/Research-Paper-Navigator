# Research-paper corpus

The course-provided PDF corpus is intentionally not committed to this public
repository. It contains 153 PDFs and produced 6,105 readable page records in the
capstone experiments.

To run the system, either:

1. place the course PDFs in this directory; or
2. set `RAG_PAPERS_DIR` in `.env` to an existing folder containing the PDFs.

The application extracts page text locally and writes generated caches under
`.cache/`. Neither the PDFs nor generated caches are tracked by Git.
