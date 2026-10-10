# Architecture

The Research Paper Navigator turns a question into a source-grounded answer with
traceable PDF page identifiers.

```text
User question
    |
    v
Input sanitization and deterministic routing
    |
    +-- clear single-paper question --> direct retrieval and answer
    |
    +-- comparison or broad question --> multistep retrieval agent
                                             |
                                             +-- title search
                                             +-- hybrid BM25/vector search
                                             +-- year filtering
                                             +-- adjacent-page expansion
                                             +-- evidence-coverage check
    |
    v
Grounded generation with paper and page IDs
    |
    v
High-threshold semantic answer cache
```

## Retrieval design

- Every readable PDF page is stored with its filename and PDF position.
- BM25 provides exact-title and technical-term matching.
- Semantic similarity helps with paraphrased and conceptual questions.
- Dynamic weighting gives more weight to BM25 for named titles and more weight
  to semantic retrieval for conceptual queries.
- The agent can decompose a question, retrieve again, and expand neighboring
  pages when the initial evidence is incomplete.

## Trust boundaries

- User input is sanitized before planning and retrieval.
- Forged `BEGIN PAPER`/`END PAPER` blocks and common instruction-overrides are
  removed or neutralized.
- User text and retrieved text are escaped and placed in separate XML sections.
- Retrieved documents are treated as evidence, never as instructions.
- Answers may cite only retrieved page IDs.
- Cache entries are scoped to a corpus fingerprint and answer-policy version.
