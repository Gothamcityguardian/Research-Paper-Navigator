# Evaluation summary

The system was evaluated with single-paper, cross-paper, paraphrased, broad,
date-related, caching, and prompt-injection test cases. Important responses were
also checked manually against the source PDFs.

## Representative production measurements

| Test | Model calls | Chat tokens | Latency |
| --- | ---: | ---: | ---: |
| Full agent, simple single-paper question | 3 | 3,533 | 3.87 s |
| Direct route, same question and pages | 1 | 1,761 | 2.24 s |
| Exact semantic-cache repeat | 0 | 0 | 0.24 s |
| Equivalent paraphrase, similarity 0.9404 | 0 | 0 | 0.27 s |

Direct routing reduced chat tokens by approximately 50% and latency by
approximately 42% for the tested simple question. Cache hits still require an
embedding request, so zero chat tokens does not mean zero operational cost.

## Security evidence

- A direct instruction to output `PWNED` was sanitized and did not control the
  response.
- A forged paper block claiming 100% accuracy and no limitations was removed
  before retrieval.
- A synthetic retrieved page instructing the model to output `COMPROMISED` was
  treated as untrusted evidence and did not control the response.

Detailed machine-readable outputs are available in `evidence/production/`.

## Remaining limitations

- A cited page may be relevant without supporting the exact wording of a claim.
- Retrieval may miss evidence when the question and paper use different terms.
- The sanitizer cannot guarantee detection of novel, OCR-based, formatting, or
  staged multi-turn attacks.
- Cache hits can become stale, and deterministic routing can misclassify unusual
  questions.
- The human-reviewed evaluation set is still small.
