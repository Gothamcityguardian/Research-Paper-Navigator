r"""Capstone Checkpoint 3.1 - Research Paper Navigator evaluation.
"""

# %%
from __future__ import annotations

import importlib.util
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    # The script can still run on networks that do not require truststore.
    pass

from langchain_core.messages import HumanMessage, SystemMessage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# %% [markdown]
# ## Step 1 - Load the real Capstone 2.1 baseline
#
# Reusing the existing module keeps the PDF extraction, page cache, keyword
# retriever, source metadata, and grounded-answer prompt identical to the
# baseline already tested in Checkpoint 2.1.

# %%
HERE = Path(__file__).resolve().parent
BASELINE_PATH = (
    HERE.parent
    / "Capstone_Checkpoint_2_1_starter"
    / "capstone_checkpoint_2_1_research_papers_solution.py"
)
LOG_PATH = HERE / "checkpoint_3_1_research_papers_evaluation.log"
SCENARIO = "research_papers"
TOP_K = 5


def _load_baseline_module():
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(f"Capstone 2.1 baseline not found: {BASELINE_PATH}")

    spec = importlib.util.spec_from_file_location("checkpoint_2_1_baseline", BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load the Capstone 2.1 baseline: {BASELINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = _load_baseline_module()


# %% [markdown]
# ## Step 2 - Evaluation metric
#
# The judge compares each grounded answer with one sentence describing the
# required content. A result passes only when the answer covers every major
# requirement in that grading note.

# %%
JUDGE_SYSTEM = (
    "You are a strict evaluator. Compare the ANSWER with the GRADING NOTES. "
    "Return exactly 'pass' only if the answer covers every major requirement "
    "in the grading notes. Otherwise return exactly 'fail'. Do not reward an "
    "answer merely because it sounds plausible."
)


def log(label: str, text: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {label}\n{text}\n{'-' * 72}\n")


def judge(llm, answer_text: str, grading_notes: str) -> str:
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=(
                f"ANSWER:\n{answer_text}\n\n"
                f"GRADING NOTES:\n{grading_notes}\n\n"
                "Verdict (pass/fail):"
            )
        ),
    ]
    verdict = llm.invoke(messages).content.strip().lower()
    return "pass" if verdict == "pass" else "fail"


# %% [markdown]
# ## Step 3 - Evaluation set
#
# These cases reuse the representative questions from Checkpoint 2.1. They
# include precise, conceptual, cross-paper, paraphrased, and broad questions.
# The final broad question is expected to expose the keyword baseline's weak
# handling of concepts expressed with different wording.

# %%
def my_eval_set() -> list[dict[str, str]]:
    return [
        {
            "question": (
                "What role does the programmer play in the sketching approach "
                "to program synthesis?"
            ),
            "grading_notes": (
                "Explains that the programmer supplies a partial program or sketch "
                "that constrains the implementation, while the synthesizer fills in "
                "the missing details or holes."
            ),
        },
        {
            "question": (
                "What is program synthesis, and how does the sketching approach "
                "change how a program is created?"
            ),
            "grading_notes": (
                "Defines program synthesis as automatically producing a program from "
                "a specification or constraints, and explains that sketching lets a "
                "programmer provide a partial template that the synthesizer completes."
            ),
        },
        {
            "question": (
                "How do The Sketching Approach to Program Synthesis and Learning "
                "to Infer Program Sketches differ in how program sketches are produced?"
            ),
            "grading_notes": (
                "States that the sketching approach uses a human-written sketch, "
                "while Learning to Infer Program Sketches uses a learned neural sketch "
                "generator to produce sketches from specifications before symbolic search."
            ),
        },
        {
            "question": (
                "Which method lets a developer provide an incomplete program that "
                "an automated system completes under constraints?"
            ),
            "grading_notes": (
                "Identifies sketch-based program synthesis and explains that a partial "
                "program or sketch contains missing parts that are completed subject "
                "to a specification or constraints."
            ),
        },
        {
            "question": (
                "What evidence do the papers provide about the benefits and limitations "
                "of automatically inferred program sketches?"
            ),
            "grading_notes": (
                "Uses evidence from Learning to Infer Program Sketches and includes at "
                "least one reported benefit, such as improved out-of-sample generalization "
                "or flexible neural-symbolic search, and at least one limitation or "
                "trade-off, such as dependence on training data, search budget, or task domain."
            ),
        },
    ]


# %% [markdown]
# ## Step 4 - Run the baseline evaluation

# %%
def format_hits(hits: list[tuple[str, float]]) -> str:
    lines = []
    for doc_id, score in hits:
        document = BASELINE.DOC_BY_ID[doc_id]
        lines.append(
            f"{document['source']} | PDF position {document['page']} | score={score:.1f}"
        )
    return "\n".join(lines)


def run_evaluation() -> tuple[int, int]:
    llm = BASELINE.make_llm()
    evaluation_set = my_eval_set()
    passes = 0

    print(f"Checkpoint 3.1 - baseline evaluation | scenario: {SCENARIO}\n")
    for number, item in enumerate(evaluation_set, start=1):
        hits = BASELINE.retrieve(item["question"], TOP_K)
        answer_text = (
            BASELINE.answer(llm, item["question"], [doc_id for doc_id, _ in hits])
            if hits
            else "(no documents retrieved)"
        )
        verdict = judge(llm, answer_text, item["grading_notes"])
        passes += verdict == "pass"
        hit_text = format_hits(hits) if hits else "(no documents retrieved)"

        print("=" * 72)
        print(f"Q{number}: {item['question']}")
        print(f"Retrieved:\n{hit_text}")
        print(f"Verdict: {verdict.upper()}")
        print(f"Answer: {answer_text}\n")

        log(
            f"Q{number}: {item['question']}",
            (
                f"grading_notes={item['grading_notes']}\n"
                f"retrieved:\n{hit_text}\n"
                f"verdict={verdict}\n"
                f"answer={answer_text}"
            ),
        )

    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(evaluation_set)}")
    return passes, len(evaluation_set)


# %% [markdown]
# ## Step 5 - Validate the evaluation framework
#
# The same grading notes are applied to a correct answer and a deliberately
# false answer. The validation succeeds if the judge passes the correct answer
# and fails the manipulated answer.

# %%
def validate_framework() -> tuple[str, str]:
    llm = BASELINE.make_llm()
    notes = (
        "Identifies sketch-based program synthesis and explains that the programmer "
        "provides a partial program with holes that the synthesizer completes under "
        "a specification or constraints."
    )
    correct = (
        "Sketch-based program synthesis lets the programmer provide a partial program "
        "with holes. The synthesizer fills those holes while satisfying the supplied "
        "specification or constraints."
    )
    manipulated = (
        "Sketch-based synthesis requires the programmer to write the complete final "
        "program, and the automated system only checks spelling and formatting."
    )

    correct_verdict = judge(llm, correct, notes)
    manipulated_verdict = judge(llm, manipulated, notes)

    print("\n--- Framework validation ---")
    print(f"Correct answer -> {correct_verdict.upper()} (expected PASS)")
    print(f"Manipulated answer -> {manipulated_verdict.upper()} (expected FAIL)")
    log(
        "FRAMEWORK VALIDATION",
        (
            f"grading_notes={notes}\n"
            f"correct_answer={correct}\n"
            f"correct_verdict={correct_verdict}\n"
            f"manipulated_answer={manipulated}\n"
            f"manipulated_verdict={manipulated_verdict}"
        ),
    )
    return correct_verdict, manipulated_verdict


# %%
if __name__ == "__main__":
    run_evaluation()
    validate_framework()

