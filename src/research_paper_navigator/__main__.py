"""Command-line entry point for the Research Paper Navigator."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query, demo, or security-test the Research Paper Navigator."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--question", help="Ask one research-paper question.")
    mode.add_argument("--chat", action="store_true", help="Start an interactive chat.")
    mode.add_argument("--demo", action="store_true", help="Run the representative demo.")
    mode.add_argument(
        "--security-tests", action="store_true", help="Run prompt-injection probes."
    )
    return parser.parse_args()


def print_result(result) -> None:
    print(f"Route: {result.route}")
    print(f"Pages: {', '.join(result.pages) if result.pages else '(none)'}")
    print(f"Model calls: {result.model_calls}")
    print(f"Tokens: {result.token_usage}")
    print(f"Elapsed: {result.elapsed_seconds}s")
    print(f"\nAnswer:\n{result.answer}")


def main() -> None:
    args = parse_args()
    from .production import (
        ProductionResearchPaperPipeline,
        run_demo,
        run_security_tests,
    )

    if args.demo:
        run_demo()
        return
    if args.security_tests:
        run_security_tests()
        return

    pipeline = ProductionResearchPaperPipeline()
    if args.question:
        print_result(pipeline.query(args.question))
        return

    print("Research Paper Navigator. Type a question or 'exit' to stop.")
    while True:
        question = input("\nYou: ").strip()
        if question.casefold() in {"exit", "quit"}:
            break
        if question:
            print_result(pipeline.query(question))


if __name__ == "__main__":
    main()
