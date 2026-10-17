"""Live evaluation against the real index and Gemini models. Makes real (free-tier) API calls.

Usage: python -m eval.run_eval [--delay SECONDS]

For each question it checks:
  - supported questions: answered, cites the expected source, and the answer contains an expected fact
  - unsupported questions: the assistant abstains
"""

import argparse
import json
import time
from pathlib import Path

from backend.config import get_settings
from backend.retrieval.qa import build_qa_service

QUESTIONS = Path(__file__).with_name("questions.json")


def evaluate(case: dict, result) -> tuple[bool, str]:
    if not case["supported"]:
        return (not result.answered, f"abstained ({result.abstain_reason})" if not result.answered else "ANSWERED")
    if not result.answered:
        return False, f"abstained ({result.abstain_reason})"
    cited = {c.source for c in result.citations}
    fact = any(term.lower() in result.answer.lower() for term in case["expect_any"])
    source_ok = case["source"] in cited
    return source_ok and fact, f"source={'ok' if source_ok else 'WRONG ' + str(sorted(cited))} fact={'ok' if fact else 'MISSING'}"


def ask_with_retry(service, question: str, wait: float):
    """Retry once on temporary Gemini errors (429/503); return the exception if it persists."""
    for attempt in range(2):
        try:
            return service.ask(question)
        except Exception as e:  # report and continue rather than abort the whole run
            if attempt == 0 and getattr(e, "code", None) in (429, 503):
                time.sleep(wait)
                continue
            return e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=5.0, help="seconds between questions (rate limits)")
    parser.add_argument("--retry-wait", type=float, default=30.0, help="seconds before retrying a 429/503")
    args = parser.parse_args()

    service = build_qa_service(get_settings())
    cases = json.loads(QUESTIONS.read_text())
    passed = {True: 0, False: 0}
    totals = {True: 0, False: 0}
    errors = 0

    for i, case in enumerate(cases):
        if i:
            time.sleep(args.delay)
        result = ask_with_retry(service, case["question"], args.retry_wait)
        kind = "supported  " if case["supported"] else "unsupported"
        if isinstance(result, Exception):
            errors += 1
            code = getattr(result, "code", None)
            print(f"ERROR [{kind}] {type(result).__name__} (code={code}) | {case['question']}")
            continue
        ok, detail = evaluate(case, result)
        totals[case["supported"]] += 1
        passed[case["supported"]] += ok
        top = result.retrieved[0].relevance if result.retrieved else None
        print(f"{'PASS' if ok else 'FAIL'} [{kind}] top_rel={top} {detail} | {case['question']}")
        print(f"      -> {result.answer[:160]!r}")

    print(f"\nsupported answered correctly: {passed[True]}/{totals[True]}")
    print(f"unsupported abstained:        {passed[False]}/{totals[False]}")
    print(f"errors (not scored):          {errors}")


if __name__ == "__main__":
    main()
