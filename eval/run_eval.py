"""
Eval harness for the Agentic RAG Researcher.

Usage:
    python eval/run_eval.py                  # run all test cases
    python eval/run_eval.py --category factual_comparison
    python eval/run_eval.py --id factual-01

Each test case in test_queries.jsonl declares one or more checks:
    must_contain            - list of strings that MUST all appear in the answer
    must_contain_any        - list of alternative-string-groups; each group needs >=1 match
    must_not_contain         - list of strings that must NEVER appear (regression guard)
    expected_pattern        - pattern_used must equal this value
    forced_pattern           - if set, Settings(forced_pattern=...) is used for this run
    pass_if_pattern_used_any - list of pattern names; if result.pattern_used is in this
                                list, the test passes immediately regardless of any
                                must_contain/must_not_contain checks below. Use this when
                                a certain routing outcome (e.g. "correctly used a local
                                document") is itself sufficient evidence of correct
                                behavior, separate from — or in addition to — a specific
                                phrase-based fallback check for when that routing doesn't
                                happen (e.g. no matching local document exists).
    check                    - one of: hops_equals_api_calls, confidence_between_0_and_1,
                                no_exception_raised, raises_value_error
    expected_exception       - exception class name, used with check=raises_value_error
    expected_tool_used_any   - list of tool names; at least one step must have used one of them

Results are written to eval/results/<timestamp>.json and summarized on the console.
Exit code is 1 if any test fails, so this can be wired into CI.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.researcher import AgenticRAGResearcher
from config.settings import Settings

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass



RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
BOLD = "\033[1m"


def load_test_cases(path: Path) -> list:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"{RED}✖ Skipping malformed line {line_num}: {e}{RESET}")
    return cases


def get_researcher(forced_pattern: str, cache: dict) -> AgenticRAGResearcher:
    key = forced_pattern or "__default__"
    if key not in cache:
        settings = Settings(forced_pattern=forced_pattern, max_hops=5, verbose=False)
        cache[key] = AgenticRAGResearcher(settings=settings)
    return cache[key]


def check_must_contain(answer: str, terms: list) -> tuple:
    lower = answer.lower()
    missing = [t for t in terms if t.lower() not in lower]
    return (len(missing) == 0, f"missing required terms: {missing}" if missing else "ok")


def check_must_contain_any(answer: str, groups: list) -> tuple:
    lower = answer.lower()
    failed_groups = []
    for group in groups:
        if not any(term.lower() in lower for term in group):
            failed_groups.append(group)
    return (len(failed_groups) == 0, f"no match found for group(s): {failed_groups}" if failed_groups else "ok")


def check_must_not_contain(answer: str, terms: list) -> tuple:
    lower = answer.lower()
    found = [t for t in terms if t.lower() in lower]
    return (len(found) == 0, f"forbidden terms present: {found}" if found else "ok")


def run_single_test(case: dict, cache: dict) -> dict:
    result_record = {
        "id": case["id"],
        "query": case["query"],
        "category": case.get("category", "uncategorized"),
        "notes": case.get("notes", ""),
        "passed": False,
        "details": [],
        "error": None,
    }

    forced_pattern = case.get("forced_pattern")
    expected_exception = case.get("expected_exception")

    try:
        researcher = get_researcher(forced_pattern, cache)
        start = time.time()
        result = researcher.research(case["query"])
        elapsed = round(time.time() - start, 2)
        result_record["elapsed_seconds"] = elapsed

        if expected_exception:
            result_record["details"].append(f"expected {expected_exception} but no exception was raised")
            result_record["passed"] = False
            return result_record

        answer = getattr(result, "final_answer", "") or ""
        actual_pattern = (getattr(result, "pattern_used", "") or "").lower()

        result_record["confidence"] = getattr(result, "confidence", None)
        result_record["pattern_used"] = getattr(result, "pattern_used", None)
        result_record["hops"] = getattr(result, "hops", None)
        result_record["api_calls"] = getattr(result, "api_calls", None)
        result_record["answer_preview"] = answer[:200]
        result_record["answer_full"] = answer

        # Alternate pass condition: certain routing outcomes are themselves
        # sufficient evidence of correct behavior (e.g. "correctly used a
        # real local document"), independent of any phrase-based check below.
        pass_if_patterns = case.get("pass_if_pattern_used_any")
        if pass_if_patterns and actual_pattern in [p.lower() for p in pass_if_patterns]:
            result_record["passed"] = True
            result_record["details"].append(
                f"pass_if_pattern_used_any: pattern_used='{actual_pattern}' matched "
                f"{pass_if_patterns} — treated as a valid pass independent of content checks."
            )
            return result_record

        checks_passed = []

        if "must_contain" in case:
            ok, msg = check_must_contain(answer, case["must_contain"])
            checks_passed.append(ok)
            result_record["details"].append(f"must_contain: {msg}")

        if "must_contain_any" in case:
            ok, msg = check_must_contain_any(answer, case["must_contain_any"])
            checks_passed.append(ok)
            result_record["details"].append(f"must_contain_any: {msg}")

        if "must_not_contain" in case:
            ok, msg = check_must_not_contain(answer, case["must_not_contain"])
            checks_passed.append(ok)
            result_record["details"].append(f"must_not_contain: {msg}")

        if "expected_pattern" in case:
            expected = case["expected_pattern"].lower()
            ok = actual_pattern == expected
            checks_passed.append(ok)
            result_record["details"].append(
                f"expected_pattern: expected '{expected}', got '{actual_pattern}'" + (" ok" if ok else " FAIL")
            )

        if "expected_tool_used_any" in case:
            steps = getattr(result, "steps", []) or []
            tools_used = {getattr(s, "tool_used", "") for s in steps}
            expected_tools = set(case["expected_tool_used_any"])
            ok = bool(tools_used & expected_tools) or actual_pattern in {t.lower() for t in expected_tools}
            checks_passed.append(ok)
            result_record["details"].append(
                f"expected_tool_used_any: expected one of {expected_tools}, "
                f"tools used: {tools_used}, pattern_used: {actual_pattern}"
            )

        check_name = case.get("check")
        if check_name == "hops_equals_api_calls":
            hops = getattr(result, "hops", None)
            api_calls = getattr(result, "api_calls", None)
            ok = hops == api_calls
            checks_passed.append(ok)
            result_record["details"].append(f"hops_equals_api_calls: hops={hops}, api_calls={api_calls}")

        elif check_name == "confidence_between_0_and_1":
            conf = getattr(result, "confidence", None)
            ok = isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0
            checks_passed.append(ok)
            result_record["details"].append(f"confidence_between_0_and_1: confidence={conf}")

        elif check_name == "no_exception_raised":
            checks_passed.append(True)
            result_record["details"].append("no_exception_raised: ok (completed without crash)")

        result_record["passed"] = all(checks_passed) if checks_passed else False

    except Exception as e:
        exc_name = type(e).__name__
        result_record["error"] = f"{exc_name}: {str(e)}"

        if expected_exception and exc_name == expected_exception:
            result_record["passed"] = True
            result_record["details"].append(f"raised expected {expected_exception} correctly")
        elif case.get("check") == "no_exception_raised":
            result_record["passed"] = False
            result_record["details"].append(f"expected no exception, but got {exc_name}: {e}")
        else:
            result_record["passed"] = False
            result_record["details"].append(f"unexpected exception: {exc_name}: {e}")

    return result_record

def is_daily_quota_exhausted(error_str: str) -> bool:
    markers = ("Daily token quota exhausted", "rate_limit_exceeded", "tokens per day")
    return any(m in error_str for m in markers)
    return "Daily token quota exhausted" in error_str


def main():
    filter_str = None
    for arg in sys.argv[1:]:
        if arg.startswith("--filter="):
            filter_str = arg.split("=", 1)[1]

    cases = load_test_cases("eval/test_queries.jsonl")  # however your current loading works
    if filter_str:
        cases = [c for c in cases if filter_str in c["id"]]
        print(f"Filtered to {len(cases)} case(s) matching '{filter_str}'")
        
    parser = argparse.ArgumentParser(description="Run the eval suite against the live researcher")
    parser.add_argument("--category", type=str, default=None, help="Only run tests in this category")
    parser.add_argument("--id", type=str, default=None, help="Only run the test with this exact id")
    parser.add_argument("--filter", type=str, default=None, help="Only run tests whose id contains this substring")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N matching test cases")
    parser.add_argument("--file", type=str, default=None, help="Path to test_queries.jsonl (default: alongside this script)")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait between test cases (reduces rate-limit hits)")

    args = parser.parse_args()

    test_file = Path(args.file) if args.file else Path(__file__).parent / "test_queries.jsonl"
    if not test_file.exists():
        print(f"{RED}✖ Test file not found: {test_file}{RESET}")
        sys.exit(1)

    cases = load_test_cases(test_file)

    if args.category:
        cases = [c for c in cases if c.get("category") == args.category]
    if args.id:
        cases = [c for c in cases if c.get("id") == args.id]
    if args.filter:
        cases = [c for c in cases if args.filter in c["id"]]
    if args.limit:
        cases = cases[:args.limit]

    if not cases:
        print(f"{YELLOW}No test cases matched the given filters.{RESET}")
        sys.exit(0)

    print(f"\n{BOLD}Running {len(cases)} eval test case(s)...{RESET}\n")

    cache = {}
    all_results = []
    passed_count = 0

    for i, case in enumerate(cases, 1):
        print(f"{DIM}[{i}/{len(cases)}]{RESET} {case['id']}: {case['query'][:60]}...", end=" ", flush=True)
        record = run_single_test(case, cache)
        all_results.append(record)

        if record["passed"]:
            passed_count += 1
            print(f"{GREEN}PASS{RESET}")
        else:
            print(f"{RED}FAIL{RESET}")
            for detail in record["details"]:
                print(f"    {DIM}- {detail}{RESET}")
            if record["error"]:
                print(f"    {RED}error: {record['error']}{RESET}")

            if record["error"] and is_quota_exhausted(record["error"]):
                print(f"\n{YELLOW}⚠️ Daily token quota exhausted — stopping early to avoid wasting "
                      f"remaining test slots on guaranteed failures.{RESET}")
                print(f"{YELLOW}Ran {i}/{len(cases)} before quota ran out. "
                      f"Re-run the remaining cases once your quota resets (see error message above for wait time).{RESET}")
                break

        if i < len(cases) and args.sleep > 0:
            time.sleep(args.sleep) 

    total = len(all_results)
    failed_count = total - passed_count

    print(f"\n{BOLD}{'=' * 50}{RESET}")
    summary_color = GREEN if failed_count == 0 else RED
    print(f"{summary_color}{BOLD}{passed_count}/{total} passed{RESET}")

    if failed_count > 0:
        print(f"\n{RED}Failed tests:{RESET}")
        for r in all_results:
            if not r["passed"]:
                print(f"  - {r['id']} ({r['category']})")

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"eval_{timestamp}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total": total,
                "passed": passed_count,
                "failed": failed_count,
                "results": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n{DIM}Full results written to: {output_path}{RESET}\n")

    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
