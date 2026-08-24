from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
from agent.researcher import AgenticRAGResearcher
from utils.display import print_banner, print_result
from config.settings import Settings

def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic RAG Researcher - ReAct / Self-RAG / CRAG auto-select"
    )

    parser.add_argument(
        "--query", type=str, help="Single query to research"
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Run interactive REPL"
    )

    parser.add_argument(
        "--forced-pattern",
        choices=["auto", "react", "selfrag", "crag"],
        default="auto",
        help="Force a specific pattern (default: auto)",
    )

    parser.add_argument(
        "--max_hops", type=int, default=5, help="Max reasoning hops (default: 5)"
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Show full scratchpad trace"
    )
    return parser.parse_args()


def run_single_query(researcher: AgenticRAGResearcher, query:str, verbose: bool):
    print(f"\n🔎  Query: {query}\n")
    result = researcher.research(query)
    print_result(result, verbose=verbose)


def run_interactive(researcher: AgenticRAGResearcher, verbose: bool):
    print("\n💬 Interactive mode - type 'quit' to exit\n")

    while True:
        try:
            query = input("❓    Ask: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋  Bye!")
            break
        if query.lower() in ("quit", "exit", "q"):
            print("👋  Bye!")
            break
        if not query:
            continue
        run_single_query(researcher, query, verbose)

def main():
    print_banner()
    args = parse_args()

    settings = Settings(
        forced_pattern=None if args.forced_pattern == "auto" else args.forced_pattern,
        max_hops=args.max_hops,
        verbose=args.verbose,
    )

    researcher = AgenticRAGResearcher(settings=settings)

    if args.query:
        run_single_query(researcher, args.query, args.verbose)
    elif args.interactive:
        run_interactive(researcher, args.verbose)
    else:
        demo_queries = [
            "What technology powers large language models and who are the key researchers?",

            "Compare the founding stories of Apple and Google - who started first?",

            "Verify: Is Python older than Java? When was each created?",
        ]

        print("\n🚀 Running demo queries (one per pattern)...\n")
        for q in demo_queries:
            run_single_query(researcher, q, args.verbose)
            print("\n" + "-" * 70 + "\n")

if __name__ == "__main__":
    main()
    
