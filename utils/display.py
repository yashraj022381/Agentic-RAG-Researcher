from dataclasses import dataclass
from typing import TYPE_CHECKING
from rich.console import Console
import re

if TYPE_CHECKING:
    from loop.scratchpad import ResearchResult

RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"

def bar(value: float, width: int = 30) -> str:
    try:
        if isinstance(value, (list, tuple)):
            value = value[0] if value else 0.8
        value = float(value)
        value = max(0.0, min(1.0, value))
        filled = int(value * width)
        empty = width - filled
        if value >= 0.8:
            colour = GREEN
        elif value >= 0.6:
            colour = YELLOW
        else:
            colour = RED
        return f"{colour}{'█' * filled}{DIM}{'░' * empty}{RESET}"
    except:
        return f"{GREEN}{'█' * 24}{DIM}{'░' * 6}{RESET}"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\{["\']?tool["\']?\}', '', text)
    text = re.sub(r'\[\s*tool\s*\]', '', text)
    return text.strip()

def _extract_document_name(sources: list) -> str:
    """Pull a clean document filename out of the sources list, so it can be
    shown prominently next to 'Tool Used' instead of only appearing buried
    in the Sources section below."""
    for s in sources or []:
        s = str(s)
        m = re.search(r'Document:\s*([^(]+?)(?:\s*\(|$)', s)
        if m:
            return m.group(1).strip()
    return ""

def print_result(result, verbose: bool = False):
    console = Console()
    print()
    
    # Answer Box
    print(f"{GREEN}╔{'═' * 68}╗{RESET}")
    print(f"{GREEN}║{RESET} {BOLD}ANSWER{RESET} {' ' * 60}{GREEN}║{RESET}")
    print(f"{GREEN}╚{'═' * 68}╝{RESET}")
    print()

    answer = clean_text(getattr(result, 'final_answer', str(result)))
    for paragraph in answer.split('\n'):
        paragraph = paragraph.strip()
        if not paragraph.strip():
            continue
        words = paragraph.split()
        lines_out = []
        line = []
        for word in words:
            if sum(len(w) + 1 for w in line) + len(word) > 68:
                lines_out.append(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            lines_out.append(" ".join(line))
        for l in lines_out:
            print(l)
        print()

    print()

    # Pattern Used / Tool Used — document_reader isn't one of the three
    # agentic patterns (ReAct/Self-RAG/CRAG), it's a direct-answer tool that
    # bypasses the agentic loop entirely, so label it distinctly rather than
    # calling it a "Pattern".
    pattern_used = getattr(result, 'pattern_used', 'UNKNOWN').upper()
    sources = getattr(result, 'sources', [])

    if pattern_used == "DOCUMENT_READER":
        doc_name = _extract_document_name(sources)
        if doc_name:
            console.print(f"Tool Used: DOCUMENT_READER  (Document: {doc_name})", style="bold cyan")
        else:
            console.print("Tool Used: DOCUMENT_READER", style="bold cyan")
    else:
        console.print(f"Pattern Used: {pattern_used}", style="bold cyan")

    print()

    # Sources
    console.print("Sources:", style="bold")
    if not sources:
        console.print(" → web_search")
    else:
        for src in sources:
            clean_src = str(src).replace('{"tool"}', '').replace('{tool}', '').replace('[{tool}]', '').strip()
            clean_src = re.sub(r'["\']?tool["\']?', '', clean_src).strip()
            if clean_src and len(clean_src) > 2 and clean_src.lower() != "tool":
                console.print(f" → {clean_src}")
            else:
                console.print(" → web_search")

    print()

    # Confidence Bar
    conf = getattr(result, 'confidence', 0.8)
    if isinstance(conf, (list, tuple)):
        conf = float(conf[0]) if conf else 0.8
    conf = float(conf)
    bar_str = bar(conf)
    pct = f"{conf:.0%}"
    print(f"Confidence: {bar_str} {pct} | Hops: {getattr(result, 'hops', 1)} | API calls: {getattr(result, 'api_calls', 2)}")
    print()

    if verbose:
        # Add verbose scratchpad if needed
        pass

def print_error(message: str):
    print(f"{RED}✖ ERROR:{RESET} {message}")

def print_info(message: str):
    print(f"{CYAN}ℹ{RESET} {message}")

def print_banner():
    """Print the startup banner."""
    width = 55
    print()
    print(f"{CYAN}╔{'═' * width}╗{RESET}")
    print(f"{CYAN}║{RESET} {BOLD}AGENTIC RAG RESEARCHER{RESET} {' ' * (width-25)}{CYAN}║{RESET}")
    print(f"{CYAN}╚{'═' * width}╝{RESET}")
    print()
    print(f"{CYAN} Auto-selects {RESET}🔥 ReAct · 🧠 Self-RAG · 📦 CRAG")
    print(f"{DIM}Multi-hop · Scratchpad tracing · Pattern auto-detection{RESET}")
    print()
