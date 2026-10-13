"""Command-line entry point.

    python main.py report     run the analysis, print a summary, write results/metrics.json
    python main.py figures    report + regenerate every figure in figures/
    python main.py deck       figures + rebuild the PowerPoint in presentation/
    python main.py app        start the interactive learning app (same as: streamlit run app.py)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="LSA Lab: document clustering with SVD")
    parser.add_argument("command", choices=["report", "figures", "deck", "app"], nargs="?", default="report")
    args = parser.parse_args(argv)

    if args.command == "app":
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")])

    from lsa_lab import report

    out = report.build()
    path = report.save(out["results"])
    print(f"Wrote {path.relative_to(ROOT)}")
    if args.command == "report":
        return 0

    from lsa_lab import figures

    news_k = out["experiments"]["news"].corpus.default_k
    figs = figures.make_all(ROOT / "figures", out["experiments"], news_k, out["sweep"], out["comparisons"]["news"])
    print(f"Wrote {len(figs['paths'])} figures to figures/")
    if args.command == "figures":
        return 0

    from lsa_lab import deck

    deck_path = deck.build(out, figs)
    print(f"Wrote {deck_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
