"""Presentation Generator for MFDS Mini Project.

Topic: Latent Semantic Analysis (LSA) with Singular Value Decomposition (SVD)
Generates: presentation/MFDS_LSA_Presentation.pptx complete with math equations,
figures, benchmark metrics, observations, conclusions, and presenter notes.

Usage:
    python generate_presentation.py
    (or run generate_presentation.bat)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    print("=" * 65)
    print("  MFDS LSA Lab: Generating Presentation Deck")
    print("=" * 65)

    from lsa_lab import report, figures, deck

    print("\n[1/3] Running numerical experiments and computing metrics...")
    out = report.build(verbose=True)
    metrics_path = report.save(out["results"])
    print(f"      -> Saved metrics to {metrics_path.relative_to(ROOT)}")

    print("\n[2/3] Rendering publication-quality figures...")
    news_k = out["experiments"]["news"].corpus.default_k
    figs = figures.make_all(
        ROOT / "figures",
        out["experiments"],
        news_k,
        out["sweep"],
        out["comparisons"]["news"],
    )
    print(f"      -> Generated {len(figs['paths'])} figures in figures/")

    print("\n[3/3] Building PowerPoint presentation with equations and speaker notes...")
    deck_path = deck.build(out, figs)
    print(f"\nSUCCESS! Presentation created at:")
    print(f"  -> {deck_path.resolve()}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
