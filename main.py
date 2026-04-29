"""
main.py
───────
Credit Card Fraud Detection System — Project Entry Point

Usage:
    python main.py --phase generate    # Generate synthetic data
    python main.py --phase ingest      # Ingest + validate data
    python main.py --phase all         # Run full pipeline (all phases)
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: str):
    print(f"\n{'─'*60}")
    print(f"▶  Running: {script}")
    print(f"{'─'*60}")
    result = subprocess.run([sys.executable, script], check=True)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection — Pipeline Runner"
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "ingest", "all"],
        default="generate",
        help="Which phase to run (default: generate)",
    )
    args = parser.parse_args()

    print("🔍 Credit Card Fraud Detection System")
    print("=" * 60)

    if args.phase in ("generate", "all"):
        run("generate_data.py")

    if args.phase in ("ingest", "all"):
        run("notebooks/01_ingest.py")

    if args.phase == "all":
        print("\n✅ Phase 1 complete!")
        print("   Next: python notebooks/02_eda.py  (Phase 2 — EDA)")


if __name__ == "__main__":
    main()