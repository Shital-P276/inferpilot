"""
McNemar's test: hybrid_gate vs always_heavy

Paired test on per-image correctness from the existing results file.
Tests whether the two strategies produce significantly different error
patterns on the same 250 held-out images.

Output: 2x2 contingency table, test statistic, p-value, interpretation.
"""
import sys
from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_RESULTS = BASE_DIR / "router" / "core_experiment_raw_results.csv"


def main():
    print("=" * 60)
    print("McNemar's Test: hybrid_gate vs always_heavy")
    print("=" * 60)

    # Step 1: Load CSV and inspect columns
    df = pd.read_csv(RAW_RESULTS)
    print(f"\nCSV columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    print(f"Strategies: {sorted(df['strategy'].unique())}")

    # Step 2: Extract paired correctness for each strategy
    hybrid = df[df["strategy"] == "hybrid_gate"].set_index("image_path")["correct"]
    heavy = df[df["strategy"] == "always_heavy"].set_index("image_path")["correct"]

    # Align on shared images (should be identical set)
    shared = hybrid.index.intersection(heavy.index)
    hybrid = hybrid.loc[shared]
    heavy = heavy.loc[shared]
    n = len(shared)
    print(f"\nPaired images: {n}")

    # Step 3: Build 2x2 contingency table
    #  Convention: rows = hybrid_gate, cols = always_heavy
    #
    #                    always_heavy    always_heavy
    #                    correct (1)     incorrect (0)
    #  hybrid correct     both_correct    hybrid_only
    #  hybrid incorrect   heavy_only      both_wrong
    #
    both_correct = int(((hybrid == 1) & (heavy == 1)).sum())
    hybrid_only = int(((hybrid == 1) & (heavy == 0)).sum())  # b: hybrid right, heavy wrong
    heavy_only = int(((hybrid == 0) & (heavy == 1)).sum())   # c: hybrid wrong, heavy right
    both_wrong = int(((hybrid == 0) & (heavy == 0)).sum())

    table = [[both_correct, hybrid_only],
             [heavy_only, both_wrong]]

    print(f"\n2x2 Contingency Table:")
    print(f"{'':30s}  always_heavy=1  always_heavy=0")
    print(f"{'hybrid_gate=1':30s}  {both_correct:>13d}  {hybrid_only:>13d}")
    print(f"{'hybrid_gate=0':30s}  {heavy_only:>13d}  {both_wrong:>13d}")

    # Step 4: Run McNemar's test
    # Discordant pairs = hybrid_only + heavy_only
    discordant = hybrid_only + heavy_only
    print(f"\nDiscordant pairs: {discordant} (hybrid_only={hybrid_only}, heavy_only={heavy_only})")

    if discordant < 25:
        use_exact = True
        reason = f"discordant count {discordant} < 25, using exact binomial test"
    else:
        use_exact = False
        reason = f"discordant count {discordant} >= 25, using chi-square approximation with continuity correction"

    print(f"Mode: {'exact' if use_exact else 'chi-square'} — {reason}")

    result = mcnemar(table, exact=use_exact, correction=not use_exact)

    # Step 5: Print results
    print(f"\nTest statistic: {result.statistic:.4f}")
    print(f"p-value:        {result.pvalue:.6f}")

    alpha = 0.05
    if result.pvalue < alpha:
        interp = (f"p={result.pvalue:.6f} < alpha={alpha}: difference IS statistically significant. "
                  f"hybrid_gate and always_heavy have different error patterns.")
    else:
        interp = (f"p={result.pvalue:.6f} >= alpha={alpha}: difference is NOT statistically significant. "
                  f"No evidence of different error patterns between hybrid_gate and always_heavy.")

    print(f"\nInterpretation: {interp}")

    # Summary
    print(f"\nSummary:")
    print(f"  hybrid_gate accuracy:   {hybrid.mean():.4f} ({int(hybrid.sum())}/{n})")
    print(f"  always_heavy accuracy:  {heavy.mean():.4f} ({int(heavy.sum())}/{n})")
    print(f"  McNemar p-value:        {result.pvalue:.6f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
