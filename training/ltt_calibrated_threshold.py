"""
LTT-calibrated threshold selection for the hybrid_gate confidence band.

Replaces the ad hoc [0.60, 0.80] rule_based confidence band (chosen by
inspection in training/confidence_band_analysis.py on the 250-image
held_out set) with a Learn-Then-Test (LTT)-style procedure that provides
a Clopper-Pearson exact-binomial lower confidence bound on selective
accuracy.

This is an OFFLINE, READ-ONLY analysis. It does NOT modify
serving/gateway_service.py, rule_based, or hybrid_gate at serving time.
It computes and justifies threshold(s) whose output should be compared
against the existing 0.60/0.80 choice.

See requirements in task description for full background.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS_PATH = BASE_DIR / "router" / "utility_labels.csv"
HELD_OUT_PATH = BASE_DIR / "router" / "held_out_images.csv"

SEED = 42

# ---------------------------------------------------------------------------
# Clopper-Pearson lower bound (90% confidence, one-sided)
# ---------------------------------------------------------------------------

def clopper_pearson_lower(k: int, n: int, alpha: float = 0.10) -> float:
    """
    Exact binomial lower confidence bound at level 1-alpha.
    Uses statsmodels proportion_confint with method='beta' (Clopper-Pearson).

    For n==0 -> 0.0, k==0 -> 0.0 (conservative), k==n -> (alpha)**(1/n) approx but
    beta interval handles it.
    """
    if n == 0:
        return 0.0
    # statsmodels is guaranteed available (requirements_wsl.txt)
    from statsmodels.stats.proportion import proportion_confint

    lower, _ = proportion_confint(count=k, nobs=n, alpha=alpha, method="beta")
    # proportion_confint can return nan for edge cases; fall back to 0.0
    if np.isnan(lower):
        return 0.0
    return float(lower)


def selective_stats(df: pd.DataFrame, mask: pd.Series):
    """
    Given a boolean mask selecting a subset, compute n, k (correct), empirical accuracy.
    df must have 'fast_correct' column (0/1).
    """
    n = int(mask.sum())
    if n == 0:
        return n, 0, 0.0
    k = int(df.loc[mask, "fast_correct"].sum())
    acc = k / n if n else 0.0
    return n, k, acc


def print_table_header(title: str):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    print(f"{'lambda':>7}  {'n_covered':>9}  {'n_total':>7}  {'coverage':>8}  {'k_correct':>9}  {'R_hat':>7}  {'CP_lower(90%)':>13}  {'target':>7}  {'pass?':>5}")
    print("-" * 96)


def run_sweep(df: pd.DataFrame, grid: np.ndarray, target: float, alpha: float, upper: bool = True):
    """
    Sweep candidate thresholds.

    upper=True:  selective set = confidence >= lambda  (confident fast, want P(fast correct) high)
                 loosest = lowest lambda that still passes.

    upper=False: selective set = confidence <= lambda  (confident hard, want P(fast incorrect) high)
                 Here R_hat = P(fast incorrect | conf <= lambda) = 1 - P(fast correct)
                 loosest = highest lambda that still passes (covers most low-conf images).
    """
    rows = []
    for lam in grid:
        if upper:
            mask = df["fast_confidence"] >= lam
            n, k, r_hat = selective_stats(df, mask)
            # k is fast_correct count, R_hat = k/n
        else:
            mask = df["fast_confidence"] <= lam
            n = int(mask.sum())
            if n == 0:
                k_inc = 0
                r_hat = 0.0
            else:
                k_correct = int(df.loc[mask, "fast_correct"].sum())
                k_inc = n - k_correct  # fast incorrect count
                r_hat = k_inc / n
            # for lower bound we treat k_inc successes out of n trials
            k = k_inc

        lower = clopper_pearson_lower(k, n, alpha=alpha) if n > 0 else 0.0
        # pass if lower bound >= target AND enough samples (require n>=10 to avoid tiny-sample fluke)
        passes = (lower >= target) and (n >= 10)
        rows.append(
            {
                "lambda": round(float(lam), 2),
                "n_covered": n,
                "n_total": len(df),
                "coverage": n / len(df) if len(df) else 0.0,
                "k": k,
                "r_hat": r_hat,
                "lower": lower,
                "pass": passes,
            }
        )
    return rows


def print_sweep_rows(rows, target: float):
    for r in rows:
        flag = "PASS" if r["pass"] else "FAIL"
        print(
            f"{r['lambda']:7.2f}  {r['n_covered']:9d}  {r['n_total']:7d}  {r['coverage']:8.3f}  "
            f"{r['k']:9d}  {r['r_hat']:7.4f}  {r['lower']:13.4f}  {target:7.4f}  {flag:>5}"
        )


def main():
    print("=" * 96)
    print("LTT-CALIBRATED THRESHOLD ANALYSIS")
    print("Replaces ad hoc [0.60, 0.80] band with Clopper-Pearson selective-accuracy guarantee")
    print("=" * 96)

    # ------------------------------------------------------------------
    # 1. Calibration set selection — which fresh split, why appropriate
    # ------------------------------------------------------------------
    print("\n[1] CALIBRATION SET SELECTION")
    print("-" * 96)
    print("Requirement: calibration set DISTINCT from both:")
    print("  (a) the 4000-sample training set in router/utility_labels.csv (minus hold-out), and")
    print("  (b) the 250-image held_out_images.csv (used for confident-zone validation + McNemar).")
    print("  Reusing either would invalidate the LTT guarantee (data leakage / selection bias).")
    print()
    print("Checking already-available fresh splits:")
    print("  - make_splits.py val/ split: data/val/ — 4385 images, but these are raw images,")
    print("    not the utility_labels.csv rows with fast_confidence/fast_correct cached. Would")
    print("    require re-running Fast inference; not the correctness-gate's own calibration slice.")
    print("  - correctness_gate val split (train_correctness_gate.py:305-311):")
    print("    train_correctness_gate.py does stratified 70/15/15 split on utility_labels.csv")
    print("    (best_tier stratify, seed=42): train 2800 / val 600 / test 600.")
    print("    This val_df (600 images) is:")
    print("      * Disjoint from the 250-image held_out_images.csv — held_out was carved")
    print("        BEFORE sampling the 4000 working set in generate_utility_labels.py:191-228")
    print("        (hold_out = all_images[:250], remainder -> 4000 working set). So val 600")
    print("        is a subset of the 4000, which itself excludes the 250 hold-out.")
    print("      * NOT used to pick [0.60,0.80] — that band was chosen in")
    print("        training/confidence_band_analysis.py:120 on the 250 held_out set")
    print("        (histogram of Fast confidence on held_out_images.csv, recommended")
    print("        +/-0.10 capturing 18.4%). See monitoring/reports/confidence_band_analysis.csv.")
    print("      * Not used for the original 0.70 single-threshold either (rule_based)")
    print("        which is a fixed constant, not tuned on this val split.")
    print()
    print("Decision: USE correctness_gate val split (600 images) as calibration set.")
    print("Reason: fresh w.r.t. hold-out, already has fast_confidence/fast_correct cached in")
    print("utility_labels.csv (no recompute needed), and matches the exact split logic")
    print("used to train CorrectnessGateNetV2b — guarantees disjointness and reproducibility.")
    print("-" * 96)

    # ------------------------------------------------------------------
    # 2. Load data — reuse utility_labels.csv columns, no recompute
    # ------------------------------------------------------------------
    print("\n[2] CALIBRATION DATA — REUSE fast_confidence FROM utility_labels.csv")
    print("-" * 96)
    # Check generation logic first
    print("Checking router/utility_labels.csv columns vs generate_utility_labels.py logic ...")
    df_all = pd.read_csv(LABELS_PATH)
    print(f"Loaded {len(df_all)} rows from {LABELS_PATH}")
    # also load held_out for size check
    df_holdout_meta = pd.read_csv(HELD_OUT_PATH)
    print(f"Loaded {len(df_holdout_meta)} rows from {HELD_OUT_PATH} (distinct hold-out)")

    # Verify fast_confidence already available — do not recompute
    assert "fast_confidence" in df_all.columns, "fast_confidence missing from utility_labels.csv"
    assert "fast_correct" in df_all.columns, "fast_correct missing"
    print("Columns present: fast_confidence, fast_correct, fast_latency_ms, best_tier, etc. -> REUSE, no recompute.")
    # Normalize paths if needed (utility_labels uses Windows backslashes? check)
    # Already stored as data\\test\\... but fast_confidence is numeric, no path issue here.

    # Replicate EXACT split from train_correctness_gate.py
    from sklearn.model_selection import train_test_split as sk_split

    train_df, temp_df = sk_split(df_all, test_size=0.30, stratify=df_all["best_tier"], random_state=SEED)
    val_df, test_df = sk_split(temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED)
    print(f"\nReplicated train_correctness_gate.py split (seed={SEED}, stratify best_tier):")
    print(f"  train: {len(train_df)}  val (CALIBRATION): {len(val_df)}  test: {len(test_df)}  total: {len(df_all)}")
    print(f"  Calibration set = val_df (600) — disjoint from held_out 250, disjoint from train 2800.")
    # distribution
    print(f"\nCalibration set best_tier distribution:")
    print(val_df["best_tier"].value_counts().to_string())
    print(f"\nCalibration set fast_correct rate (base accuracy of Fast tier): {val_df['fast_correct'].mean():.4f}")
    print(f"Calibration set fast_confidence stats: min {val_df['fast_confidence'].min():.4f} "
          f"max {val_df['fast_confidence'].max():.4f} mean {val_df['fast_confidence'].mean():.4f} "
          f"median {val_df['fast_confidence'].median():.4f}")

    # Also show overlap check with held-out
    calib_paths = set(val_df["image_path"].str.replace("\\", "/", regex=False))
    holdout_paths = set(df_holdout_meta["image_path"].str.replace("\\", "/", regex=False))
    overlap = calib_paths.intersection(holdout_paths)
    print(f"\nOverlap calibration vs held_out: {len(overlap)} images (expected 0) -> {'PASS disjoint' if len(overlap)==0 else 'FAIL overlap!'}")

    calibration_df = val_df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. LTT threshold search — upper threshold (>= lambda)
    # ------------------------------------------------------------------
    print("\n[3] LTT SELECTIVE-ACCURACY THRESHOLD SEARCH")
    print("-" * 96)
    print("Hybrid gate band in serving/gateway_service.py (route_hybrid_gate / route_hybrid_dumb):")
    print("  if fast_confidence < 0.60 or fast_confidence > 0.80:")
    print("      # outside uncertain band -> rule_based (Fast if >=0.70 else Heavy)")
    print("  else: # inside [0.60,0.80] -> delegate to correctness_gate (or dumb Heavy)")
    print("  RULE_BASED_CONFIDENCE_THRESHOLD = 0.70")
    print("So two boundaries: lower=0.60 (confident hard -> escalate), upper=0.80 (confident easy -> fast).")
    print("We mirror BOTH, with Clopper-Pearson 90% lower bound (alpha=0.10, method='beta').")
    print()

    # First, what does the existing 0.70 single point achieve on calibration?
    mask_07 = calibration_df["fast_confidence"] >= 0.70
    n07, k07, acc07 = selective_stats(calibration_df, mask_07)
    lower07 = clopper_pearson_lower(k07, n07, alpha=0.10)
    print(f"Baseline on CALIBRATION set (single threshold 0.70, route to fast if >=0.70):")
    print(f"  n_covered (conf>=0.70): {n07}/{len(calibration_df)} ({n07/len(calibration_df):.3f})")
    print(f"  empirical selective accuracy R_hat = {k07}/{n07} = {acc07:.4f}")
    print(f"  Clopper-Pearson lower bound (90%): {lower07:.4f}")
    print()
    # Also check existing band boundaries on calibration
    for lam in [0.60, 0.80]:
        mask = calibration_df["fast_confidence"] >= lam
        n, k, acc = selective_stats(calibration_df, mask)
        lb = clopper_pearson_lower(k, n, alpha=0.10)
        print(f"  conf>={lam:.2f}: n={n:3d} acc={acc:.4f} lower={lb:.4f} (coverage {n/len(calibration_df):.3f})")
    print()
    mask_low_06 = calibration_df["fast_confidence"] <= 0.60
    n06_low, k06_corr, acc06_low = selective_stats(calibration_df, mask_low_06)  # this is fast_correct rate, not incorrect
    # for lower we want fast incorrect rate
    k_inc_06 = n06_low - k06_corr
    r_inc_06 = k_inc_06 / n06_low if n06_low else 0.0
    lb_inc_06 = clopper_pearson_lower(k_inc_06, n06_low, alpha=0.10)
    print(f"  conf<=0.60 (existing lower, escalate): n={n06_low} fast_correct={k06_corr} fast_incorrect={k_inc_06} "
          f"P(fast wrong|conf<=0.60)={r_inc_06:.4f} lower={lb_inc_06:.4f}")
    print()

    # Choose target near but not above 0.70 baseline, so comparison meaningful
    # If acc07 ~0.97-0.99, set target 0.97. If acc07 lower, set target acc07 - 0.01 or 0.97 whichever lower.
    print("Choosing LTT target selective accuracy:")
    print(f"  Observed at 0.70 on calibration: {acc07:.4f} (lower bound {lower07:.4f})")
    # Reasoning: pick target slightly below empirical at 0.70 so that a threshold exists,
    # but still at least 0.97 if 0.70 itself achieves >=0.97; otherwise pick 0.95 or acc07-0.02.
    if acc07 >= 0.97:
        target_upper = 0.97
        reason = f"0.70 achieves {acc07:.4f} >=0.97, so target 0.97 is both ambitious and feasible (just below observed)."
    elif acc07 >= 0.95:
        target_upper = 0.95
        reason = f"0.70 achieves {acc07:.4f} in [0.95,0.97), so target 0.95 chosen (cannot demand 0.97 above observed)."
    else:
        target_upper = round(max(0.90, acc07 - 0.02), 2)
        reason = f"0.70 only achieves {acc07:.4f}, so target set to {target_upper:.2f} (acc07 - 0.02, floor 0.90) to stay meaningful."

    # Also decide lower target: for fast-incorrect rate among low conf, what is reasonable?
    # At <=0.60, fast_incorrect rate is maybe 0.4-0.6? Might not reach 0.97. So we pick a lower target for lower side,
    # e.g., 0.70 or r_inc_06 - 0.05. Print reasoning.
    # Let's compute baseline at <=0.60 fast-incorrect
    if r_inc_06 >= 0.70:
        target_lower = 0.70
        reason_low = f"fast_incorrect at <=0.60 is {r_inc_06:.4f} >=0.70, so target 0.70 for P(fast wrong | low conf)."
    elif r_inc_06 >= 0.50:
        target_lower = 0.50
        reason_low = f"fast_incorrect at <=0.60 is {r_inc_06:.4f}, so target 0.50 (escalation justified if >50% would be wrong on fast)."
    else:
        target_lower = 0.40
        reason_low = f"fast_incorrect at <=0.60 is only {r_inc_06:.4f}, so low-confidence escalation has weak signal; target 0.40."

    print(f"  -> Target UPPER (P(fast correct | conf>=lambda) >= target): {target_upper:.4f}")
    print(f"     Reason: {reason}")
    print(f"  -> Target LOWER (P(fast incorrect | conf<=lambda) >= target): {target_lower:.4f}")
    print(f"     Reason: {reason_low}")
    print(f"  (Both at 90% Clopper-Pearson confidence, alpha=0.10, method='beta')")
    print()

    grid_upper = np.arange(0.50, 0.951, 0.01)
    grid_upper = np.round(grid_upper, 2)
    grid_upper = np.unique(grid_upper)

    rows_upper = run_sweep(calibration_df, grid_upper, target_upper, alpha=0.10, upper=True)

    print_table_header(f"UPPER THRESHOLD SWEEP — P(fast correct | conf >= lambda) >= {target_upper:.2f} (calibration N={len(calibration_df)})")
    print_sweep_rows(rows_upper, target_upper)

    # Select loosest passing: lowest lambda that passes (covers most images)
    passing_upper = [r for r in rows_upper if r["pass"]]
    if passing_upper:
        chosen_upper = min(passing_upper, key=lambda r: r["lambda"])
        print(f"\nChosen UPPER lambda (LOOSEST passing, lowest lambda, max coverage): {chosen_upper['lambda']:.2f}")
        print(f"  n_covered={chosen_upper['n_covered']} coverage={chosen_upper['coverage']:.3f} "
              f"R_hat={chosen_upper['r_hat']:.4f} lower={chosen_upper['lower']:.4f} >= {target_upper:.4f}")
    else:
        chosen_upper = None
        print(f"\nNo UPPER lambda passes target {target_upper:.4f} at 90% confidence — would need to lower target or accept no confident-fast region.")

    # ------------------------------------------------------------------
    # Lower threshold sweep
    # ------------------------------------------------------------------
    # Grid for lower: 0.30 to 0.75 is sensible (existing 0.60). We'll do 0.40-0.75 step 0.01 to mirror upper.
    grid_lower = np.arange(0.40, 0.751, 0.01)
    grid_lower = np.round(grid_lower, 2)
    grid_lower = np.unique(grid_lower)

    rows_lower = run_sweep(calibration_df, grid_lower, target_lower, alpha=0.10, upper=False)

    print_table_header(f"LOWER THRESHOLD SWEEP — P(fast WRONG | conf <= lambda) >= {target_lower:.2f} (calibration N={len(calibration_df)})")
    print_sweep_rows(rows_lower, target_lower)

    passing_lower = [r for r in rows_lower if r["pass"]]
    if passing_lower:
        # loosest = highest lambda that passes (covers most low-conf images)
        chosen_lower = max(passing_lower, key=lambda r: r["lambda"])
        print(f"\nChosen LOWER lambda (LOOSEST passing, highest lambda, max low-coverage): {chosen_lower['lambda']:.2f}")
        print(f"  n_covered={chosen_lower['n_covered']} coverage={chosen_lower['coverage']:.3f} "
              f"R_hat (fast wrong)={chosen_lower['r_hat']:.4f} lower={chosen_lower['lower']:.4f} >= {target_lower:.4f}")
    else:
        chosen_lower = None
        print(f"\nNo LOWER lambda passes target {target_lower:.4f} at 90% confidence.")

    # ------------------------------------------------------------------
    # 5. Comparison vs existing [0.60,0.80] — calibration coverage
    # ------------------------------------------------------------------
    print("\n[5] COMPARISON — EXISTING vs LTT-DERIVED THRESHOLDS")
    print("-" * 96)
    existing_low, existing_high = 0.60, 0.80
    if chosen_upper is not None and chosen_lower is not None:
        ltt_low, ltt_high = chosen_lower["lambda"], chosen_upper["lambda"]
        print(f"Existing band: [{existing_low:.2f}, {existing_high:.2f}]  (uncertain = inside band -> delegate to gate)")
        print(f"LTT-derived band: [{ltt_low:.2f}, {ltt_high:.2f}]  (targets {target_lower:.2f} low / {target_upper:.2f} high at 90% CP)")
    elif chosen_upper is not None:
        ltt_low, ltt_high = None, chosen_upper["lambda"]
        print(f"Existing band: [{existing_low:.2f}, {existing_high:.2f}]")
        print(f"LTT-derived UPPER only: {ltt_high:.2f} (lower had no passing threshold at target {target_lower:.2f})")
        print(f"  -> LTT lower would retain existing {existing_low:.2f} or be flagged as not learnable at this target.")
    else:
        ltt_low, ltt_high = None, None
        print("No LTT thresholds passed — comparison degenerates to existing band only.")

    # Calibration set: how many images classified as confident under each?
    def band_stats(df, low, high):
        # confident fast: conf > high, confident hard: conf < low, uncertain: in [low, high]
        n_fast_conf = int((df["fast_confidence"] > high).sum()) if high is not None else 0
        n_hard_conf = int((df["fast_confidence"] < low).sum()) if low is not None else 0
        n_uncertain = len(df) - n_fast_conf - n_hard_conf
        # handle inclusive edges for hybrid: hybrid uses <0.60 or >0.80 outside, so uncertain is [0.60,0.80] inclusive
        # We'll report inclusive [low, high] as uncertain for apples-to-apples
        n_uncertain_inc = int(((df["fast_confidence"] >= low) & (df["fast_confidence"] <= high)).sum())
        n_fast_conf_inc = int((df["fast_confidence"] > high).sum())
        n_hard_conf_inc = int((df["fast_confidence"] < low).sum())
        return n_fast_conf_inc, n_hard_conf_inc, n_uncertain_inc

    print(f"\nOn CALIBRATION set (N={len(calibration_df)}, disjoint, used for guarantee):")
    n_fast_e, n_hard_e, n_unc_e = band_stats(calibration_df, existing_low, existing_high)
    print(f"  Existing [{existing_low:.2f},{existing_high:.2f}]: confident-fast >{existing_high:.2f}: {n_fast_e} ({n_fast_e/len(calibration_df):.3f}), "
          f"confident-hard <{existing_low:.2f}: {n_hard_e} ({n_hard_e/len(calibration_df):.3f}), "
          f"uncertain [{existing_low:.2f},{existing_high:.2f}]: {n_unc_e} ({n_unc_e/len(calibration_df):.3f})")
    if ltt_low is not None and ltt_high is not None:
        n_fast_l, n_hard_l, n_unc_l = band_stats(calibration_df, ltt_low, ltt_high)
        print(f"  LTT      [{ltt_low:.2f},{ltt_high:.2f}]: confident-fast >{ltt_high:.2f}: {n_fast_l} ({n_fast_l/len(calibration_df):.3f}), "
              f"confident-hard <{ltt_low:.2f}: {n_hard_l} ({n_hard_l/len(calibration_df):.3f}), "
              f"uncertain [{ltt_low:.2f},{ltt_high:.2f}]: {n_unc_l} ({n_unc_l/len(calibration_df):.3f})")
        print(f"  Delta uncertain: {n_unc_l - n_unc_e:+d} images ({(n_unc_l - n_unc_e)/len(calibration_df):+.3f})")

    # ------------------------------------------------------------------
    # Apply to ORIGINAL 250 hold-out set — ILLUSTRATIVE ONLY, clearly labeled
    # ------------------------------------------------------------------
    print("\nOn ORIGINAL 250 hold-out set (monitoring/reports/confidence_band_analysis.csv source) —")
    print("ILLUSTRATIVE ONLY, NOT part of calibration/guarantee (hold-out was used to pick [0.60,0.80] by inspection,")
    print("so evaluating LTT thresholds here is optimistic/biased; shown only for qualitative comparison):")
    print("-" * 96)

    # Need fast_confidence for hold-out. Utility_labels does NOT contain these 250 rows (they were excluded).
    # So we must compute fast_confidence for hold-out by either (a) loading Fast model or (b) if Fast model not
    # available, reporting that hold-out confidences are not in utility_labels and showing what can be shown.
    # We'll try to load Fast model; if unavailable, we fall back to reporting via confidence_band_analysis histogram.

    holdout_df_raw = pd.read_csv(HELD_OUT_PATH)
    # Normalize paths
    holdout_df_raw["image_path_norm"] = holdout_df_raw["image_path"].str.replace("\\", "/", regex=False)

    # Try to get per-image confidence for hold-out: first check if any hold-out image_path appears in utility_labels
    # (should be 0 overlap, verify)
    util_paths = set(df_all["image_path"].str.replace("\\", "/", regex=False))
    holdout_overlap_in_util = sum(1 for p in holdout_df_raw["image_path_norm"] if p in util_paths)
    print(f"Hold-out overlap with utility_labels: {holdout_overlap_in_util}/250 (expected 0, confirms disjoint)")

    # Attempt to run Fast inference for hold-out if checkpoint exists and torch available
    fast_ckpt_path = BASE_DIR / "training" / "checkpoints" / "fast_cnn_best.pt"
    holdout_confidences = None
    if fast_ckpt_path.exists():
        try:
            import torch
            import torch.nn.functional as F
            from PIL import Image
            from torchvision.transforms import v2

            # Need FastCNN class
            sys.path.insert(0, str(BASE_DIR / "training"))
            from models.fast_cnn import FastCNN

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(fast_ckpt_path, map_location=device)
            class_names = ckpt["class_names"] if isinstance(ckpt, dict) and "class_names" in ckpt else None
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            num_classes = len(class_names) if class_names else 28
            model = FastCNN(num_classes=num_classes)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            preprocess = v2.Compose([
                v2.ToImage(),
                v2.Resize(256),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            confidences = []
            missing = 0
            for _, row in holdout_df_raw.iterrows():
                img_path = BASE_DIR / row["image_path_norm"]
                if not img_path.exists():
                    missing += 1
                    continue
                img = Image.open(img_path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1)
                    conf = float(torch.max(probs, dim=1)[0].item())
                confidences.append(conf)
            if missing:
                print(f"  Warning: {missing} hold-out images missing on disk, skipped.")
            holdout_df = pd.DataFrame({"fast_confidence": confidences})
            print(f"  Computed Fast confidence for {len(holdout_df)}/250 hold-out images via {fast_ckpt_path} on {device}")
            holdout_confidences = holdout_df["fast_confidence"]
        except Exception as e:
            print(f"  Could not compute hold-out confidences via Fast model: {e}")
            print(f"  Falling back to histogram-level comparison from confidence_band_analysis.csv")
            holdout_confidences = None
    else:
        print(f"  Fast checkpoint not found at {fast_ckpt_path}, cannot compute per-image hold-out confidences.")
        holdout_confidences = None

    if holdout_confidences is not None and len(holdout_confidences) > 0:
        # Use computed confidences for band stats (illustrative)
        # Create a dummy df with fast_confidence column
        dummy_holdout = pd.DataFrame({"fast_confidence": holdout_confidences})
        n_fast_e_h, n_hard_e_h, n_unc_e_h = band_stats(dummy_holdout, existing_low, existing_high)
        print(f"  Existing [{existing_low:.2f},{existing_high:.2f}]: fast-conf {n_fast_e_h} ({n_fast_e_h/len(dummy_holdout):.3f}), "
              f"hard-conf {n_hard_e_h} ({n_hard_e_h/len(dummy_holdout):.3f}), uncertain {n_unc_e_h} ({n_unc_e_h/len(dummy_holdout):.3f})")
        if ltt_low is not None and ltt_high is not None:
            n_fast_l_h, n_hard_l_h, n_unc_l_h = band_stats(dummy_holdout, ltt_low, ltt_high)
            print(f"  LTT      [{ltt_low:.2f},{ltt_high:.2f}]: fast-conf {n_fast_l_h} ({n_fast_l_h/len(dummy_holdout):.3f}), "
                  f"hard-conf {n_hard_l_h} ({n_hard_l_h/len(dummy_holdout):.3f}), uncertain {n_unc_l_h} ({n_unc_l_h/len(dummy_holdout):.3f})")
            print(f"  Delta uncertain on hold-out: {n_unc_l_h - n_unc_e_h:+d} ({(n_unc_l_h - n_unc_e_h)/len(dummy_holdout):+.3f})")
        print(f"  (Illustrative: hold-out was used to choose [0.60,0.80] in confidence_band_analysis.py, so this comparison is biased in favor of existing band.)")
    else:
        # Fallback: histogram comparison via monitoring/reports/confidence_band_analysis.csv
        cba_path = BASE_DIR / "monitoring" / "reports" / "confidence_band_analysis.csv"
        if cba_path.exists():
            print(f"  Loaded histogram from {cba_path}:")
            hist_df = pd.read_csv(cba_path)
            # SUMMARY row has capture fractions
            summary = hist_df[hist_df["bin_low"] == "SUMMARY"].iloc[0] if "SUMMARY" in hist_df["bin_low"].values else None
            if summary is not None:
                print(f"    Original band captures: +/-0.05 {summary['capture_+/-0.05']}, +/-0.10 {summary['capture_+/-0.10']} (= [0.60,0.80] 18.4%), "
                      f"+/-0.15 {summary['capture_+/-0.15']}, +/-0.20 {summary['capture_+/-0.20']}")
                print(f"    Histogram bins show distribution not per-image thresholds, so per-image LTT comparison requires Fast inference.")
            print(f"  Without per-image hold-out confidences, illustrative per-threshold counts cannot be computed exactly; see histogram above.")

    # ------------------------------------------------------------------
    # 6. Final verdict
    # ------------------------------------------------------------------
    print("\n[6] FINAL VERDICT — Does LTT agree with [0.60, 0.80]?")
    print("-" * 96)
    if chosen_upper is None:
        print("INCONCLUSIVE / CORRECTION NEEDED: No upper threshold meets target at 90% confidence.")
        print("This suggests the calibration set cannot guarantee 0.97 selective accuracy for ANY")
        print("confident-fast region at 90% CB, or the target is too stringent for N=600.")
        print("Recommendation: lower target (e.g., 0.95) or larger calibration set; flag for report")
        print("as a CORRECTION (existing 0.80 has no statistical guarantee).")
    else:
        diff_upper = chosen_upper["lambda"] - existing_high
        diff_str = f"{diff_upper:+.2f}"
        # Assess agreement: within 0.05 is roughly agree, 0.10 is meaningfully different
        if abs(diff_upper) <= 0.05:
            verdict_upper = "AGREES — within 0.05, validates existing 0.80 as reasonable."
        elif abs(diff_upper) <= 0.10:
            verdict_upper = "ROUGHLY AGREES — within 0.10, minor correction suggested."
        else:
            verdict_upper = "DIFFERS MEANINGFULLY — >0.10 away, flag as CORRECTION."

        print(f"Upper threshold: existing {existing_high:.2f} vs LTT {chosen_upper['lambda']:.2f} (delta {diff_str}) -> {verdict_upper}")
        print(f"  LTT guarantee: P(fast correct | conf>={chosen_upper['lambda']:.2f}) >= {target_upper:.2f} with 90% confidence over calibration sample.")
        print(f"  Existing 0.80 on calibration: n={(calibration_df['fast_confidence'] >= 0.80).sum()} acc≈{(calibration_df.loc[calibration_df['fast_confidence'] >= 0.80, 'fast_correct'].mean() if (calibration_df['fast_confidence'] >= 0.80).any() else 0):.4f}")

        if chosen_lower is not None:
            diff_lower = chosen_lower["lambda"] - existing_low
            print(f"Lower threshold: existing {existing_low:.2f} vs LTT {chosen_lower['lambda']:.2f} (delta {diff_lower:+.2f})")
            if abs(diff_lower) <= 0.05:
                print("  -> AGREES with existing lower.")
            else:
                print("  -> DIFFERS from existing lower; but lower-side signal is weaker (fast_incorrect is not strong predictor of heavy_correct),")
                print("     so verdict should weight upper more heavily.")
        else:
            print(f"Lower threshold: existing {existing_low:.2f} vs LTT none (no passing) — lower-side guarantee not achievable at target {target_lower:.2f}.")

        # Overall
        if chosen_upper is not None and abs(diff_upper) <= 0.05 and (chosen_lower is None or abs(chosen_lower["lambda"] - existing_low) <= 0.05):
            print("\nOVERALL VERDICT: LTT VALIDATES existing [0.60,0.80] as reasonable — no correction needed.")
            print("Report as SUPPORTING EVIDENCE (statistical justification for ad hoc choice).")
        elif chosen_upper is not None and abs(diff_upper) <= 0.10:
            print("\nOVERALL VERDICT: LTT ROUGHLY VALIDATES but suggests a minor adjustment (e.g., use "
                  f"{chosen_upper['lambda']:.2f} instead of {existing_high:.2f} for upper).")
            print("Report as MINOR CORRECTION / sensitivity note.")
        else:
            print("\nOVERALL VERDICT: LTT SUGGESTS A MEANINGFULLY DIFFERENT THRESHOLD.")
            print("Flag for report as CORRECTION. Replace [0.60,0.80] with LTT-derived band in future work,")
            print("and note that the original band has no finite-sample guarantee.")

    print("\n" + "=" * 96)
    print("END OF LTT ANALYSIS — serving/gateway_service.py UNCHANGED (offline analysis only)")
    print("=" * 96)


if __name__ == "__main__":
    main()
