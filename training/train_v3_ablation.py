"""
training/train_v3_ablation.py

V3 ablation: which of V3's two changes drove the AUROC gain over V2b?
  (a) unfrozen last backbone block, (b) auxiliary confidence head.

2x2 design (backbone x heads):
  frozen  + single = V2b (checkpoint exists: correctness_gate_v2b_best.pt)
  unfrozen+ dual   = V3  (checkpoint exists: correctness_gate_v3_best.pt)
  unfrozen+ single = V3a (trained here -> correctness_gate_v3a_best.pt)
  frozen  + dual   = V3b (trained here -> correctness_gate_v3b_best.pt)

Training mirrors train_correctness_gate_v3.py EXACTLY (same 70/15/15 split
seed=42 stratify best_tier, same transforms, AdamW 1e-3 / wd 1e-4 /
CosineAnnealing T_max=25 epochs / batch 32, per-tier weighted BCE normalized
sum-to-3, confidence MSE weight 0.5 where a confidence head exists,
checkpoint-by-avg-AUROC) so the only differences between cells are the two
ablated factors.

The CorrectnessGateNetV3 architecture change enabling this (unfreeze_last_block
/ dual_head flags, both default True) preserves the published V3 architecture
and state-dict keys -- verified: published V3 checkpoint loads strict.

Run:  source ~/venv_inferpilot/bin/activate
      python training/train_v3_ablation.py            # train V3a+V3b, then report
      python training/train_v3_ablation.py --report-only   # re-evaluate all 4 cells
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.models.correctness_gate import (  # noqa: E402
    CorrectnessGateNetV2b,
    CorrectnessGateNetV3,
)
from train_correctness_gate_v3 import (  # noqa: E402
    CorrectnessGateV3Dataset,
    compute_bce_weights,
)
from train_correctness_gate import (  # noqa: E402
    per_tier_metrics,
    BASE_DIR, LABELS_PATH, CHECKPOINT_DIR,
    TARGET_COLUMNS, TIER_NAMES,
    train_transform, eval_transform,
)

EPOCHS = 25
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
CONFIDENCE_LOSS_WEIGHT = 0.5  # same as V3; unused for single-head cells
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CELLS = {
    # Only the two missing cells are trained here; v2b/v3 load published ckpts.
    "v3a": {"flags": {"unfreeze_last_block": True, "dual_head": False},
            "ckpt": CHECKPOINT_DIR / "correctness_gate_v3a_best.pt",
            "desc": "unfrozen backbone + single correctness head"},
    "v3b": {"flags": {"unfreeze_last_block": False, "dual_head": True},
            "ckpt": CHECKPOINT_DIR / "correctness_gate_v3b_best.pt",
            "desc": "frozen backbone + dual heads (correctness + confidence aux)"},
}
PUBLISHED = {
    "v2b": {"cls": CorrectnessGateNetV2b, "flags": {},
            "ckpt": CHECKPOINT_DIR / "correctness_gate_v2b_best.pt",
            "desc": "frozen backbone + single head (published)"},
    "v3": {"cls": CorrectnessGateNetV3, "flags": {},
           "ckpt": CHECKPOINT_DIR / "correctness_gate_v3_best.pt",
           "desc": "unfrozen backbone + dual heads (published)"},
}


def split_data():
    df = pd.read_csv(LABELS_PATH)
    df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["best_tier"], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["best_tier"], random_state=SEED)
    return train_df, val_df, test_df


def unwrap_corr(model, images):
    out = model(images)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def train_cell(name, cfg, train_loader, val_loader, bce_weights):
    model = CorrectnessGateNetV3(num_tiers=3, **cfg["flags"]).to(DEVICE)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{name}] {cfg['desc']} | trainable params: {n_trainable:,}")
    for p_name, p in model.named_parameters():
        if p.requires_grad and ("backbone" in p_name):
            print(f"  trainable backbone: {p_name} shape {tuple(p.shape)}")

    dual = cfg["flags"]["dual_head"]
    mse = nn.MSELoss()
    bce_none = nn.BCELoss(reduction="none")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_auroc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        run_loss, total = 0.0, 0
        for images, corr_t, conf_t in train_loader:
            images, corr_t, conf_t = (images.to(DEVICE), corr_t.to(DEVICE),
                                      conf_t.to(DEVICE))
            opt.zero_grad()
            corr_p, conf_p = unwrap_corr(model, images)
            w = bce_weights.to(DEVICE)
            loss = (bce_none(corr_p, corr_t) * w).mean()
            if dual:
                loss = loss + CONFIDENCE_LOSS_WEIGHT * mse(conf_p, conf_t)
            loss.backward()
            opt.step()
            run_loss += loss.item() * images.size(0)
            total += images.size(0)
        sched.step()

        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for images, corr_t, _ in val_loader:
                corr_p, _ = unwrap_corr(model, images.to(DEVICE))
                all_p.append(corr_p.cpu())
                all_t.append(corr_t)
        yp = torch.cat(all_p).numpy()
        yt = torch.cat(all_t).numpy()
        _, aurocs = per_tier_metrics(yt, yp)
        avg = float(np.nanmean(aurocs))
        print(f"  epoch {epoch:2d}/{EPOCHS} loss {run_loss/total:.4f} "
              f"val_avg_auroc {avg:.4f} "
              f"(f/b/h {aurocs[0]:.3f}/{aurocs[1]:.3f}/{aurocs[2]:.3f}) "
              f"{time.time()-t0:.1f}s")
        if avg > best_auroc:
            best_auroc = avg
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    torch.save({"model_state_dict": best_state, "val_avg_auroc": best_auroc,
                "flags": cfg["flags"]}, cfg["ckpt"])
    print(f"  -> saved {cfg['ckpt']} (val_avg_auroc={best_auroc:.4f})")


@torch.no_grad()
def eval_on_test(model, loader):
    model.eval()
    all_p, all_t = [], []
    for images, corr_t, _ in loader:
        corr_p, _ = unwrap_corr(model, images.to(DEVICE))
        all_p.append(corr_p.cpu())
        all_t.append(corr_t)
    yp = torch.cat(all_p).numpy()
    yt = torch.cat(all_t).numpy()
    aurocs = [float(roc_auc_score(yt[:, t], yp[:, t])) for t in range(3)]
    return aurocs


def report(test_loader):
    print("\n" + "=" * 70 + "\nABLATION REPORT: 2x2 (backbone x heads), same test partition\n" + "=" * 70)
    table = {}
    for name, cfg in {**PUBLISHED, **CELLS}.items():
        cls = cfg.get("cls", CorrectnessGateNetV3)
        model = cls(num_tiers=3, **cfg.get("flags", {})).to(DEVICE)
        ckpt = torch.load(cfg["ckpt"], map_location=DEVICE)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd)
        aurocs = eval_on_test(model, test_loader)
        table[name] = aurocs
        print(f"  {name:4s} ({cfg['desc']}): "
              f"fast={aurocs[0]:.4f} balanced={aurocs[1]:.4f} heavy={aurocs[2]:.4f} "
              f"avg={np.mean(aurocs):.4f}")
    print("\n  backbone\\heads | single          | dual (aux conf)")
    print("  ----------------+-----------------+------------------")
    print(f"  frozen         | v2b  avg={np.mean(table['v2b']):.4f}  | "
          f"v3b  avg={np.mean(table['v3b']):.4f}")
    print(f"  unfrozen       | v3a  avg={np.mean(table['v3a']):.4f}  | "
          f"v3   avg={np.mean(table['v3']):.4f}")
    d_unfr = np.mean(table['v3a']) - np.mean(table['v2b'])
    d_dual = np.mean(table['v3b']) - np.mean(table['v2b'])
    d_both = np.mean(table['v3']) - np.mean(table['v2b'])
    print(f"\n  Effect of unfreezing alone (v3a-v2b): {d_unfr:+.4f}")
    print(f"  Effect of aux head alone  (v3b-v2b): {d_dual:+.4f}")
    print(f"  Effect of both            (v3-v2b):  {d_both:+.4f}")
    if d_unfr > 0.005 and d_dual < 0.005:
        print("  VERDICT: unfreezing drove the gain; aux head contributed ~nothing alone.")
    elif d_dual > 0.005 and d_unfr < 0.005:
        print("  VERDICT: aux head drove the gain; unfreezing contributed ~nothing alone.")
    elif d_unfr > 0.005 and d_dual > 0.005:
        print("  VERDICT: both factors contribute; compare magnitudes above.")
    else:
        print("  VERDICT: neither factor alone reproduces the gain -- joint effect "
              "(or noise); report as such, do not force a narrative.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--cells", default="v3a,v3b")
    args = ap.parse_args()

    print(f"Using device: {DEVICE}")
    train_df, val_df, test_df = split_data()
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)} "
          f"(seed=42, stratify best_tier -- same as V2b/V3 training)")
    bce_weights = compute_bce_weights(pd.read_csv(LABELS_PATH).assign(
        image_path=lambda d: d["image_path"].str.replace("\\", "/", regex=False)))

    kwargs = {"num_workers": 2, "pin_memory": DEVICE.type == "cuda"}
    train_loader = DataLoader(CorrectnessGateV3Dataset(train_df, train_transform),
                              batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = DataLoader(CorrectnessGateV3Dataset(val_df, eval_transform),
                            batch_size=BATCH_SIZE, shuffle=False, **kwargs)
    test_loader = DataLoader(CorrectnessGateV3Dataset(test_df, eval_transform),
                             batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    if not args.report_only:
        for name in [c.strip() for c in args.cells.split(",")]:
            if CELLS[name]["ckpt"].exists():
                print(f"\n[{name}] checkpoint exists -- loading and evaluating only.")
                continue
            train_cell(name, CELLS[name], train_loader, val_loader, bce_weights)
    report(test_loader)
    print("\nDone.")


if __name__ == "__main__":
    main()
