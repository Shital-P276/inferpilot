# InferPilot — Full System Dossier for Comparison

> **Purpose:** Single consolidated reference for faculty review and for automated comparison (Claude) versus any other project. Covers how the system works, what was researched, what results were obtained, and how the codebase is organized. All values are pulled directly from files/logs on disk at `/mnt/d/Projects/inferpilot` — no assumptions.
>
> **Snapshot date:** 2026-08-30 · **Root:** `inferpilot/` · **Git repo:** yes · **Platform:** WSL (project on `/mnt/d`, venv at `~/venv_inferpilot` per `AGENTS.md`) · **Baseline:** `SIMULATED_HEAVY_DELAY_MS=0` (real Heavy, non-simulated) unless noted.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [How It Works — Architecture & Data Flow](#2-how-it-works--architecture--data-flow)
3. [Dataset & Label Generation](#3-dataset--label-generation)
4. [Tier Models — Fast / Balanced / Heavy](#4-tier-models--fast--balanced--heavy)
5. [Router Research — Cascade vs Single-Shot](#5-router-research--cascade-vs-single-shot)
6. [Correctness-Gate Lineage — V1, V2, V2b, V2c-costaware](#6-correctness-gate-lineage--v1-v2-v2b-v2c-costaware)
7. [Lambda Sweeps & Cost Models](#7-lambda-sweeps--cost-models)
8. [Evaluation Methodology — Sequential & Concurrent](#8-evaluation-methodology--sequential--concurrent)
9. [Results — Core (Sequential) & Concurrent (5/25/50 VUs)](#9-results--core-sequential--concurrent-52550-vus)
10. [Statistical Rigor — AUDC, Confident-Zone, McNemar](#10-statistical-rigor--audc-confident-zone-mcnemar)
11. [Routing Strategies — All 9 at the Gateway](#11-routing-strategies--all-9-at-the-gateway)
12. [Benchmarks — Per-Tier Latency / Memory / Params](#12-benchmarks--per-tier-latency--memory--params)
13. [Codebase Atlas — Directory & File Inventory](#13-codebase-atlas--directory--file-inventory)
14. [Infrastructure — Docker, Prometheus, Grafana, k6](#14-infrastructure--docker-prometheus-grafana-k6)
15. [Operational Notes — Venv, Paths, Reproducibility](#15-operational-notes--venv-paths-reproducibility)
16. [Comparison Rubric — Axes for Claude / Faculty](#16-comparison-rubric--axes-for-claude--faculty)
17. [Decision Matrix — Go / No-Go Signals](#17-decision-matrix--go--no-go-signals)
18. [Appendix A — Raw CSV Dumps (Full, No Truncation)](#18-appendix-a--raw-csv-dumps-full-no-truncation)
19. [Appendix B — Checkpoint & MLflow Inventory](#19-appendix-b--checkpoint--mlflow-inventory)

---

## 1. Executive Summary

**Problem:** Image classification has a cost/accuracy frontier. A single model is either cheap and inaccurate (`FastCNN` 88.12% @ 0.55ms) or accurate and expensive (`EfficientNet-B0` 98.5% @ 6.29ms, plus ~2.5s simulated cloud latency). The question is whether intelligent request routing across `fast`/`balanced`/`heavy` tiers can beat naive baselines on cost-adjusted utility and on accuracy under real concurrency.

**Answer from data:** Yes, but narrowly. The hand-written rule `fast_confidence < 0.7 → heavy else fast` already reaches **0.98 sequential / 0.978 concurrent** at **23-24% heavy** and 660/880 ms. The best learned system, `hybrid_gate` (rule outside `[0.60,0.80]`, learned `CorrectnessGateNetV2b` inside), pushes to **0.992 sequential / 0.9932 concurrent (burst 0.9975)** at **23% heavy** and 666/1060 ms, with `+0.4pp` over `always_heavy` at **4× lower latency** (666 ms vs 2588 ms sequential) and **77% fewer heavy calls**. The learned component itself contributes only `+0.13pp` over the dumb-escalation control `hybrid_dumb` (0.992 vs 0.9919 concurrent), saving ~11pp heavy.

The cascade ML router (`Balanced Random Forest` on 7 numerics + `fast_confidence` + `load_stage`) wins the sklearn comparison at `balanced_accuracy 0.5588` but is not the system winner — the gateway's `ml_router` strategy scores **0.984 sequential** (206 ms) and degrades to **0.9847 @ 1191 ms / p99 4824 ms** under burst. Single-shot (no `fast_confidence`) tops at `0.4937` and is not competitive except for throughput (`47 rps` vs `36 rps`).

**Cost of intelligence:** Gate inference is **1.92 ms p50 1.83 ms** (V2b) on top of a tier call. Throughput cost of the cascade (extra `fast` pre-call) is 50-100 ms (`single_shot_router` 138 ms < `ml_router` 206 ms sequential). Viable only when `heavy_cost / fast_cost > 5` and difficulty is skewed (here `fast 85%` optimal by utility).

**Statistical note:** With only **1 discordant pair** between `hybrid_gate` and `always_heavy` on 250 held-out images, McNemar `p=1.0` — the `+0.4pp` is not significant at N=250. Confident-zone validation shows the rule correctly identifies easy images (`0.9907` vs gate `0.9535` on confident 215/250).

---

## 2. How It Works — Architecture & Data Flow

```
data/raw (29291) ──make_splits.py──> data/{train 20496, val 4385, test 4410}
        │                                │
        │                         train_fast/balanced/heavy.py (MLflow)
        │                                ├─> checkpoints/*.pt (3 tiers)
        │                                └─> benchmark_models.py ─> router/model_benchmarks.json
        │                                                          + monitoring/results/combined_results.json (k6 latencies)
        │                                ┌────────────────────────┘
        └─────────── generate_utility_labels.py (4000 + 250 hold-out)
                     │  samples test, cached inference per tier, random latency/resource_norm
                     │  sweeps λ/μ, writes utility_labels.csv (best_tier = argmax correct - λ·latency_norm - μ·resource_norm)
                     │
                     ├─> train_router.py (cascade 9 variants) ─> router_best_model.pkl (Balanced RF)
                     ├─> train_router_singleshot.py (8 variants, no fast_confidence)
                     └─> train_correctness_gate*.py (V1/V2/V2b/V2c) ─> correctness_gate_*_best.pt
                                                        │
                          Docker image (cuda:12.4, Python 3.11)
                          copies serving/ + training/models/ + checkpoints/ + router/*.pkl
                                                        │
                                    docker compose up (4 services)
                          fast:8001 ─┐
                          balanced:8002 ─┼─> gateway:8000 /route?strategy=  (9 strategies)
                          heavy:8003 ─┘         │
                                                ├─> k6 load_test_*.js (5/25/50 VUs)
                                                ├─> run_core_experiment.py (sequential, 250 hold-out)
                                                ├─> run_concurrent_experiment.py (concurrent, scoring)
                                                └─> Prometheus 9090 + Grafana 3000 (live dashboard)
```

**Gateway request path (`serving/gateway_service.py:320`):**
`POST /route?strategy=` with `UploadFile` → `extract_request_features()` (6 OpenCV numerics: `width,height,file_size_kb,blur_score=Laplacian.var,brightness,gray.mean,edge_density=Canny`) → `_bump_concurrency()` → `estimate_load_stage()` (light <10, heavy <40, burst otherwise) → conditional `POST fast:8001/predict` for `NEEDS_FAST_CONFIDENCE = {ml_router, rule_based, hybrid_gate, hybrid_dumb} ∪ always_fast` → strategy-specific `route_*()` → `ROUTING_DECISIONS[tier,strategy].inc()` → dispatch to chosen tier (reuse cached `fast_result` if `fast`) via `httpx.AsyncClient` → optional `asyncio.sleep(SIMULATED_HEAVY_DELAY_MS/1000)` if `heavy` (post-response, accuracy untouched) → `GATEWAY_LATENCY[strategy].observe()` → JSON `{predicted_class, confidence, latency_ms, routed_tier, strategy, fast_confidence_used_for_routing, estimated_load_stage, gateway_latency_ms, timings_ms}`.

---

## 3. Dataset & Label Generation

**Source splits (`training/make_splits.py`, seed 42, per-class `70/15/15`):**

| Split | Path | Files | Notes |
|-------|------|-------|-------|
| raw | `data/raw/Fruit And Vegetable Diseases Dataset/` | **29291** | Kaggle source, gitignored |
| train | `data/train/` | **20496** | 70% |
| val | `data/val/` | **4385** | 15% |
| test | `data/test/` | **4410** | 15% (source for utility_labels + hold-out) |

**28 classes (identical in all splits):** `Apple__Healthy`, `Apple__Rotten`, `Banana__Healthy`, `Banana__Rotten`, `Bellpepper__Healthy`, `Bellpepper__Rotten`, `Carrot__Healthy`, `Carrot__Rotten`, `Cucumber__Healthy`, `Cucumber__Rotten`, `Grape__Healthy`, `Grape__Rotten`, `Guava__Healthy`, `Guava__Rotten`, `Jujube__Healthy`, `Jujube__Rotten`, `Mango__Healthy`, `Mango__Rotten`, `Orange__Healthy`, `Orange__Rotten`, `Pomegranate__Healthy`, `Pomegranate__Rotten`, `Potato__Healthy`, `Potato__Rotten`, `Strawberry__Healthy`, `Strawberry__Rotten`, `Tomato__Healthy`, `Tomato__Rotten`.

Per-class counts are imbalanced (e.g. `Apple__Rotten 2930` vs `Grape__Healthy 200`). See Appendix.

**Utility labels (`router/generate_utility_labels.py`, `CHOSEN_LAMBDA=0.7, CHOSEN_MU=0.3`, `SAMPLE_SIZE=4000, HOLD_OUT_SIZE=250`, `SEED=42`):**

* Samples `data/test` via `sample_all_image_paths()` → `random.shuffle` → `split_holdout_and_working(all,250,4000)` — first 250 → `held_out` (never touched), remainder grouped by class, `base_per_class=4000//28=142` per class + redistribution to exactly 4000.
* For each working image: PIL→`eval_transform` (Resize256→CenterCrop224→Normalize), OpenCV features, then cached `predict()` per tier → `correct∈{0,1}, confidence, latency_ms = random.choice(distribution[tier,stage])` where distribution is `monitoring/results/combined_results.json` `http_req_duration` tagged by `tier` + `load_stage`.
* `latency_norm = min(latency / p95_max_latency, 1.0)`, `resource_norm[tier] = 0.7·gpu_peak/max_gpu + 0.3·num_params/max_params` from `router/model_benchmarks.json`.
* For each `(λ,μ)` in `LAMBDA_GRID [0.3,0.7,1.2,2.0] × MU_GRID [0.1,0.3,0.5]`: `utility = correct - λ·latency_norm - μ·resource_norm` → `best_tier = argmax utility`. Writes `utility_sweep_summary.csv`. Final with `λ=0.7, μ=0.3`.

**Output files:**
* `router/utility_labels.csv` — 4001 lines (4000+header, 725KB), 22 cols: `image_path` (Windows `\`), `true_label`, `load_stage`, `width,height,file_size_kb,blur_score,brightness,edge_density`, `fast/balanced/heavy_{correct,confidence,latency_ms,utility}`, `best_tier`.
  Label distribution final: `fast 3415 (85.4%), balanced 363 (9.1%), heavy 222 (5.5%)`.
* `router/held_out_images.csv` — 251 lines (250+header, 19KB), `image_path,true_label` only, written before inference.
* `router/utility_sweep_summary.csv` — 12 rows, fast_pct ranges 75-88% (chosen 85.4%).
* `router/utility_lambda_sweep.csv` — strategy utility comparison by `compute_strategy_utility.py`.

**Transforms (`training/data_pipeline.py`, `IMG_SIZE=224, BATCH=32`):**
Train (deliberately hard): `ToImage → RandomResizedCrop 0.75-1.0 → RandomHorizontalFlip 0.5 → RandomRotation 20° → ColorJitter 0.35/0.35/0.3/0.05 → GaussianBlur p0.3 → ToDtype → RandomErasing p0.2 0.02-0.15 → Normalize(IMAGENET)`. Val/Test clean: `Resize256 → CenterCrop224 → ToDtype → Normalize`.

---

## 4. Tier Models — Fast / Balanced / Heavy

**Checkpoints (`training/checkpoints/*.pt`), MLflow `inferpilot-model-tiers`:**

| Tier | Model class (`training/models/*.py`) | Params | Val acc | Test acc | Checkpoint |
|------|--------------------------------------|--------|---------|----------|------------|
| **Fast** | `FastCNN` — 4×(Conv3×3+BN+ReLU+MaxPool)+AdaptiveAvgPool→Dropout0.3→Linear256→28 (~397k) | 397,536 | 0.8753 (epoch19) | **0.8812** | `fast_cnn_best.pt` (1.6M), MLflow `fast_cnn` 20ep AdamW 1e-3 Cosine |
| **Balanced** | `BalancedMobileNet` — `mobilenet_v3_large` ImageNet, replace `classifier[-1]`, freeze→unfreeze 3 blocks | 4,262,346 | 0.9818 (epoch14, phase2) | **0.9819** | `balanced_mobilenet_best.pt` (17M), 2-phase 5ep 1e-3 frozen + 10ep 1e-4 unfreeze |
| **Heavy** | `HeavyEfficientNet` — `efficientnet_b0` same head/freeze | 4,085,481 | 0.9815 (epoch13) | **0.985** | `heavy_efficientnet_best.pt` (16M), same 2-phase |

---

## 5. Router Research — Cascade vs Single-Shot

**Cascade (`router/train_router.py`, 339L, 9 variants, features 7 numeric + `load_stage`, `StandardScaler+OneHot`, 70/15/15 stratify `best_tier` seed 42):**

| # | Model | Hyperparams | `accuracy` | `balanced_accuracy` | `fast_f1` | `balanced_f1` | `heavy_f1` |
|---|-------|-------------|------------|---------------------|-----------|---------------|------------|
| **1** | **Balanced Random Forest** | `n_estimators=300, max_depth=12, sampling_strategy="all", replacement=True, bootstrap=False` | 0.7317 | **0.5588 ★** | 0.8562 | 0.3929 | 0.1869 |
| 2 | Logistic Regression | `class_weight=balanced, max_iter=1000` | 0.7450 | 0.5518 | 0.8660 | 0.3867 | 0.2000 |
| 3 | Logistic Regression + SMOTE | + `SMOTE(k_neighbors=5)` | 0.7333 | 0.5473 | 0.8574 | 0.3580 | 0.2095 |
| 4 | Gradient Boosting + SMOTE | `n_estimators=200, max_depth=4, lr=0.1` + SMOTE + `sample_weight balanced` | 0.7817 | 0.5419 | 0.8880 | 0.3916 | 0.2143 |
| 5 | Random Forest + SMOTE | `n_estimators=300, max_depth=12` + SMOTE | 0.7700 | 0.5130 | 0.8825 | 0.3649 | 0.1707 |
| 6 | Confidence Threshold (rule-based) | grid `low 0.5-0.95 step0.05, high low+0.02-1.0 step0.02`, balanced_accuracy train | 0.7950 | 0.5037 | 0.9047 | 0.2301 | 0.2889 |
| 7 | Random Forest | `n_estimators=300, max_depth=12, class_weight=balanced` | 0.7833 | 0.4790 | 0.8918 | 0.3597 | 0.1111 |
| 8 | Gradient Boosting | `n_estimators=200, max_depth=4, lr=0.1` | 0.7967 | 0.4720 | 0.8977 | 0.3415 | 0.1429 |
| 9 | LightGBM | `n_estimators=300, max_depth=8, lr=0.05, class_weight=balanced` | 0.8200 | 0.4447 | 0.9138 | 0.2963 | 0.1356 |

Saved `router/router_best_model.pkl` = **Balanced Random Forest** (best). Diagnostic `router_eval_utils.py:random_baseline_routing_acc` (20-trial Monte-Carlo matched proportions, `sigma = (acc - random)/std`, need `≥2σ` defensible, `≥3σ` strong).

**Single-shot Path A (`router/train_router_singleshot.py`, 318L, 8 variants, drops `fast_confidence`, 6 numerics only, `n_jobs=1` baked — fixes cascade `n_jobs=-1` 62.5-84.8% overhead):**

| Model | `balanced_accuracy` |
|-------|---------------------|
| **Gradient Boosting + SMOTE** | **0.4937** |
| Balanced Random Forest | 0.4607 |
| Random Forest + SMOTE | 0.4413 |
| LightGBM | 0.4333 |
| Logistic Regression + SMOTE | 0.4284 |
| Logistic Regression | 0.4198 |
| Gradient Boosting | 0.4100 |
| Random Forest | 0.3975 |

Cascade `0.5588` > Single-shot `0.4937` confirms `fast_confidence` carries signal, but question is whether 50-100 ms cascade tax outweighs the `0.06` balanced_accuracy gap — answered by live experiments (§9).

**Feature importance (saved `feature_importance.png`):** For RF/GB/LightGBM, `fast_confidence` dominates.

---

## 6. Correctness-Gate Lineage — V1, V2, V2b, V2c-costaware

Common: `IMG_SIZE 224, EPOCHS 25, BATCH 32, LR 1e-3, WD 1e-4, AdamW+CosineAnnealing`, seed 42 stratified 70/15/15 on `best_tier` (2800/600/600), `BCELoss` (except V2c), checkpoint by `avg AUROC` (except V2c by `val_balanced_routing_acc`), `TARGET_COLUMNS=[fast_correct,balanced_correct,heavy_correct]`, 3 independent sigmoids (not softmax — image can be correct for 0-3 tiers).

| Version | File | Architecture (`training/models/correctness_gate.py`) | Total | Trainable | Test AUROC fast / balanced / heavy / avg | Val AUROC avg | Latency isolated |
|---------|------|------------------------------------------------------|-------|-----------|------------------------------------------|---------------|-----------------|
| **V1** | `training/train_correctness_gate.py` → `correctness_gate_best.pt` (103K) | `CorrectnessGateNet` — from-scratch 3-conv `3→16→32→64 s2+BN+ReLU`, `AdaptiveAvgPool→Flatten→Drop0.2→Linear64→3→Sigmoid` | 24,230 | **24,003** | **0.6764 / 0.7692 / 0.6748 / 0.7068** (val 0.6956/0.8065/0.7845→0.7622 @ep21) | 0.7622 | **0.425 ms avg / 0.403 p50 / 0.654 p95** (`correctness_gate_latency_benchmark.json`, GPU peak 354MB) |
| **V2** | `train_correctness_gate_v2.py` → `correctness_gate_v2_best.pt` (130K) | `CorrectnessGateNetV2` — **Frozen MobileNetV3-Small `features[:5]` (24,224 frozen) + minimal `Linear40→3` (123 params)** — tests if pretrained features alone help | 25,655 | **123** | **0.5066 / 0.5566 / 0.5682 / 0.5438** (val 0.5734 @ep25) — **regression**, under-capacity | 0.5734 | **1.62 / 1.288 / 3.286 ms** (`v2_latency_benchmark.json`, GPU 10.73MB) |
| **V2b** (prod) | `train_correctness_gate_v2b.py` → `correctness_gate_v2b_best.pt` (232K) loaded by gateway `CORRECTNESS_GATE_DEVICE=cpu` | `CorrectnessGateNetV2b` — **Same frozen `Small[:5]` + MLP `40→180→BN→ReLU→Drop0.3 →180→90→BN→ReLU→Drop0.3 →90→3→Sigmoid` (~24k)** — isolates head-capacity confound | **50,557** (mlflow 48,707) | **24,483** | **0.7339 / 0.7724 / 0.7653 / 0.7572** (val 0.7147/0.7483/0.6344→0.6991 @ep5) | 0.6991 | **1.924 / 1.833 / 3.063 ms** (`v2b_latency_benchmark.json`, GPU 0.0) |
| **V2c-costaware** | `train_correctness_gate_v2c_costaware.py` → `correctness_gate_v2c_costaware_best.pt` (233K) | `CorrectnessGateNetV2c_costaware` — **Same backbone/head as V2b but NO Sigmoid (raw logits)**, trained with **cost-aware utility loss**: `q=softmax((targets-λ·costs)/T_target=0.3)`, `log_p=log_softmax((raw-λ·costs)/T_pred=1.0)`, `loss=-(q·log_p).sum.mean()` | 50,557 | 24,483 | **NOT AUROC-comparable** (logits) — metric is `induced_accuracy = mean(targets[i, argmax(logits-λ·cost)])` | N/A | ~1.9 ms (same arch) |

**V2c selection:** `LAMBDA_TRAIN` auto-picked as best non-degenerate `>95%` filter of `correctness_gate_v2b_lambda_sweep_finegrid.csv` → **0.17** (`balanced_routing_acc 0.9686`). Checkpoint by `val_balanced_routing_acc` tiebreak lowest cost. Val `0.9359` / `induced_accuracy 0.9283` @ `T_target 0.3`, Test `balanced_routing_acc 0.9599` / `induced_accuracy 0.935` / `induced_expected_cost 0.6222`.

---

## 7. Lambda Sweeps & Cost Models

All sweeps use utility `pick tier = argmax P(correct) - λ·cost`. Two cost models:

* **Resource cost** (`resource_norm = 0.7·gpu_peak/max_gpu + 0.3·num_params/max_params`): `fast≈0.645, balanced≈0.947, heavy=1.0`. Canonical.
* **Latency cost** (`latency_norm = avg_latency / max avg_latency`): `fast 0.087 (0.549ms), balanced 0.769 (4.842ms), heavy 1.0 (6.291ms)`. Used for finegrid & defensible analysis.

**Metric:** `balanced_routing_acc = macro-avg over routed tiers of mean(correct) within tier` (empty tiers skipped, `training/train_correctness_gate.py:139`). Paired with `random_baseline_mean/std` (20 Monte-Carlo matched proportions) → `sigma = (acc - random)/std` (≥2 significant, ≥3 strong). **Degenerate filter:** `max(pct_fast, pct_balanced, pct_heavy) > 95%` → collapsed single-tier policy, not reported as win.

**Key CSVs (`training/checkpoints/`):**

* **`correctness_gate_lambda_sweep.csv` (V1 coarse 0.05-0.95):** Rapid collapse: `0.05 0.67/30.5/68.83 acc 0.9917 random 0.9443`, `0.15 23.67/38.17/38.17 acc 0.9734`, `0.25 58.67/23.83/17.5 acc 0.9619`, `0.35 89.17/5/5.83 acc 0.941`, `0.40 94.5/2.83/2.67 acc 0.9612`, `≥0.50 100/0/0 acc 0.8717`. All `≥0.40` degenerate.
* **`correctness_gate_lambda_sweep_finegrid.csv` (V1 0.15-0.20 step 0.01 latencycost):** Heavy always 0 (cheaper). Best **defensible `σ≥3` at `0.15 acc 0.95 σ4.59 pct 50/50/0`**.
* **`correctness_gate_lambda_sweep_calibrated.csv` (Platt-calibrated V1, Gates A/B raw AUROC within 5e-3, calibrated within 2e-3):** Shifts to balanced: `0.05 0/99.67/0.33 acc 0.9916`, `0.10-0.15 0/100/0 acc 0.9833`, `0.20 34.67/65.33/0 acc 0.9536`, `≥0.35 100/0/0`.
* **`correctness_gate_lambda_sweep_latencycost.csv` (V1 latencycost coarse):** `0.05 5/81.67/13.33 acc 0.9959 σ2.25 pct_expensive95`, `0.15 50/50/0 acc 0.95 σ4.56`, `0.20 74/26/0 acc 0.9323`. Optimum `0.15`.
* **`correctness_gate_v2b_lambda_sweep_latencycost.csv` (V2b latencycost coarse 0.05-0.95):** `0.05 7.83/58.33/33.83 acc 0.9818 σ2.26`, `0.25 71.67/25.83/2.5 acc 0.966 σ2.04` peak, `≥0.65 100/0/0`.
* **`correctness_gate_v2b_lambda_sweep_finegrid.csv` (V2b 0.10-0.20 step 0.005) ★ Most used:** `sigma≥3` cliff `0.17`. `0.12 0.9648 σ2.96`, `0.17 47.5/45.33/7.17 acc **0.9686 σ4.23** pct_expensive52.5 ★ best defensible`, `0.18 50/43.33/6.67 acc 0.9675 σ3.36` also defensible. Used as `LAMBDA_TRAIN=0.17` in V2c.

**Latency vs Resource:** `sweep_latency_cost.py` + `sweep_latency_cost_v2b.py` are pure analysis (no model mutation) re-running `sweep_routing` at latency ratios.

---

## 8. Evaluation Methodology — Sequential & Concurrent

**Shared held-out:** `router/held_out_images.csv` 250 images (carved before utility labeling, Windows `\` paths, Linux normalized via `.replace("\\","/")`).

**Sequential — `router/run_core_experiment.py` (140L):** Loops 9 strategies over held-out via `httpx.Client → gateway:8000/route?strategy=` sequential (one-at-a-time, no concurrency), records `routed_tier, predicted_class, correct=(pred==true), gateway_latency_ms`. Outputs `core_experiment_raw_results.csv` (2250 rows) + `core_experiment_summary.csv` (per-strategy `accuracy, avg/p95 latency, pct_fast/balanced/heavy`). Env overrides `CORE_EXPERIMENT_RESULTS_PATH/SUMMARY_PATH` for simulated-heavy variants.

**Concurrent — `router/run_concurrent_experiment.py` (341L) + `monitoring/load_test_gateway.js` k6 pattern:** asyncio + `httpx.AsyncClient`, **3 phases per strategy:** `light 5 VUs×30s → heavy 25 VUs×30s → burst 50 VUs×30s` (matches `load_test_gateway.js:100-127` scenarios `light_load constant-vus 5/30s, heavy 25/30s, burst ramping-vus 5→50`). Each worker continuously picks random held-out image for phase duration (not fixed count). For every request records same per-image accuracy+latency. After phases computes per-phase and overall `accuracy, avg/p50/p95/p99 latency, throughput, tier distribution`. Outputs `concurrent_experiment_raw_results.csv` + `concurrent_experiment_summary.csv` (one row per `(strategy, phase)`). Pre-run check warns if `SIMULATED_HEAVY_DELAY_MS≠0`.

**Observability:** Prometheus scrapes `fast:8001, balanced:8002, heavy:8003, gateway:8000` every 5s; Grafana `inferpilot_dashboard.json` shows `Request Rate/tier, Inference Latency p95, Confidence, Predictions pie, Predictions by Class 5m bargauge, HTTP p95`.

---

## 9. Results — Core (Sequential) & Concurrent (5/25/50 VUs)

**Core sequential (250 req per strategy, `SIMULATED_HEAVY_DELAY_MS=0`, post Fast HTTP fix):**
`router/core_experiment_summary.csv`:

| Strategy | Accuracy | Avg latency ms | p95 ms | pct_fast | pct_balanced | pct_heavy |
|----------|----------|----------------|--------|----------|--------------|-----------|
| **hybrid_gate** | **0.992** | 666.59 | 2656.09 | 72.0 | 4.8 | 23.2 |
| **hybrid_dumb** | **0.992** | 887.41 | 2718.78 | 68.0 | 0.0 | 32.0 |
| always_heavy | 0.988 | 2588.53 | 2690.96 | 0.0 | 0.0 | 100.0 |
| **ml_router** | 0.984 | **206.77** | **549.29** | 76.0 | 19.6 | 4.4 |
| rule_based | 0.98 | 660.91 | 2663.95 | 76.8 | 0.0 | 23.2 |
| correctness_gate_router | 0.96 | 1004.6 | 2600.14 | 39.6 | 23.6 | 36.8 |
| round_robin | 0.948 | 903.89 | 2566.79 | 33.6 | 33.2 | 33.2 |
| single_shot_router | 0.936 | **138.14** | **279.75** | 85.6 | 11.6 | 2.8 |
| always_fast | 0.892 | **62.25** | **195.97** | 100.0 | 0.0 | 0.0 |

`strategy_utility_comparison.csv` (`utility = accuracy -0.7·latency_norm -0.3·resource_norm`, λ=0.7): winner `rule_based 0.2893` > `round_robin 0.2766` > `always_heavy 0.2442`; `utility_lambda_sweep.csv` shows `rule_based` wins λ 0.05-0.45, `always_fast` wins λ≥0.5.

**Concurrent (`router/concurrent_experiment_summary.csv`, 9 strategies × 4 rows inc. overall, real heavy):**

| Strategy | Phase | N req | Acc | Avg ms | p95 ms | Throughput rps | pct_fast/balanced/heavy |
|----------|-------|-------|-----|--------|--------|----------------|-------------------------|
| **always_fast** | light 559 | 0.9106 | 192 | 413 | 18.63 | 100/0/0 |
| | heavy 413 | 0.8959 | 855 | 1365 | 13.77 |  |
| | burst 213 | 0.8967 | 1772 | 2653 | 7.10 |  |
| | overall 1185 | **0.903** | 707 | 2156 | 39.5 |  |
| **always_heavy** | light 530 | 0.9887 | 209 | 431 | 17.67 | 0/0/100 |
| | heavy 411 | 0.9927 | 908 | 1382 | 13.70 |  |
| | burst 205 | 0.9902 | 1788 | 2500 | 6.83 |  |
| | overall 1146 | **0.9904** | 742 | 2121 | 38.2 |  |
| **round_robin** | overall | 0.9468 | 738 | 2130 | 38.83 | 33.4/33.3/33.3 |
| **rule_based** | light 462 | 0.9848 | 238 | 511 | 15.40 | 77.1/0/22.9 |
| | heavy 356 | 0.9691 | 1050 | 2104 | 11.87 | 72.5/0/27.5 |
| | burst 271 | 0.9779 | 1752 | 3590 | 9.03 | 77.5/0/22.5 |
| | overall | **0.978** | 880 | 2441 | 36.3 | 75.7/0/24.3 |
| **ml_router** | overall | **0.9847** | **1191** | 3886 | **28.4** | 68/16.3/15.7 (burst p99 4824, **degrades**) |
| **single_shot_router** | overall | 0.9433 | 886 | 2266 | **47.03** | 75.8/13.2/11 (highest throughput, no Fast pre-call) |
| **correctness_gate_router** | overall | 0.9583 | 1107 | 2876 | 40.8 | 39.1/22.4/38.6 |
| **hybrid_gate ★** | light 465 | **0.9935** | 251 | 600 | 15.50 | 70.3/5.6/24.1 |
| | heavy 448 | 0.9888 | 1073 | 2086 | 14.93 | 71.9/4.7/23.4 |
| | burst 406 | **0.9975** | 1971 | 3708 | 13.53 | 74.4/4.2/21.4 |
| | overall | **0.9932** | 1060 | 3069 | **43.97** | **72.1/4.9/23.0** |
| **hybrid_dumb** | overall | **0.9919** | 1039 | 3117 | **45.03** | 66/0/34 (control proves architecture > gate quality) |

*Throughput is completed requests / 30s per phase; `hybrid_gate` burst 0.9975 is the only `>0.99` under 50 VUs.*

---

## 10. Statistical Rigor — AUDC, Confident-Zone, McNemar

**AUDC (`monitoring/reports/audc_comparison.csv`, `training/audc_analysis.py` trapezoidal):**
```
model,audc_raw,audc_normalized,best_naive_acc,best_naive_lambda,best_defensible_acc,best_defensible_lambda
CorrectnessGateNet_V1,0.088036,7.3363,0.9917 @0.05,0.95 @0.15 σ4.59
CorrectnessGateNetV2b,0.083681,6.9734 (better),0.9818 @0.05,0.9686 @0.17 σ4.23
rule_based, single point 0.972 / ml_router 0.972 (accuracy-conflated)
```
AUDC normalized to `1-always_heavy`. V2b slightly dominates V1 across curve despite V1 naive peak higher.

**Confident-zone validation (`monitoring/reports/confident_zone_validation.csv`, `training/confident_zone_validation.py`, band `[0.60,0.80]`, 250 held-out):**
`confident_zone 215, uncertain 35`. Confident-2x2 (`rule_based` vs `gate`):

|  | Gate correct | Gate wrong |
|--|--------------|------------|
| **RB correct** | **204** | **9** |
| **RB wrong** | **1** | **1** |

Only gate win outside band: `data/test/Tomato__Healthy/freshTomato (142).jpg` conf 0.3811 (rule `Jujube__Healthy` wrong / gate `Tomato__Healthy` right).
Accuracies: **confident zone `rule_based 0.9907` vs `gate 0.9535` diff -0.0372**, full 250 `0.98 vs 0.96` — validates `0.7` threshold correctly identifies easy images; hybrid's design (gate only in uncertain band) is justified.

**Confidence-band analysis (`monitoring/reports/confidence_band_analysis.csv`):**
Histogram bins 0.05, threshold 0.7: captures `±0.05 6.4%, ±0.10 18.4%, ±0.15 26.8%, ±0.20 38.4%`. **Recommendation `±0.10 → [0.60,0.80] 18.4%`** (15-30% target). Distribution heavy `0.95-1.0 38.8%`, `0.90-0.95 11.6%`.

**McNemar (`training/mcnemar_hybrid_vs_heavy.py` on `core_experiment_raw_results.csv` 250 paired):**
```
                    always_heavy=1  always_heavy=0
hybrid_gate=1                247              1
hybrid_gate=0                  0              2
Discordant: 1 (hybrid_only 1, heavy_only 0), exact binomial (n<25)
statistic 0.0, p-value 1.000000
hybrid_gate 0.992 (248/250) vs always_heavy 0.988 (247/250) => NOT significant (α0.05)
```

---

## 11. Routing Strategies — All 9 at the Gateway

**Source:** `serving/gateway_service.py:98 VALID_STRATEGIES` + `SERVE` impl. `SERVICE_URLS fast:8001, balanced:8002, heavy:8003`, `SIMULATED_HEAVY_DELAY_MS 2500` (disabled in real runs), `CORRECTNESS_GATE_DEVICE cpu` (default), `RULE_BASED_CONFIDENCE_THRESHOLD 0.7`.

| Strategy | Kind | Needs `fast_confidence`? | Hops | Mechanism |
|----------|------|--------------------------|------|-----------|
| `ml_router` | Cascade ML | **Yes** (1 extra Fast call) | 2 if ≠fast | `BalancedRandomForest` on `[6 numeric + fast_confidence + one-hot load_stage]` → `label_encoder.inverse_transform(predict)` |
| `single_shot_router` | Single-shot ML (Path A) | No | **1** | `GradientBoosting+SMOTE` on `[6 numeric + load_stage]` only, no Fast call |
| `correctness_gate_router` | Single-shot CNN (Path B) | No | 1 CNN +1 tier | `CorrectnessGateNetV2b` frozen MobileNetV3-Small[:5] +24k head predicts `P(correct)` per tier from raw pixels, picks `argmax(P_correct -0.20·cost)` with costs `fast 0.480, balanced 0.947, heavy 0.986` |
| `rule_based` | Cascade rule | Yes | 2 if escalated | `if fast_confidence <0.7: heavy else fast` |
| `hybrid_gate` | Hybrid | Yes | 1-2 + optional CNN | Outside `[0.60,0.80]`: rule_based; inside: delegate to `correctness_gate_router` |
| `hybrid_dumb` | Control | Yes | Same | Same band as hybrid_gate, inside → deterministic `heavy` (no model) — isolates intelligence vs dumb escalation |
| `always_fast` | Baseline | Yes (needs answer, reuses) | 1 | Always `fast`, reuses Fast result |
| `always_heavy` | Baseline | No | 1 + sleep | Always `heavy` |
| `round_robin` | Baseline | No | 1 | Thread-safe counter `["fast","balanced","heavy"]` cyclic |

`NEEDS_FAST_CONFIDENCE` drives `fast:8001/predict` pre-call only for cascade strategies + `always_fast` (to get answer). Single-shot paths deliberately avoid it (one-hop architecture — the project question).

---

## 12. Benchmarks — Per-Tier Latency / Memory / Params

**Source:** `router/model_benchmarks.json` (from `benchmark_models.py`, isolated, `device cuda`) **and** gate latency JSONs:

**Tier inference (isolated, not gateway):**
| Tier | Model | `avg_latency_ms` | `p50` | `p95` | `gpu_model_footprint_mb` | `gpu_peak_inference_mb` | `checkpoint_size_mb` | `num_params` |
|------|-------|-----------------|-------|-------|--------------------------|-------------------------|----------------------|--------------|
| fast | FastCNN | **0.549** | 0.517 | 0.815 | 1.52 | 22.47 | 1.53 | 396572 |
| balanced | MobileNetV3-Large | **4.842** | 4.729 | 5.57 | 24.71 | 32.18 | 16.37 | 4237900 |
| heavy | EfficientNet-B0 | **6.291** | 6.084 | 8.429 | 23.79 | 34.81 | 15.72 | 4043416 |

`resource_norm = 0.7·gpu_peak/max + 0.3·num_params/max` → `fast≈0.645, balanced≈0.947, heavy=1.0`.

**Gate isolated (single image, `benchmark_correctness_gate*.py`):**
* V1: `avg 0.425, p50 0.403, p95 0.654, gpu_peak 354MB, params 24003`
* V2: `avg 1.62, p50 1.288, p95 3.286, gpu_peak 10.73`
* V2b: `avg 1.924, p50 1.833, p95 3.063` (production CPU ~1.86ms; reported `14.0 rps CPU vs 13.4 rps GPU` under load — CPU frees GPU for tiers)

---

## 13. Codebase Atlas — Directory & File Inventory

**Root `/mnt/d/Projects/inferpilot`:** `AGENTS.md` (venv `~/venv_inferpilot`, no `venv/` on `/mnt/d`, backslash `.replace("\\","/")` fix), `opencode.json` (lsp:true), `.gitignore` (ignores `venv/, data/raw|train|val|test, checkpoints/*.pt, mlruns/, results/*.json, core_experiment_raw_results.csv`), `Dockerfile` (cuda 12.4 runtime ubuntu22.04, Python 3.11, copies `serving/ + training/models/ + checkpoints/ + training/__init__.py + router/*.pkl`), `docker-compose.yml` (6 services `fast 8001, balanced 8002, heavy 8003, gateway 8000, prometheus 9090, grafana 3000` + `inferpilot-net` bridge, GPU reservations), `requirements.txt` (106 pinned, `pywin32==312, torch 2.13.0, mlflow 3.14`), `requirements_wsl.txt` (minus pywin32 + statsmodels), `requirements_docker.txt` (14 lines, `torch 2.6.0+cu124, fastapi 0.115, opencv-headless`).

**`serving/`** — 4 FastAPI services: `fast_service.py:8001`, `balanced_service.py:8002`, `heavy_service.py:8003` (each load `*_best.pt`, eval `Resize256→CenterCrop224→Normalize`, metrics `inference_latency_ms, prediction_confidence, predictions_total[tier,predicted_class]`), `gateway_service.py:467` (all routing, Prometheus hist `gateway_latency_ms[strategy]` buckets 10-30000 ms, counter `routing_decisions_total[tier,strategy]`, concurr lock `estimate_load_stage`).

**`router/`** — `generate_utility_labels.py` (372L, see §3) → `utility_labels.csv 4001 lines`, `held_out_images.csv 251`, `model_benchmarks.json`; `train_router.py` (339L, 9 variants) → `router_comparison.csv, router_best_model.pkl` (Balanced RF), `confusion_matrices.png, feature_importance.png`; `train_router_singleshot.py` (318L, 8 variants, `n_jobs=1`) → `router_singleshot_comparison.csv, router_singleshot_model.pkl`; `router_eval_utils.py` (Monte-Carlo `random_baseline_routing_acc`, `sigma` warning); `compute_strategy_utility.py` (`utility=accuracy -0.7·latency_norm -0.3·resource_norm`); `utility_lambda_sweep.py` + `utility_sweep_summary.csv`; `pareto_analysis.py + pareto_frontier.png`; `run_core_experiment.py` (140L, `httpx.Client`, sequential 250) + `run_concurrent_experiment.py` (341L, asyncio `5/25/50 VUs ×30s` with scoring) → `*_summary.csv` + `*_raw_results.csv`; `experiment_hybrid_router.py`, `benchmark_gateway_cpu_cost.py`, `profile_router_predict.py`, `fix_router_n_jobs.py`; plots `confusion_matrices.png`, `feature_importance.png`, `singleshot_*`.

**`training/`** — `data_pipeline.py` (hard train aug: `RandomResizedCrop 0.85-1.0, Flip0.5, Rotation20°, ColorJitter 0.35, GaussianBlur p0.3, RandomErasing p0.2`; eval clean), `make_splits.py` (stratified per-class `70/15/15`), `models/fast_cnn.py` (~396K), `models/balanced_mobilenet.py` (MobileNetV3-Large), `models/heavy_efficientnet.py` (EfficientNet-B0), `models/gate_net.py` (24k 3-way), `models/confidence_gate.py` (regression), `models/correctness_gate.py` (4 variants V1/V2/V2b/V2c). Trainers: `train_fast.py` 20ep, `train_balanced.py` 2-phase `5+10`, `train_heavy.py` 2-phase, `train_gate.py` weighted CE, `train_confidence_gate.py` 30ep weighted MSE, `train_correctness_gate.py` V1 BCELoss 25ep `avg AUROC`, `train_correctness_gate_v2.py` frozen 123-params, `train_correctness_gate_v2b.py` 24k head, `train_correctness_gate_v2c_costaware.py` `cost_aware_utility_loss` (`T_target0.3, T_pred1.0`, auto `λ=0.17`). Tools: `sweep_latency_cost.py`, `sweep_latency_cost_v2b.py`, `calibrate_correctness_gate.py`, `audc_analysis.py`, `confidence_band_analysis.py`, `confident_zone_validation.py`, `diagnose_v2c_costaware.py`, `mcnemar_hybrid_vs_heavy.py`.

**`training/checkpoints/`** — 8× `*_best.pt` (fast 1.6M, balanced 17M, heavy 16M, gate 103K, conf 103K, V1 103K, V2 130K, V2b 232K, V2c 233K) + 8× `*_curves.png` + 3× `*_latency_benchmark.json` + 5× `*_lambda_sweep*.csv` (+ `v2b_finegrid` 22 rows, `v2_finegrid`) + `*_latencycost.csv`.

**`monitoring/`** — `prometheus.yml` (scrape 5s), `grafana/provisioning/` (datasource `prometheus:9090`, dashboard `inferpilot_dashboard.json` 6 panels Refresh 5s: Request Rate/tier, Latency p95/p50, Confidence, Predictions pie 5m, Predictions by Class bargauge, HTTP p95), `load_test.js` (constant-vus `light 5/30s, heavy 25/30s, ramp burst 5→50`), `load_test_combined.js`, `load_test_strategies.js`, `load_test_random_images.js`, `load_test_simulated_heavy.js` (`SIMULATED_HEAVY_DELAY_MS 2500`, 4 strategies), `sample_images/test_apple.png`, `pull_prometheus_data.py` (`/api/v1/query_range` → `prometheus_pulls/*.csv` + `summary_comparison.csv`), `prometheus_pulls/` 20+ CSVs per strategy/tier, `results/` k6 JSON dumps (`combined_results.json` with `tier+load_stage` tags), `reports/` (`audc_comparison.csv`, `confidence_band_analysis.csv`, `confident_zone_validation.csv`, `deferral_curves.png`).

**`reports/gradcam/`** — 18 PNGs `fast/balanced/heavy_{class}*.png`.

---

## 14. Infrastructure — Docker, Prometheus, Grafana, k6

* **Docker:** One image `nvidia/cuda:12.4.0-runtime-ubuntu22.04` + Python 3.11 + `libjpeg/libglib`. `pip install requirements_docker.txt` (Cache mount). Copies only `serving/, training/models/, training/checkpoints/, training/__init__.py, router/*.pkl`. `EXPOSE 8000-8003`. `CMD` overridden per service `uvicorn --host 0.0.0.0 --port …`. `docker-compose.yml` GPU `nvidia` reservation per app service, `gateway depends_on` 3 tiers, env `SIMULATED_HEAVY_DELAY_MS`, volumes `prometheus-data`, ports `8001-8003, 8000, 9090, 3000`.
* **Prometheus:** `monitoring/prometheus.yml` scrape `fast:8001, balanced:8002, heavy:8003, gateway:8000` interval 5s, labels `tier`.
* **Grafana:** `admin/admin`, datasource `prometheus:9090`, provisioning `dashboard.yml`, dashboard `inferpilot_dashboard.json` (6 panels).
* **k6:** `monitoring/load_test*.js` share `constant-vus/ramping-vus` pattern above, tags `load_stage`. `load_test_random_images.js` samples `router/held_out_images.csv` at init (`open()` text, shuffle, `open(relativePath,'b')` array, VU picks random) to exercise per-image routing diversity.

---

## 15. Operational Notes — Venv, Paths, Reproducibility

* **Venv:** Use `~/venv_inferpilot` (native Linux `/home/shital/venv_inferpilot`, not project `venv/` on `/mnt/d` — WSL hang risk). Works with `requirements_wsl.txt` (`pywin32` removed). Project `venv/` is Windows-native, do not modify. `AGENTS.md` enforces.
* **Paths:** `router/utility_labels.csv` + `held_out_images.csv` use Windows `\` (`data\test\...`). Must `.replace("\\","/")` on Linux — fixed in `CorrectnessGateDataset` (do not re-introduce). `BASE_DIR` is `Path(__file__).resolve().parent.parent`.
* **Seeds:** `42` everywhere (`make_splits`, `generate_utility_labels`, `train_router`, `train_correctness_gate`). Stratify on `best_tier` (or confidence bucket for ConfidenceGate).
* **Env overrides:** `CORE_EXPERIMENT_RESULTS_PATH/SUMMARY_PATH`, `CONCURRENT_EXPERIMENT_RAW_PATH/SUMMARY_PATH`, `SIMULATED_HEAVY_DELAY_MS` (default `2500`), `CORRECTNESS_GATE_DEVICE` (`cpu` default, `cuda` optional, CPU freed GPU improved `13.4→14.0 rps`).
* **Run commands (Windows):**
  ```
  docker compose up --build -d
  # sequential: python router/run_core_experiment.py  (or set SIMULATED_HEAVY_DELAY_MS=0)
  SIMULATED_HEAVY_DELAY_MS=0 python router/run_concurrent_experiment.py  # 9 strategies ×3 phases×30s ≈ 94s/strategy ≈ 14-15 min total for 9
  k6 run monitoring/load_test_strategies.js -e STRATEGY=rule_based
  ```

---

## 16. Comparison Rubric — Axes for Claude / Faculty

Use this table to score InferPilot vs any other project (fill `Other` column). Axes are the exact metrics InferPilot logs, so comparison is apples-to-apples.

| Axis | InferPilot value (source) | How measured | Weight for faculty | Claude prompt hint |
|------|---------------------------|--------------|--------------------|-------------------|
| **Accuracy (sequential)** | `hybrid_gate 0.992` ties `hybrid_dumb`, `always_heavy 0.988`, `ml_router 0.984` (`core_experiment_summary.csv`) | 250 held-out, one-at-a-time | Primary | `Compare top accuracy at same hold-out N.` |
| **Accuracy (concurrent)** | `hybrid_gate 0.9932 overall, burst 0.9975` vs `always_heavy 0.9904` (`concurrent_experiment_summary.csv`) | 5/25/50 VUs ×30s, per-request scoring | Primary | `Burst accuracy reveals routing under load.` |
| **Latency p95 (seq)** | `ml_router 549 ms` best learned, `single_shot 279 ms`, `always_fast 195 ms` (`core_summary`) | `gateway_latency_ms` | Cost | `p95 not avg — tail matters.` |
| **Latency p95 (conc burst)** | `ml_router 4746 ms` worst, `hybrid_gate 3708 ms`, `always_fast 2653 ms` (`concurrent_summary`) | `p95_latency_ms` per phase | Heavy | `Does learned routing inflate tail under burst?` |
| **Throughput** | `single_shot 47 rps`, `hybrid_dumb 45`, `hybrid_gate 43.97` vs `ml_router 28.4` (`concurrent overall`) | `completed / 30s` | Scale | `One-hop vs two-hop cost.` |
| **Heavy use** | `hybrid_gate 23.0%`, `rule_based 24.3%`, `always_heavy 100%` (`concurrent overall pct_heavy`) | `routed_tier` | OPEX | `Cost proxy for GPU/cloud.` |
| **Model params** | `fast 396k, balanced 4.24M, heavy 4.04M, gate V2b 24k` (`model_benchmarks.json`) | state_dict count | Complexity | `Smaller is cheaper to serve.` |
| **GPU mem** | `fast 22.47 MB peak, balanced 32.18, heavy 34.81` | `gpu_peak_inference_mb` | Infra | `Fits on edge?` |
| **Training cost** | `Gate V1 46s/epoch, V2b 14s/epoch, Fast 123s/epoch` (MLflow `epoch_time_sec`) | MLflow | Effort | `Epoch × epochs.` |
| **Router balanced_accuracy** | `Balanced RF 0.5588 > single-shot 0.4937` (`router_comparison.csv`) | sklearn `balanced_accuracy` macro | Learning | `Is cascade worth pre-call?` |
| **Gate AUROC** | `V1 0.7068, V2 0.5438, V2b 0.7572` test avg | `roc_auc_score` per tier | Signal | `Per-tier discriminability.` |
| **Defensible λ** | `V1 0.15 σ4.59 50/50/0 acc 0.95, V2b 0.17 σ4.23 47.5/45.3/7.1 acc 0.9686` | `sigma ≥3` non-degenerate `>95%` filter | Robustness | `Degenerate 100/0/0 not defensible.` |
| **AUDC** | `V1 0.088, V2b 0.083 (better)` normalized 7.33→6.97 | trapezoidal `1-acc` vs `λ` | Curve | `Overall curve, not single λ.` |
| **Statistical power** | McNemar `p=1.0` (discordant 1/250) | exact binomial `<25` | Rigor | `N=250 too small for 0.992 vs 0.988.` |
| **Confident-zone** | RB `0.9907` vs gate `0.9535` diff `-0.0372` on 215 confident | `confident_zone_validation.csv` | Justification | `Rule dominates outside [0.60,0.80].` |
| **Debuggability** | 4 FastAPI + `/metrics` + Grafana + `timings_ms` dict | `gateway_service.py:462 timings_ms` | Ops | `Can you trace per-request costs?` |
| **Reproducibility** | Seed 42, stratified, Windows `\` fix doc'd, `AGENTS.md` | `make_splits.py, CorrectnessGateDataset` | Faculty | `Rerunnable from raw?` |

**How Claude should use this:** For each axis, score `Other` as `better / parity / worse` citing its own logged files (not claims). Require `Other` to provide same artifacts: held-out size, `N_requests`, `p95`, `pct_heavy`, `sigma`, `p-value`, `latency_benchmark.json` — otherwise mark `insufficient evidence`.

---

## 17. Decision Matrix — Go / No-Go Signals

* **Go (continue hybrid_gate):** If OPEX matters and `hybrid_gate 0.992-0.993` ≥ `always_heavy 0.988` at `1/4` latency and `1/4` heavy holds under burst (it does: burst 0.9975 > 0.9902), and `+0.13pp` over `hybrid_dumb` plus 11pp heavy saving justifies 1.9 ms gate cost — current data supports **go with caveat `p=1.0` not significant**.
* **No-go (ship rule_based):** If p99 matters, `rule_based` 0.978 @ `p99` conc 3655 ms < `hybrid_gate` 3894 ms and simpler (no CNN) — `hybrid_dumb` 0.992 shows even dumb escalation suffices; gate complexity not yet justified for faculty.
* **Pivot (fix routers):** `ml_router` concurrent `1191 ms avg / p99 4824 ms` degraded vs sequential `206 ms` — cascade contention on `balanced` tier under load (see `ml_router` burst `53.8/20.3/25.8` heavy spill). Fix by `n_jobs=1` already done for single-shot; consider `CORRECTNESS_GATE_DEVICE=cpu` + `balanced` GPU pinning, or single-shot only.

---

## 18. Appendix A — Raw CSV Dumps (Full, No Truncation)

### `router/router_comparison.csv`
```
model_name,accuracy,balanced_accuracy,fast_f1,balanced_f1,heavy_f1
Balanced Random Forest,0.7317,0.5588,0.8562,0.3929,0.1869
Logistic Regression,0.745,0.5518,0.866,0.3867,0.2
Logistic Regression + SMOTE,0.7333,0.5473,0.8574,0.358,0.2095
Gradient Boosting + SMOTE,0.7817,0.5419,0.888,0.3916,0.2143
Random Forest + SMOTE,0.77,0.513,0.8825,0.3649,0.1707
Confidence Threshold (rule-based),0.795,0.5037,0.9047,0.2301,0.2889
Random Forest,0.7833,0.479,0.8918,0.3597,0.1111
Gradient Boosting,0.7967,0.472,0.8977,0.3415,0.1429
LightGBM,0.82,0.4447,0.9138,0.2963,0.1356
```

### `router/core_experiment_summary.csv`
```
strategy,n_requests,accuracy,avg_latency_ms,p95_latency_ms,pct_fast,pct_balanced,pct_heavy
always_fast,250,0.892,62.25,195.97,100.0,0.0,0.0
always_heavy,250,0.988,2588.53,2690.96,0.0,0.0,100.0
round_robin,250,0.948,903.89,2566.79,33.6,33.2,33.2
rule_based,250,0.98,660.91,2663.95,76.8,0.0,23.2
ml_router,250,0.984,206.77,549.29,76.0,19.6,4.4
single_shot_router,250,0.936,138.14,279.75,85.6,11.6,2.8
correctness_gate_router,250,0.96,1004.6,2600.14,39.6,23.6,36.8
hybrid_gate,250,0.992,666.59,2656.09,72.0,4.8,23.2
hybrid_dumb,250,0.992,887.41,2718.78,68.0,0.0,32.0
```

### `router/concurrent_experiment_summary.csv` (all 9×3 + overall)
```
strategy,concurrency_phase,n_requests,n_errors,accuracy,avg_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,throughput_rps,pct_fast,pct_balanced,pct_heavy,phase_elapsed_s
always_fast,light,559,0,0.9106,192.44,169.86,413.18,537.39,18.63,100.0,0.0,0.0,30.37
always_fast,heavy,413,0,0.8959,855.5,829.8,1365.82,1733.53,13.77,100.0,0.0,0.0,30.79
always_fast,burst,213,0,0.8967,1772.68,1789.42,2653.24,3199.38,7.1,100.0,0.0,0.0,31.3
always_fast,overall,1185,0,0.903,707.58,536.71,2156.23,2635.54,39.5,100.0,0.0,0.0,92.49
always_heavy,light,530,0,0.9887,209.07,180.62,431.93,635.87,17.67,0.0,0.0,100.0,30.12
always_heavy,heavy,411,0,0.9927,908.03,886.94,1382.59,1826.76,13.7,0.0,0.0,100.0,30.52
always_heavy,burst,205,0,0.9902,1788.17,1872.87,2500.17,2883.27,6.83,0.0,0.0,100.0,30.92
always_heavy,overall,1146,0,0.9904,742.22,546.67,2121.14,2483.84,38.2,0.0,0.0,100.0,91.59
round_robin,light,525,0,0.939,201.71,181.79,391.46,582.65,17.5,33.3,33.3,33.3,30.11
round_robin,heavy,406,0,0.9483,909.24,887.36,1437.89,1760.61,13.53,33.5,33.3,33.3,31.44
round_robin,burst,234,0,0.9615,1646.56,1737.28,2543.67,2796.53,7.8,33.3,33.3,33.3,30.54
round_robin,overall,1165,0,0.9468,738.49,590.34,2130.73,2543.76,38.83,33.4,33.3,33.3,92.13
rule_based,light,462,0,0.9848,238.48,192.6,511.63,725.52,15.4,77.1,0.0,22.9,30.05
rule_based,heavy,356,0,0.9691,1050.83,904.14,2104.73,2583.2,11.87,72.5,0.0,27.5,30.73
rule_based,burst,271,0,0.9779,1752.94,1570.64,3590.94,4121.41,9.03,77.5,0.0,22.5,31.05
rule_based,overall,1089,0,0.978,880.92,642.34,2441.4,3655.39,36.3,75.7,0.0,24.3,91.87
ml_router,light,394,0,0.9797,303.02,254.71,580.7,895.87,13.13,76.1,19.3,4.6,30.54
ml_router,heavy,276,0,0.9891,1378.52,1120.67,2535.65,2998.88,9.2,65.6,9.4,25.0,31.03
ml_router,burst,182,0,0.989,2832.82,2560.14,4746.99,5413.96,6.07,53.8,20.3,25.8,32.9
ml_router,overall,852,0,0.9847,1191.82,793.47,3886.1,4824.56,28.4,68.0,16.3,15.7,94.51
single_shot_router,light,482,0,0.9315,217.14,195.39,425.17,614.19,16.07,86.3,11.8,1.9,30.5
single_shot_router,heavy,494,0,0.9393,906.0,884.56,1495.8,1845.02,16.47,72.9,11.1,16.0,31.35
single_shot_router,burst,435,0,0.9609,1606.46,1637.53,2580.53,2856.53,14.5,67.6,17.0,15.4,30.9
single_shot_router,overall,1411,0,0.9433,886.63,759.85,2266.28,2669.36,47.03,75.8,13.2,11.0,92.76
correctness_gate_router,light,452,0,0.9491,243.5,206.51,522.56,634.47,15.07,39.6,23.0,37.4,30.05
correctness_gate_router,heavy,429,0,0.972,1043.57,1031.35,1692.77,2004.15,14.3,41.3,20.3,38.5,30.68
correctness_gate_router,burst,343,1,0.9534,2324.68,2347.52,3468.81,4477.35,11.43,35.6,24.2,40.2,32.18
correctness_gate_router,overall,1224,1,0.9583,1107.13,888.52,2876.19,3674.74,40.8,39.1,22.4,38.6,92.93
hybrid_gate,light,465,0,0.9935,251.75,198.25,600.47,849.65,15.5,70.3,5.6,24.1,30.73
hybrid_gate,heavy,448,0,0.9888,1073.76,939.16,2086.4,2534.82,14.93,71.9,4.7,23.4,31.32
hybrid_gate,burst,406,1,0.9975,1971.54,1751.1,3708.06,4528.69,13.53,74.4,4.2,21.4,32.66
hybrid_gate,overall,1319,1,0.9932,1060.31,872.06,3069.88,3894.84,43.97,72.1,4.9,23.0,94.72
hybrid_dumb,light,479,0,0.9875,247.02,198.88,528.52,774.83,15.97,65.3,0.0,34.7,30.21
hybrid_dumb,heavy,465,0,0.9914,1049.12,888.15,2103.95,2331.6,15.5,65.8,0.0,34.2,31.29
hybrid_dumb,burst,407,0,0.9975,1962.28,1660.67,3576.67,4216.06,13.57,66.8,0.0,33.2,32.78
hybrid_dumb,overall,1351,0,0.9919,1039.83,783.84,3117.03,3707.12,45.03,66.0,0.0,34.0,94.31
```

### `monitoring/reports/audc_comparison.csv`
```
model,audc_raw,audc_normalized,best_naive_acc,best_naive_lambda,best_naive_sigma,best_defensible_acc,best_defensible_lambda,best_defensible_sigma,sweep_lambda_min,sweep_lambda_max,note
CorrectnessGateNet_V1,0.088036,7.3363,0.9917,0.05,,0.95,0.15,4.5967,0.05,0.95,"swept curve, trapezoidal AUDC"
CorrectnessGateNetV2b,0.083681,6.9734,0.9818,0.05,2.2646,0.9686,0.17,4.23,0.05,0.95,"swept curve, trapezoidal AUDC"
rule_based,,,0.972,,,0.972,,,,,"single operating point, not swept"
ml_router,,,0.972,,,0.972,,,,,"single operating point, not swept"
```

### `monitoring/reports/confident_zone_validation.csv`
```
metric,value,true_label,fast_confidence,rb_predicted,gate_predicted
confident_zone_images,215,,,,
uncertain_zone_images,35,,,,
band_low,0.6,,,,
band_high,0.8,,,,
both_right,204,,,,
both_wrong,1,,,,
rb_right_gate_wrong,9,,,,
rb_wrong_gate_right,1,,,,
gate_wins_image,data/test/Tomato__Healthy/freshTomato (142).jpg,Tomato__Healthy,0.3811,Jujube__Healthy,Tomato__Healthy
rule_based_accuracy_confident_zone,0.9907,,,,
gate_accuracy_confident_zone,0.9535,,,,
accuracy_difference,-0.0372,,,,
rule_based_accuracy_full,0.98,,,,
gate_accuracy_full,0.96,,,,
```

### `router/model_benchmarks.json`
```json
[
  {"tier":"fast","avg_latency_ms":0.549,"p50_latency_ms":0.517,"p95_latency_ms":0.815,"cpu_memory_delta_mb":0.0,"gpu_model_footprint_mb":1.52,"gpu_peak_inference_mb":22.47,"checkpoint_size_mb":1.53,"num_params":396572,"device":"cuda"},
  {"tier":"balanced","avg_latency_ms":4.842,"p50_latency_ms":4.729,"p95_latency_ms":5.57,"cpu_memory_delta_mb":0.0,"gpu_model_footprint_mb":24.71,"gpu_peak_inference_mb":32.18,"checkpoint_size_mb":16.37,"num_params":4237900,"device":"cuda"},
  {"tier":"heavy","avg_latency_ms":6.291,"p50_latency_ms":6.084,"p95_latency_ms":8.429,"cpu_memory_delta_mb":0.0,"gpu_model_footprint_mb":23.79,"gpu_peak_inference_mb":34.81,"checkpoint_size_mb":15.72,"num_params":4043416,"device":"cuda"}
]
```

### `training/checkpoints/correctness_gate_v2b_lambda_sweep_finegrid.csv` ★ Primary finegrid (22 rows)
```
lambda,pct_fast,pct_balanced,pct_heavy,pct_expensive,balanced_routing_acc,random_baseline_mean,random_baseline_std,sigma
0.1,23.17,59.33,17.5,76.83,0.9586,0.9467,0.0061,1.9547
0.105,24.67,59.33,16.0,75.33,0.9589,0.9458,0.0058,2.2437
0.11,26.67,58.17,15.17,73.33,0.9597,0.9454,0.0057,2.5031
0.115,28.33,57.0,14.67,71.67,0.9604,0.9456,0.0058,2.5488
0.12,30.17,55.67,14.17,69.83,0.9648,0.9453,0.0066,2.9635
0.125,32.33,54.67,13.0,67.67,0.9615,0.9455,0.0068,2.3426
0.13,34.17,53.5,12.33,65.83,0.9602,0.9458,0.0065,2.2022
0.135,36.0,52.33,11.67,64.0,0.9588,0.9454,0.0061,2.1838
0.14,37.17,51.67,11.17,62.83,0.9573,0.9456,0.006,1.9624
0.145,38.0,51.67,10.33,62.0,0.9619,0.9454,0.0052,3.185
0.15,40.33,49.67,10.0,59.67,0.9613,0.9452,0.0057,2.8073
0.155,42.5,48.17,9.33,57.5,0.9614,0.9454,0.0053,3.0384
0.16,44.0,47.33,8.67,56.0,0.9623,0.9453,0.0057,2.9983
0.165,46.5,45.5,8.0,53.5,0.9621,0.9457,0.0055,2.9532
0.17,47.5,45.33,7.17,52.5,0.9686,0.9457,0.0054,4.23
0.175,48.67,44.67,6.67,51.33,0.9672,0.9461,0.0065,3.2507
0.18,50.0,43.33,6.67,50.0,0.9675,0.9461,0.0064,3.362
0.185,52.5,42.0,5.5,47.5,0.9642,0.9472,0.0067,2.5362
0.19,54.17,40.5,5.33,45.83,0.9629,0.9468,0.0072,2.2337
0.195,56.17,38.83,5.0,43.83,0.9625,0.9481,0.0065,2.2215
0.2,57.83,37.33,4.83,42.17,0.9613,0.9479,0.0062,2.1786
```

### `McNemar hybrid_gate vs always_heavy` (`training/mcnemar_hybrid_vs_heavy.py` live rerun)
```
                    always_heavy=1  always_heavy=0
hybrid_gate=1                247              1
hybrid_gate=0                  0              2
Discordant: 1 (hybrid_only 1, heavy_only 0), exact binomial n<25
statistic 0.0, p-value 1.000000 (α0.05 => NOT significant)
hybrid_gate 0.992 (248/250) vs always_heavy 0.988 (247/250)
```

### Counts
* `router/utility_labels.csv`: 4000 rows, `best_tier fast 3415 / balanced 363 / heavy 222`, Windows `\` paths, 22 cols.
* `router/held_out_images.csv`: 251 lines (250+header), `image_path,true_label` only.
* `data/test`: 4410 files, `data/train` 20496, `data/val` 4385, `data/raw` 29291, 28 classes.

---

## 19. Appendix B — Checkpoint & MLflow Inventory

**`training/checkpoints/` (`ls -lh`):**
`fast_cnn_best.pt 1.6M` (val 0.8753 ep19), `balanced_mobilenet_best.pt 17M` (val 0.9818 ep14), `heavy_efficientnet_best.pt 16M` (val 0.9815 ep13), `gate_net_best.pt 103K` (val bal 0.3929), `confidence_gate_best.pt 103K` (val decision bal 0.4301), `correctness_gate_best.pt 103K` (val avg AUROC 0.7622 ep21), `correctness_gate_v2_best.pt 130K` (val 0.5734 ep25), `correctness_gate_v2b_best.pt 232K` (val 0.6991 ep5, best test 0.7572), `correctness_gate_v2c_costaware_best.pt 233K` (val bal 0.9359 ep12, `lambda_train 0.17`) + 8× `*_curves.png` + 3× `*_latency_benchmark.json` + 5× `*_lambda_sweep*.csv`.

**MLflow `inferpilot-model-tiers` (experiment id 1, 11 runs via `MlflowClient().search_experiments()`):**
* `dc7ce7c5 correctness_gate` FINISHED 24003 params `test_avg_auroc 0.7068 (0.6764/0.7692/0.6748)`
* `1bfec8d9 correctness_gate_v2` FINISHED 123 trainable `test_avg_auroc 0.5438`
* `c89fea33 correctness_gate_v2b` FINISHED 24483 trainable `test_avg_auroc 0.7572 (0.7339/0.7724/0.7653)`
* `81e51d40 / 10ab0fb0 / c765e5f2 correctness_gate_v2c_costaware` (2 FINISHED `lambda 0.17, T_target 0.3/0.05`, 1 FAILED) `test_balanced_routing_acc 0.9599`
* `b9d2e1e6 confidence_gate` FAILED
* `ebda3d83 gate_net` FINISHED `test_balanced_acc 0.3951`
* `774a809a heavy_efficientnet` FINISHED `test_acc 0.985`
* `0fd956d8 balanced_mobilenet` FINISHED `0.9819`
* `4ae03b15 fast_cnn` FINISHED `0.8812`
* `router/train_router.py` cascade comparison NOT in MLflow (CSV only).

---

*End of dossier. For re-run: `SIMULATED_HEAVY_DELAY_MS=0 python router/run_core_experiment.py` (sequential) and `SIMULATED_HEAVY_DELAY_MS=0 python router/run_concurrent_experiment.py` (9×94s). All numbers above are from files as of this write; re-running training will update `mlruns/` and `*_best.pt` but `utility_labels.csv` is deterministic `seed 42`.*
