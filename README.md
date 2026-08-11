# Fraud Detection on IEEE-CIS: Anomaly Detection vs. Supervised Classification

An end-to-end, reproducible fraud detection study on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) dataset — 590,540 e-commerce transactions, 3.5% fraud — built to run start-to-finish in a free Google Colab session in about an hour.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Notebook](https://img.shields.io/badge/notebook-Colab%20ready-orange)

> **Research question:** can unsupervised anomaly detection identify fraudulent transactions purely from their deviation from normal behaviour, and how does that compare with a supervised classifier trained on labels?

**Answer:** deviation from normal behaviour carries genuine fraud signal — Isolation Forest ranks fraud **4.5× better than chance** having never seen a label — but it captures only about a quarter of what a supervised model achieves on identical features. The useful part of the result is *why*, and where each approach fails.

---

## Results

Held-out **final 15% of the timeline** (88,581 transactions, 3,083 frauds, 3.480% base rate). Models were frozen and thresholds fixed on the validation split before the test set was scored once. `K = 886` is the 1%-of-volume "investigation capacity" point.

| Model | Uses labels? | PR-AUC | ROC-AUC | P@100 | P@500 | P@886 | R@886 | Fraud $ captured | Fit time |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LightGBM** (supervised) | yes | **0.574** | **0.904** | **1.000** | **0.938** | **0.895** | **0.257** | 14.1% | 35 min |
| **Isolation Forest** | no | 0.155 | 0.769 | 0.190 | 0.260 | 0.307 | 0.088 | 5.3% | 46 s |
| **Autoencoder** † | no | 0.128 | 0.748 | 0.160 | 0.198 | 0.219 | 0.063 | 3.4% | 69 s |
| **Local Outlier Factor** | no | 0.036 | 0.523 | 0.020 | 0.036 | 0.036 | 0.010 | 1.1% | 32 s |
| *random ranking* | — | *0.035* | *0.500* | *0.035* | *0.035* | *0.035* | *0.010* | *~3.5%* | — |

† Fitted on non-fraud rows only, which makes it semi-supervised: labels shape the training *population* but never enter the loss.

**Precision@100 = 1.000** for the supervised model — the 100 highest-scoring transactions in the final period were all fraud. The Isolation Forest queue is 31% fraud at the capacity point versus 3.5% by chance: a 9× lift from a model that was never told what fraud looks like.

### The threshold is the product decision

Same LightGBM model, three operating points chosen on validation, evaluated on test:

| Operating mode | Alerts | Alert rate | Precision | Recall |
|---|---:|---:|---:|---:|
| High recall (≥80% on validation) | 8,982 | 10.1% | 25.2% | 73.5% |
| Balanced (max F1) | 2,396 | 2.7% | 64.2% | 49.9% |
| Investigation capacity (top 1%) | 748 | 0.8% | **91.7%** | 22.3% |

A 3.6× swing in precision and a 3.3× swing in recall out of one frozen model. Under the notebook's hypothetical cost assumptions (\$5 per wasted investigation, \$100 + the transaction amount per missed fraud), high-recall is cheapest — it avoids \$538k of a \$778k baseline — while the capacity mode, which looks best on precision, saves only \$135k. Which column matters is a business question, not a modelling one.

### Where the models fail

The error analysis is the most interesting output in the notebook:

- **Missed fraud is expensive fraud.** At the capacity threshold the model catches 686 frauds worth \$66k and misses 2,397 worth \$403k. Median caught: \$39.78. Median missed: \$75.00. Value-weighted capture (14.1%) trails count-based recall (22.3%), so the ranking is mildly biased toward cheap fraud — a real weakness for anything charged with limiting losses rather than counting cases.
- **Misses look normal.** False negatives sit on familiar entities with ordinary amounts and fewer populated fields (median 106 missing values vs 51 for true positives). That is precisely the blind spot of a deviation-based detector, and the argument for running both tracks rather than picking one.
- **False alarms are rare but concentrated.** Only 62 false positives at the capacity point (8.3% of the queue), skewed toward small amounts on identity-carrying transactions.

### What drives the predictions

**Supervised (mean |SHAP|):** `C13`, `V70`, `nuniq_card1__device_name`, `D1n`, `C14`, `P_email_bin`, `C1`, `amt_ratio_addr1`. Encouragingly, three engineered features — device diversity per card, client tenure, and amount relative to the address's norm — rank alongside the vendor's anonymised blocks.

**Unsupervised (largest separation between the most- and least-anomalous 2,000 test rows):** `has_identity`, `email_match`, and the `C4`/`C8`/`C10` counting features. The detectors are largely keying on *identity-block presence* rather than transaction behaviour — worth knowing, and a caveat on how much "behavioural anomaly detection" is really happening. See [Notes on this run](#notes-on-this-run).

---

## Why the methodology looks like this

- **Chronological split, never random** (70/15/15 on `TransactionDT`). A random split leaks the same card, day and fraud campaign across both sides. Deployment means training on the past and scoring the future, and `max(train) < min(val) < min(test)` is asserted, not assumed.
- **PR-AUC and Precision@K, not accuracy.** Predicting "legitimate" for everything scores 96.5% accuracy and catches zero fraud. At a 3.5% base rate ROC-AUC is also flattering — the supervised model's 0.904 ROC-AUC sits alongside a far more sober 0.574 PR-AUC.
- **Thresholds are an operational choice**, computed three ways on validation and never re-tuned on test.
- **A documented leakage audit** naming what was excluded and why, including risks it *cannot* eliminate: the anonymised `V`/`C`/`D` blocks are vendor-engineered aggregations with undisclosed computation windows.
- **Train-only fitting of every statistic** — encoders, entity frequencies, per-entity amount statistics, scaler. The one deliberate compromise, that training-split aggregations are not point-in-time correct, is stated in the notebook rather than hidden.

---

## What's in the notebook

26 sections, 50 executable cells, 19 figures, every stage narrated:

| | |
|---|---|
| **Data** | automated ingestion, validation battery, EDA, leakage audit, temporal split |
| **Features** | 264 supervised / 69 anomaly features: row-wise, entity aggregations, missingness indicators |
| **Unsupervised** | Isolation Forest, Local Outlier Factor (PCA subspace), Keras autoencoder (PCA fallback) |
| **Supervised** | LightGBM → XGBoost → HistGradientBoosting fallback chain, seeded random search |
| **Evaluation** | threshold optimisation, PR/ROC/Precision@K curves, confusion matrices, comparison table |
| **Explainability** | SHAP for the classifier; deviation + per-feature reconstruction error for the detectors |
| **Analysis** | error analysis, temporal drift across three test windows, hypothetical cost framework |
| **Delivery** | model persistence, experiment log, Gradio app, auto-generated executive summary |

---

## Quickstart

1. **Open in Colab:**

   ```
   https://colab.research.google.com/github/<USER>/<REPO>/blob/main/ieee_cis_fraud_anomaly_detection.ipynb
   ```

2. **Get the data.** The dataset is a Kaggle competition asset and is *not* redistributed here. Two routes; the notebook uses whichever it finds:

   - **Archive (no credentials).** Drop `ieee-fraud-detection.zip` (~118 MB) into Drive — `MyDrive/ieee_cis_fraud_detection/data/raw/` is ideal, the root works too. Section 4.1 also provides an upload button. Only the two training CSVs are extracted, and the archive is matched by contents, so a renamed download still works.
   - **Kaggle API.** Accept the [competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) once, create an API token, and add `KAGGLE_USERNAME` / `KAGGLE_KEY` as Colab secrets. No credential is ever written into the notebook.

3. **Run all.** ~60–70 minutes on a free CPU runtime, or ~15 minutes with `FAST_MODE = True`. Peak RAM ~4 GB of Colab's 13 GB.

Everything persists to Drive, so a second run resumes from a Parquet cache and skips ingestion and CSV parsing entirely.

**Runtime breakdown** (reference run): supervised training and tuning 35 min · SHAP 19 min · ingestion, features and preprocessing ~5 min · all three anomaly detectors under 3 min combined. Set `RUN_SHAP = False` and `RUN_HYPERPARAMETER_TUNING = False` for a ~20-minute pass.

### Configuration

All switches live in one cell (Section 2) and are written to `artifacts/experiment_config.json` with every run:

```python
FAST_MODE     = False      # subsample everything for a smoke test
V_MODE        = "reduced"  # 120 curated V columns, or "all" for 339
RUN_LOF = RUN_AUTOENCODER = RUN_HYPERPARAMETER_TUNING = RUN_SHAP = RUN_GRADIO = True
CAPACITY_RATE = 0.01       # investigation capacity = top 1% of transactions
COST_FALSE_POSITIVE, COST_FALSE_NEGATIVE = 5.0, 100.0   # hypothetical, for comparison only
```

Any expensive component can be switched off without breaking downstream cells — each checks whether its artifact exists.

---

## Outputs

Written to `MyDrive/ieee_cis_fraud_detection/`:

```
data/raw/          archived competition zip
data/cache/        Parquet cache of the parsed frames (42 MB + 4 MB)
models/            isolation_forest · lof · autoencoder.keras · supervised_model · preprocessor (24 MB)
artifacts/         experiment_config · feature_metadata · thresholds · evaluation_results · manifests
figures/           19 PNGs
reports/           model_comparison · experiment_log · leakage_audit · temporal_drift · cost_analysis
predictions/       test_predictions · top_anomalies · gradio_demo_sample
logs/              per-run execution log
```

The **Gradio app** (Section 23) loads these saved artifacts rather than the in-memory objects — which doubles as an end-to-end test that persistence works. It has a demo tab scoring real held-out transactions and a manual what-if tab; risk bands are derived from the validation thresholds (medium ≥ 0.021, high ≥ 0.278, critical ≥ 0.976), not hard-coded. The public share link Colab prints is temporary and expires within a week.

---

## Engineering notes

Two findings worth more than the leaderboard numbers.

**Early stopping silently trained a one-tree model.** LightGBM halts when *any* monitored metric stops improving. With `scale_pos_weight` set for a 27:1 imbalance, weighted log-loss degrades from iteration 1 while PR-AUC is still climbing — so the default configuration stopped after a single tree and still reported a plausible 0.83 ROC-AUC. Nothing errored. Setting `metric="average_precision"` plus `first_metric_only=True` moved validation PR-AUC from **0.15 to 0.68** and let the model train to ~1,500 iterations. A metric that looks reasonable is not evidence that training worked; if a boosted model's `best_iteration_` is in single digits, something is wrong.

**No SMOTE, deliberately.** Interpolating between ordinal category codes, count features and `NaN`-carrying columns fabricates transactions that could not physically occur — a fractional card ID, a blended device string. Class weighting achieves the same objective without inventing records.

---

## Notes on this run

Honest observations from the reference run, including places where the notebook's narrative is templated and does not match what the data actually showed. These are documented rather than papered over, and are the first things to address in a revision.

1. **Performance did not decay across the test period — it improved.** PR-AUC by window (early → middle → late): LightGBM 0.585 → 0.524 → 0.611 (+4.5%), Isolation Forest 0.139 → 0.148 → 0.181 (+30%), Autoencoder +41%. The commentary in Section 19 and "key finding 4" of the auto-generated executive summary assert that performance decays; **that text is templated and is not supported by this run.** The likely explanation is a shift in fraud mix in the final weeks rather than genuinely improving generalisation — but either way the correct reading is "no measurable decay within six months", and the retraining argument rests on general principle, not on this evidence.
2. **The tree budget was the binding constraint.** Every configuration stopped at `best_iter` 1493–1499 against an `n_estimators=1500` cap, with validation PR-AUC still rising. Raising the budget (or lowering the learning rate with a larger cap) should improve the supervised numbers further, at several minutes per fit.
3. **Local Outlier Factor is effectively uninformative here** (ROC-AUC 0.523, PR-AUC 0.036 against a 0.035 base rate). The approximations forced by cost — a 20,000-row reference set in a 15-dimensional PCA subspace — are the likely cause. It should either be given a proper budget or dropped; it is retained as a documented negative result.
4. **Anomaly explanations over-weight binary features.** The rank-Gauss transform saturates at roughly ±5.2σ, and binary flags (`is_night`, `is_weekend`, `amt_is_round`, `email_match`) land on that bound, so they dominate the per-transaction deviation ranking regardless of importance. The explanations remain directionally useful, but near-binary features should be excluded from that ranking.
5. **The detectors key on identity-block presence more than on behaviour.** `has_identity` and `email_match` show by far the largest separation between anomalous and normal rows. Since identity data is missing for ~76% of transactions and its presence correlates with the channel, part of what the "anomaly" score measures is *which channel the transaction came through* rather than how unusual the behaviour was.
6. **Value capture lags count recall** (14.1% of fraud value vs 22.3% of fraud cases at the capacity point). A revision should surface fraud-dollar capture as a headline metric and consider amount-weighted training.

---

## Limitations

The notebook has a dedicated section; the headlines:

- The `V`, `C`, `D` and `id_*` blocks are **anonymised**, so findings cannot be translated into business rules, and their undisclosed computation windows mean lookahead cannot be fully ruled out.
- One processor, one merchant mix, one six-month window. Nothing here transfers to another institution without revalidation.
- Labels are imperfect: fraud discovered after the labelling cut-off is recorded as legitimate, so some "false positives" may be undetected fraud — precision is, if anything, understated.
- The cost figures are **assumptions**, not measurements. No chargeback, recovery or operational cost data exists in this dataset, and the largest real cost of a false decline — customer attrition — is not modelled at all.
- Anomaly explanations are **correlational**. A rare device is a reason to look, never evidence about a person.
- Nothing here addresses fairness auditing, adverse-action notification, model governance, latency budgets, or production monitoring.

**This is a research and portfolio prototype on a public benchmark — not a production banking system,** and no decision affecting a real customer should be made from it.

---

## Repository

```
├── ieee_cis_fraud_anomaly_detection.ipynb   # the entire project (2.8 MB with outputs)
├── README.md
└── .gitignore
```

**Do not commit the dataset.** Kaggle's competition rules restrict redistribution, and the files are large (683 MB of CSV, a 118 MB zip). The included `.gitignore` excludes data, models and generated artifacts.

The notebook is committed **with outputs**, so all 19 figures and every result table render on GitHub without running anything. That costs 2.8 MB and makes diffs unreadable; if you would rather keep the history clean, strip them before committing:

```bash
jupyter nbconvert --clear-output --inplace ieee_cis_fraud_anomaly_detection.ipynb
```

Pick one convention and stay with it — a notebook that alternates between the two produces enormous, meaningless diffs.

---

## Reproducibility

Seed 42 throughout (Python, NumPy, scikit-learn, TensorFlow). Every run writes its configuration, package versions, dataset metadata, split boundaries, model parameters and metrics to `artifacts/`, and appends one row per model to `reports/experiment_log.csv`. Every number in this README traces to run `20260811_101329`.

Two caveats on exact reproduction: LightGBM's histogram binning is not bit-identical across thread counts, and the Keras autoencoder varies slightly with backend version — expect PR-AUC to move in the third decimal place, not the first.

## License & acknowledgements

Code released under the MIT License. The dataset is provided by **Vesta Corporation** through the IEEE Computational Intelligence Society competition and is subject to Kaggle's competition rules; it is not included in this repository.
