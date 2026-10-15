# EnergyMind Tunisia: trustworthy smart-metered neighbourhood

IEEE TSYP14 · PES × IES challenge "Architects of the Energy Noosphere" · **Phase 1 submission**

**Scope.** We monitor smart-metered low-voltage neighbourhoods. Each node is one neighbourhood with one transformer, 2–3 feeders, 15–30 meters and rooftop PV. The dataset has 3 nodes (N1 coastal suburb: 30 meters / 3 feeders; N2 interior town: 15 / 2; N3 mixed residential–commercial: 15 / 2), so 60 meters on 7 feeders in total. **The Phase 1 prototype runs on node N1** (`energymind.data.NODE`; `load()` filters to it, `load(node=None)` keeps all three). Goal 4 connects N1, N2 and N3; multi-node sharing is a documented design in Phase 1, as the challenge allows. For every suspicious reading, the system decides whether the **grid** did something unusual or the **data** is wrong, because the two need opposite responses. It explains the verdict, routes it to the right level of human control, and lets several nodes learn from one another without sharing raw readings.

| Goal | Question | Design | Code |
|---|---|---|---|
| 1. Detection | Is this reading unusual or biased? | [`docs/goal1_detection`](docs/goal1_detection/goal1_detection.pdf) | `src/energymind/detect/` |
| 2. Confidence & explainability | What is it, how sure are we, and why? | [`docs/goal2_explainability`](docs/goal2_explainability/goal2_confidence_explainability.pdf) | `src/energymind/explain/` |
| 3. Human–AI escalation | Who decides or acts? | [`docs/goal3_governance`](docs/goal3_governance/goal3_escalation_governance.pdf) | `src/energymind/govern/` |
| 4. Shared learning | What do nodes share, and what stays local? | [`docs/goal4_shared_learning`](docs/goal4_shared_learning/goal4_shared_learning.pdf) | `src/energymind/shared/` |

**Start here:** the full Phase 1 proposal, [`docs/proposal/proposal.pdf`](docs/proposal/proposal.pdf) (46 pages): scope, objectives, data and anomaly-simulation plan, and the complete preliminary design of each goal (Goal 3 condensed). The standalone goal reports below are its detailed sources; `docs/proposal/build_parts.py` rebuilds the proposal from them.

## Dataset

- **File:** `data/energymind_full_joined.csv.gz`, 33 MB.
- **Size:** 1,051,200 rows × 67 columns, 30-min resolution, year 2025.
- **Nodes:** 3 (N1 coastal, N2 interior, N3 mixed), 60 meters in total.
- **Faults:** 29 injected fault and event types with exact labels.
- **Data card:** [`docs/data_card.md`](docs/data_card.md).
- **Sample:** one N1 week in `data/sample/` lets everything run in seconds.
- **Reproducible:** `simulate/` regenerates the file exactly from seed 2025 (all three neighbourhoods, about 35 s).

All data are simulated and calibrated on public Tunisian figures. No real STEG data is included.

## Quick start

```bash
pip install -r requirements.txt
python scripts/check_dataset.py --sample                  # dataset facts used by the design reports
pytest -q                                                 # rule, cascade and policy tests
cd docs/goal1_detection && python n1_experiment.py        # Goal 1 results on N1 (about 30 s)
(cd simulate && python inject_anomalies.py --out ../build --seed 2025 && python join_dataset.py --data ../build/data --out ../build/energymind_full_joined.csv.gz)   # regenerate the dataset
jupyter notebook notebooks/01_dataset_tour.ipynb          # tour of the dataset
```

## Repository layout

```
├── data/                    dataset (.csv.gz) + one-week sample
├── docs/
│   ├── proposal/            full Phase 1 proposal (LaTeX parts + PDF, build_parts.py)
│   ├── data_card.md         dataset description, injection recipes, limits
│   ├── decision_log_schema.json   shared record format, Goals 2 + 3
│   ├── goal1_detection/     report (LaTeX + PDF); gen.py (figures), n1_experiment.py (results)
│   ├── goal2_explainability/  report (LaTeX + PDF)
│   ├── goal3_governance/      report (LaTeX + PDF)
│   └── goal4_shared_learning/ report (LaTeX + PDF)
├── configs/
│   ├── thresholds.yaml      rule and cascade thresholds (versioned)
│   └── assets.yaml          synthetic asset-criticality overlay (Goal 3)
├── src/energymind/
│   ├── data/                loading, label hygiene, 4-class map, event-grouped folds
│   ├── detect/              Goal 1: physics rules, robust z, CUSUM
│   ├── explain/             Goal 2: rule cascade, hash-chained decision log
│   ├── govern/              Goal 3: routing policy
│   └── shared/              Goal 4: calibration summaries, robust aggregation
├── simulate/                dataset generator: generate_baseline.py → inject_anomalies.py → join_dataset.py
├── notebooks/               01_dataset_tour.ipynb
├── scripts/                 check_dataset.py, make_sample.py
└── tests/
```

## Evaluation conventions (all goals)

- **Model inputs:** use only `feature_columns(df)`. Never use `gt_*`, oracle columns or simulator settings.
- **Unsupervised training:** fit on `gt_label == "normal"` rows only.
- **Calibration and metrics:** compute them out-of-fold with `event_folds(df)`, which groups folds by `gt_event_id`. Do not use the Sep–Oct / Nov–Dec time split, because most rare fault types have no event after August.
- **Reporting:** give results per `gt_subtype`, and per event with confidence intervals. Many subtypes have only 2–4 events in the year.

## Status (Phase 1)

| Part | State |
|---|---|
| Dataset + data card | ✅ done |
| Design reports, Goals 1–4 | ✅ done |
| Goal 1 detection stack (rules, statistics, IF, OCSVM, autoencoder) run on N1 | ✅ `docs/goal1_detection/n1_experiment.py` |
| Physics rules, verdict cascade, decision log | ✅ prototype + tests |
| Goal 3 escalation engine (V1), 6 scenarios + 5 safety invariants on 400 random cases | ✅ `src/energymind/govern/policy.py`, `tests/test_governance.py` |
| Calibration fit, SHAP, LLM explanation, operator console | 🔜 Phase 2 |
| Multi-node sharing | design only (allowed in Phase 1) |
