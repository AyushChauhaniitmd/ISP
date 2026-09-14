# DP-ForgetBench: A Privacy-Aligned Audit of Federated Unlearning

**When is explicit client unlearning redundant? A privacy-aligned audit of differentially private federated learning.**

This repository builds a rigorous, reproducible benchmark for client deletion in federated learning (FL). The core objective is not to claim that Differential Privacy (DP) perfectly deletes data, nor is it to inflate exploratory metrics to claim "State-of-the-Art" (SOTA). 

The goal is sharper and more scientifically grounded: **To compare DP-only model release, explicit federated unlearning baselines, and exact retraining references under a privacy contract where the unit of privacy matches the unit of the deletion request.**

---

## 🚀 Phase 2 Major Upgrades & Current Status

In the most recent phase of development, the DP-ForgetBench execution pipeline was dramatically upgraded to solve critical blockers and formalize the evaluation framework. 

### 1. Architecture & Convergence Fixes
* **The Problem:** Early CIFAR-10 experiments under DP-FL collapsed to chance-level accuracy (~10%).
* **The Fix:** We audited the model and found a flaw in `SmallGroupNormCNN` where aggressive `AdaptiveAvgPool2d((1, 1))` pooling was destroying spatial features. This was replaced with `Flatten(1)`.
* **DP Hyperparameter Sweep:** We systematically swept DP hyperparameters and scaled the population size to $N=100$ clients ($20,000$ samples) with `noise_multiplier=0.5`. The model now successfully learns under strict client-level DP, achieving **>54% accuracy** and breaking the random-chance plateau.

### 2. Retrain Ensemble & Validity Gating
* **Retrain Variability:** A single target retrain ($M_R$) is too statistically noisy to serve as the ground-truth "gold standard" for unlearning. We implemented **Retrain Ensembles**, where the pipeline automatically trains $N$ independent target retrains using different random seeds to form an empirical distribution of retrain variability.
* **Validity Gate Layer:** The execution pipeline (`run.py`) now includes an automated validity check. If the median test accuracy of the retrain ensemble falls below a threshold (e.g., 40%), the run is explicitly flagged as `INVALID`.

### 3. Marginal Unlearning Benefit (MUB)
We formalized the definition of redundancy by introducing the **Marginal Unlearning Benefit (MUB)**. 
Calculated via `scripts/aggregate_mub.py` with 95% Bootstrap Confidence Intervals, MUB measures the exact gain of applying an explicit unlearning algorithm over doing nothing (DP-only).
* **Utility MUB:** Does unlearning bring test accuracy closer to the retrain ensemble?
* **Alignment MUB:** Does unlearning match the predictive behavior of the retrain ensemble better than DP-only? (Measured via Jensen-Shannon Divergence).
* **Privacy MUB:** Does unlearning reduce the Membership Inference Attack (MIA) advantage?

---

## 🔬 The Evaluation Framework

The benchmark rigorously evaluates three distinct outcomes for every configuration:

1. **$M_{DP}$ (DP-Only / No Action):** A model trained on all data with client-level DP, left completely unchanged after a deletion request.
2. **$M_U$ (Explicit Unlearning):** The model obtained by applying a specific federated unlearning algorithm after the deletion request.
3. **$M_R$ (Exact Retrain):** The gold-standard models, trained from scratch on the dataset *without* the deleted client's data.

By comparing $M_U$ and $M_{DP}$ against the $M_R$ ensemble, we can map the **Redundancy Frontier**—the specific combinations of privacy budget ($\epsilon$), data heterogeneity ($\alpha$), and deletion size where explicit unlearning adds material benefit vs. where it is statistically redundant.

---

## 🔒 The Privacy Rule: Client-Level DP

The primary experiment utilizes **client-level central DP**:
* Whole client updates are L2-clipped.
* Gaussian noise is added to the aggregate at the server.
* The DP accountant strictly tracks client-sampling privacy loss. 

*Note: Example-level DP-SGD belongs only to a separate record-deletion control and must never be conflated with client-level removal guarantees.*

---

## 🛠️ Quick Start Guide

Python 3.10-3.13 is supported.

### 1. Installation
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Run a Deterministic Smoke Test
```powershell
python -m dp_forgetbench.run --config configs/phase0_toy.yaml --mode full
```

### 3. Execute the Phase 2 DP Sweep & Pilot
To run the validated configuration that achieves >54% accuracy under client-level DP:
```powershell
python scripts/run_dp_sweep.py
# Or run the pilot directly:
python -m dp_forgetbench.run --config configs/phase2_private_pilot.yaml --mode full
```

### 4. Validate Artifacts
To check if a run passed the Validity Gate (accuracy >40%):
```powershell
python scripts/validate_artifact.py results\<run-dir>
```

### 5. Generate Documentation PDFs
```powershell
python scripts/build_pdf.py RESEARCH_GRADE_PLAN.md reports/RESEARCH_GRADE_PLAN.pdf
python scripts/build_pdf.py README.md reports/README.pdf
```

---

## 📁 Repository Layout

```text
configs/                 Immutable YAML experiment configurations (sweeps, pilots, grids)
docs/                    Protocol and threat-model documentation
partitions/              Versioned deletion manifests generated by runs
reports/                 PDFs, result summaries, validity notices
results/                 Raw experiment outputs (metrics, privacy ledgers, metadata)
scripts/                 Grid generation, validation, MUB aggregation, and PDF helpers
src/dp_forgetbench/      Core Benchmark Implementation (FL, DP, Unlearning, Evaluation)
tests/                   Pytest suite for determinism, privacy, deletion, and smoke testing
```

## 📊 What Each Run Outputs

Inside every result directory under `results/`, the framework securely logs:
* `metrics.json`: Utility, deleted/retained metrics, JS alignment, attack diagnostics, and computational cost.
* `privacy_ledger.json`: The exact DP mechanism, $\epsilon$, $\delta$, sampling rate, and release counts.
* `run_metadata.json`: Configuration hashes, runtime details, deletion manifest checksums, and reference configs.

---

## 🔮 What Is Still Needed For A Final Paper Claim?

While Phase 2 solidified the core architecture, DP convergence, and evaluation rigor, the following steps are required before a final scientific claim can be made:
1. **Faithful Unlearning Baselines:** Implementation of established algorithms like FedEraser or Retain-Data Fine-Tuning.
2. **Upgraded Privacy Attacks:** Moving LiRA/A-LiRA and Tri-population (Forget/Retain/Unseen) probes from baseline implementations to rigorous, calibrated audits.
3. **Core Factorial Execution:** Running the full grid across $\epsilon \approx \{2, 8, \infty\}$ and Dirichlet $\alpha = \{1.0, 0.1\}$.
4. **Final Confirmation:** $\ge 10$ confirmation seeds on frontier transition cells.

*Read [RESEARCH_GRADE_PLAN.md](RESEARCH_GRADE_PLAN.md) before changing the experimental scope.*
