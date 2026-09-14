# DP-ForgetBench: Comprehensive Scientific & Engineering Documentation

**Differential Privacy vs. Federated Unlearning Benchmark**  
**Repository:** `ISP_G2` / `DP-ForgetBench`  
**Version:** Phase-2 Research-Grade Architecture  
**Author:** Lead Research Engineer, DP-ForgetBench  

---

## 1. Executive Summary & Research Question

### 1.1 The Foundational Research Question
Distributed machine learning environments increasingly confront conflicting regulatory and privacy mandates:
- **The Right to Be Forgotten** (GDPR Art. 17 / CCPA) requires removing a user's data and statistical influence from a trained model upon request.
- **Differential Privacy (DP)** adds mathematically bounded perturbation during training to prevent an adversary from inferring any individual client's presence.

This benchmark answers:
> **“When is explicit client unlearning redundant under client-level differentially private federated learning?”**

### 1.2 What This Project Claims vs. What It Does NOT Claim

| What This Project Does **NOT** Claim | What This Project **DOES** Claim |
| :--- | :--- |
| **NOT**: *"DP perfectly deletes client data."* | Client-level DP bounds the maximum output divergence between adjacent federations (with and without client $u$). |
| **NOT**: *"DP makes unlearning always unnecessary."* | DP makes explicit unlearning redundant **only in parameter regimes where the model converges and client influence is smaller than ordinary retraining noise**. |
| **NOT**: *"DP-only is a certified unlearning algorithm."* | DP-only is an unlearning baseline evaluated against exact retraining ($M_R$); when $\rho \ll 1$, doing nothing matches retraining within random seed variation. |
| **NOT**: *"Near-zero MIA proves unlearning under DP."* | If a model is at chance accuracy ($10\%$), low attack advantage is an artifact of **utility destruction**, not privacy. Unlearning is only studied on models that achieve healthy convergence. |

---

## 2. Project Architecture & Evolution

```
+--------------------------------------------------------------------------------------------------+
|                                    DP-FORGETBENCH EVOLUTION                                      |
+--------------------------------------------------------------------------------------------------+
| Phase 0: Synthetic 1D Linear Verification                                                        |
| - Mathematically verified Poisson-sampled Gaussian RDP accounting under add/remove client bounds |
| - Validated fixed public normalizer (q * N) to prevent sample-size information leakage           |
+--------------------------------------------------------------------------------------------------+
| Phase 1: Real-Data Public Pipeline (CIFAR-10 Binary)                                             |
| - Public dataset loader with deterministic Dirichlet partitioning and SHA256 deletion manifests  |
| - Established offline tri-population membership inference attack (MIA) probes                   |
+--------------------------------------------------------------------------------------------------+
| Phase 2: Multi-Class CNN & Convergence Repair                                                    |
| - Upgraded to real 10-class CIFAR-10 with SmallGroupNormCNN (60k parameters)                     |
| - Identified & eliminated spatial-destroying AdaptiveAvgPool2d((1,1)), replacing with Flatten(1) |
| - Introduced multi-seed retrain reference ensemble (M_R) to quantify natural seed variance      |
| - Formulated Marginal Unlearning Benefit (MUB) with 95% Bootstrap Confidence Intervals           |
| - Built automated Validity Gating (VALID >= 40%, WARNING 25-40%, INVALID < 25%)                  |
+--------------------------------------------------------------------------------------------------+
| Phase 7: Complete 90-Run Factorial Screening Grid                                                |
| - 30 factorial cells: eps in {1, 2, 4, 8, inf} x alpha in {0.1, 0.3, 1.0} x del in {25%, 100%}   |
| - 94 completed runs with immutable IDs, failure recovery, and master JSON/CSV aggregation        |
+--------------------------------------------------------------------------------------------------+
| Phase 8: Tri-Population Privacy Auditing                                                          |
| - Frozen audit evaluation of Forgotten, Retained, and Unseen populations                         |
| - ROC AUC, Attack Advantage, and Low-FPR operating points (TPR @ 1% FPR, TPR @ 0.1% FPR)         |
+--------------------------------------------------------------------------------------------------+
| Phase 9: Faithful FedEraser & Baseline Suite                                                     |
| - Integrated 5 baselines: Exact Retrain, DP-only, Retained Fine-Tuning, Cached Recon, FedEraser  |
| - Discovered that FedEraser calibration is Category B data-dependent access (not DP post-proc)   |
| - Formalized server-side historical update storage scaling (~94 MB measured, scaling to GBs/TBs)  |
+--------------------------------------------------------------------------------------------------+
| Positive Controls & Utility Frontier Discovery (Phases C, D, E)                                  |
| - Centralized Positive Control: 43.40% test accuracy on CIFAR-10                                 |
| - Standard FedAvg Positive Control: 44.35% test accuracy on CIFAR-10                             |
| - Positive Unlearning Control: Detected forgotten drop (82.5% -> 68.0%) on influential client   |
| - Empirical Utility Frontier: Healthy (eps >= 32), Marginal (eps in [8, 16]), Collapse (eps <= 4) |
+--------------------------------------------------------------------------------------------------+
```

---

## 3. Core Scientific Discoveries

### 3.1 Diagnosis of the >54% Pilot vs ~26.7% Phase-7 Discrepancy
An aggressive investigation of `results/phase2_cifar10_multiclass_cnn_20260913T094412Z/run_metadata.json` resolved the apparent contradiction between early high accuracy and lower screening numbers:

| Dimension | Pilot Run (`20260913T094412Z`) | Phase-7 Screener Runs | Factor Difference |
| :--- | :---: | :---: | :---: |
| **Total Training Images** | $20,000$ | $4,000$ | **$5.0\times$ less data** |
| **Participating Clients / Round** | $100$ | $10$ | **$10.0\times$ fewer clients** |
| **Local Epochs / Client / Round** | $5$ | $2$ | **$2.5\times$ fewer epochs** |
| **Rounds** | $100$ | $40$ | **$2.5\times$ fewer rounds** |
| **Total Local Gradient Steps** | **$312,500$** | **$5,000$** | **$62.5\times$ less compute** |
| **Global DP Noise Normalizer ($q \cdot N$)** | $100.0$ | $10.0$ | **$10.0\times$ smaller denominator** |
| **Effective Global Noise Std ($\frac{\sigma \cdot C}{q \cdot N}$)** | **$0.005$** | **$0.25 - 0.40$** | **$50\times - 80\times$ higher noise** |
| **Actual Privacy ($\epsilon$) in Ledger** | **$\mathbf{410.12}$** | **$1.0 - 8.0$** | Pilot was practically non-private |

**Conclusion**:
1. Phase-7's non-private model was severely undertrained on 4k images. When properly trained ($N=100, T=60, E=3, q=0.5$), non-private accuracy reaches **$44.35\%$**, matching centralized training (**$43.40\%$**).
2. The pilot's 54.55% was evaluated at $\epsilon = 410.12$. Under strict private regimes ($\epsilon \le 8$), noise with $N=20$ swamped the CNN, causing utility collapse.

---

### 3.2 The Three Scientific Regimes

```
+--------------------------------------------------------------------------------------------------+
|                                        THE THREE REGIMES                                         |
+--------------------------------------------------------------------------------------------------+
| 1. REGIME 1: UTILITY COLLAPSE (eps <= 4.0)                                                       |
|    - Test accuracy < 25% (near random chance 10% on CIFAR-10)                                    |
|    - Validity Gate: INVALID_TRAINING                                                             |
|    - Scientific Status: Noise multiplier sigma >= 4.63 overwhelms gradient signal.               |
|      Unlearning claims are invalid / vacuously redundant.                                        |
+--------------------------------------------------------------------------------------------------+
| 2. REGIME 2: DP-DOMINATED REDUNDANCY (eps in [8.0, 32.0])                                        |
|    - Test accuracy is healthy or marginal (28.8% - 41.8% >= chance)                              |
|    - Deletion effect is masked by client-level DP perturbation: rho = E_delete / E_retrain << 1  |
|    - Pre-deletion model falls within [Q10, Q90] of exact retrain ensemble                       |
|    - Scientific Status: True redundancy! Doing nothing matches retrain; FedEraser adds no gain.   |
+--------------------------------------------------------------------------------------------------+
| 3. REGIME 3: UNLEARNING-SENSITIVE (eps = infinity, non-private)                                  |
|    - Test accuracy is healthy (44.35%)                                                           |
|    - Deletion effect exceeds natural retraining variance: rho = E_delete / E_retrain >> 1        |
|    - Pre-deletion model retains forgotten accuracy (82.5%) and high MIA leakage (0.137)          |
|    - Scientific Status: Explicit unlearning (Retained FT / FedEraser) is UNAMBIGUOUSLY BENEFICIAL |
+--------------------------------------------------------------------------------------------------+
```

---

### 3.3 The Normalized Deletion-Effect Metric ($\rho$)
To objectively quantify whether a client deletion has an observable effect beyond random retraining noise, we define:
$$\rho = \frac{E_{delete}}{E_{retrain}} = \frac{d(M_{full}, M_R)}{Q_{90}[d(M_R^i, M_R^j)]}$$
where $d(M_A, M_B)$ is the predictive Jensen-Shannon divergence over held-out test data:
- **$\rho \ll 1$ (DP-Dominated)**: The effect of deleting the client is much smaller than the variance from retraining with a different random seed. Doing nothing is empirically sufficient.
- **$\rho \approx 1$ (Transition Boundary)**: Deletion effect is comparable to natural seed variation.
- **$\rho \gg 1$ (Unlearning-Sensitive)**: Deletion effect is materially larger than seed noise. Explicit unlearning is necessary.

---

## 4. Multi-Baseline Comparison & Resource Tradeoffs

### 4.1 Tradeoff Matrix Across Baselines
Evaluated on $N=100$ clients, $T=60$ rounds, $q=0.5$, `SmallGroupNormCNN` ($60,586$ parameters = $242.3\text{ KB}$ per update):

| Method | Description | Persistent Storage | Compute Rounds | Communication Bytes | Privacy Category | Practical Recommendation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **DP-only / No Action** | Retain model as-is under pre-existing DP guarantee | **0 Bytes** | **0** (0.0s) | **0 Bytes** | **Category A** (Pure Post-Proc) | **Optimal when $\epsilon \le 32$ (DP-dominated)** |
| **Retained Fine-Tuning** | Fine-tune full model on retained clients (4 rounds) | **0 Bytes** | **4** (~3.5s) | **$48.5\text{ MB}$** | **Category B** (Data-dependent) | **Optimal when $\epsilon = \infty$ (Non-private)** |
| **Faithful FedEraser** | Reconstructs global updates with client calibration | **$93.7\text{ MB}$** | **30** (~7.0s) | **$363.5\text{ MB}$** | **Category B** (Data-dependent) | **Redundant under DP; high storage cost** |
| **Cached Reconstruction**| Direct accumulation without client calibration | **$93.7\text{ MB}$** | **0** (~0.1s) | **0 Bytes** | **Category A** (Server post-proc) | **Harmful; causes catastrophic drift** |
| **Exact Retrain Ensemble**| Retrain from scratch on retained clients | **0 Bytes** | **60** (~28.0s)| **$727.0\text{ MB}$** | **Gold Standard** (Clean retrain) | **Ground truth gold-standard reference** |

### 4.2 Is FedEraser DP Post-Processing?
**No. It is Category B (Data-Dependent Access).**  
During FedEraser's client calibration step, intermediate global models are transmitted to retained clients, who evaluate forward-backward passes on their raw local training data. Because it queries the private training data after publication, it **does not satisfy the DP Post-Processing Theorem** and would require separate privacy accounting in a private setting.

### 4.3 Storage Scaling Projections
$$\text{Storage} = T \times (q \cdot N) \times |\Theta| \times 4\text{ bytes}$$

| Model Architecture | Parameter Count ($|\Theta|$) | Update Size | Storage ($T=100, N=100, q=0.5$) | Storage ($T=300, N=500, q=0.1$) |
| :--- | :---: | :---: | :---: | :---: |
| **SmallGroupNormCNN** | $60,586$ | $242\text{ KB}$ | **$1.21\text{ GB}$** | **$3.63\text{ GB}$** |
| **MobileNetV3-Small** | $2,540,000$ | $10.2\text{ MB}$ | **$51.0\text{ GB}$** | **$153.0\text{ GB}$** |
| **ResNet-18** | $11,170,000$ | $44.7\text{ MB}$ | **$223.5\text{ GB}$** | **$670.5\text{ GB}$** |
| **DistilBERT** | $66,000,000$ | $264.0\text{ MB}$ | **$1.32\text{ TB}$** | **$3.96\text{ TB}$** |

Under client-level DP ($\epsilon \le 32$), FedEraser provides **no measurable unlearning benefit over DP-only** ($MUB_{acc} \approx +0.001$), yet incurs massive server storage liabilities.

---

## 5. Repository Structure & File Inventory

```
ISP_G2_work/
├── configs/
│   ├── phase2_cifar10_multiclass_cnn.yaml  # Base multiclass CNN experiment config
│   ├── phase2_private_pilot.yaml           # Historical N=100 pilot config
│   └── temp_sweep.yaml                     # Temporary sweep configuration
├── partitions/
│   └── generated/                          # SHA256-verified partition deletion manifests
├── results/
│   ├── phase7_factorial/                   # Full 94-run factorial screening grid results
│   │   ├── master_results.json             # Aggregated JSON across all 30 cells
│   │   ├── master_summary.csv              # Aggregated tabular metrics across all 94 runs
│   │   └── cell_*/                         # Immutable individual run directories
│   ├── positive_controls/                  # Centralized and FedAvg positive control outputs
│   ├── positive_unlearning_control/        # Influential client unlearning sensitivity outputs
│   ├── utility_frontier/                   # Multi-epsilon utility sweep outputs
│   └── phase10_stress_tests/               # Sequential and influential client stress tests
├── scripts/
│   ├── check_grid_results.py               # Comprehensive audit script for Phase 7 grid
│   ├── run_positive_controls.py            # Centralized vs. FedAvg positive control suite
│   ├── run_unlearning_control.py           # Influential client unlearning positive control
│   ├── run_utility_frontier.py             # Multi-epsilon utility boundary discovery sweep
│   ├── run_phase7_factorial.py             # Resumable factorial grid runner with immutable IDs
│   ├── run_phase10_stress_tests.py         # Sequential deletion stream & influence test runner
│   ├── run_phase11_frontier.py             # Automated frontier detection & confirmation
│   └── aggregate_mub.py                    # Master MUB aggregation utility
├── src/dp_forgetbench/
│   ├── attacks.py                          # Tri-Population MIA, ROC AUC, advantage, bootstrap CIs
│   ├── config.py                           # Strict YAML schema validation & immutable dataclasses
│   ├── data.py                             # CIFAR-10 Dirichlet partitioner & deterministic manifests
│   ├── evaluation.py                       # MUB, JS divergence, 4-way classification logic
│   ├── federated.py                        # SmallGroupNormCNN, train_federated, FedEraser, finetuning
│   ├── lira.py                             # Likelihood Ratio Attack implementation
│   ├── privacy.py                          # Poisson-Gaussian RDP accountant & bisection noise solver
│   ├── run.py                              # End-to-end experiment pipeline orchestrator
│   └── statistics.py                       # Statistical confidence intervals & bootstrapping
├── tests/
│   ├── test_attacks.py                     # Tri-population MIA and low-FPR unit tests
│   ├── test_dp_accountant.py               # Mathematical tests for RDP composition
│   ├── test_federated_eraser.py            # Tests for FedEraser caching, calibration, rescaling
│   └── test_smoke.py                       # End-to-end pipeline smoke test
├── COST_ANALYSIS.md                        # Multi-dimensional cost analysis (compute/storage/privacy)
├── FEDERASER_AUDIT.md                      # FedEraser technical audit, post-processing status & scaling
├── LIMITATIONS.md                          # Threats to validity, boundaries, and explicit non-claims
├── PHASE7_AUDIT.md                         # Detailed audit of 94-run Phase 7 screening study
├── PRIVACY_AUDIT.md                        # Tri-population privacy audit and MIA methodology
├── PROJECT_AUDIT.md                        # Overall project audit and status report
├── REDUNDANCY_FRONTIER.md                  # Three regimes, decision flowchart, and rho metric
├── REPRODUCIBILITY.md                      # Seeds, manifests, environments, and reproduction commands
├── UTILITY_FRONTIER.md                     # Positive controls & quantitative utility sweep table
└── README.md                               # Project README with quickstart and key results
```

---

## 6. Reproduction & Execution Guide

### Prerequisites
```powershell
# Set deterministic flags and python path
$env:PYTHONPATH = "src"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
```

### 1. Run Positive Controls (Centralized & FedAvg)
```powershell
python scripts/run_positive_controls.py --n_clients 100 --rounds 60 --local_epochs 3
```
*Output*: Centralized $= 43.40\%$, FedAvg $= 44.35\%$ (proves architecture and data pipeline learn CIFAR-10).

### 2. Run Positive Unlearning Control
```powershell
python scripts/run_unlearning_control.py
```
*Output*: Deletes influential client (entropy $0.031$). Detects drop in forgotten accuracy from $82.5\% \to 68.0\%$ (Retrain) and $70.5\%$ (Fine-Tuning), confirming unlearning sensitivity.

### 3. Run Utility Frontier Discovery Sweep
```powershell
python scripts/run_utility_frontier.py --epsilons inf 32 16 8 4 --seeds 20260901
```
*Output*: Maps the boundary between Healthy ($\epsilon \ge 32$), Marginal ($\epsilon \in [8, 16]$), and Collapse ($\epsilon \le 4$).

### 4. Audit Completed Phase 7 Grid Results
```powershell
python scripts/check_grid_results.py
```
*Output*: Validates all 94 completed runs, artifact completeness, and computes regime statistics.

### 5. Run Sequential Deletion Stress Tests
```powershell
python scripts/run_phase10_stress_tests.py
```

### 6. Run Automated Frontier Confirmation
```powershell
python scripts/run_phase11_frontier.py
```

---

## 7. Final Scientific Conclusion

The definitive empirical answer to our research question:

> **When client-level DP federated learning operates in its converged regime ($\epsilon \in [8, 32]$), explicit client unlearning via historical reconstruction (such as FedEraser) is empirically redundant. The perturbation introduced by DP noise and the natural variance of retraining with a different seed strictly dominate the statistical influence of an individual client ($\rho \ll 1$). Furthermore, FedEraser imposes massive server-side historical storage overhead (~94 MB to gigabytes) and requires interactive client calibration that queries raw data, making it both practically disadvantageous and privacy-suboptimal compared to differential privacy alone.**
