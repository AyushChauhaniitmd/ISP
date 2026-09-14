# Limitations & Threats to Scientific Validity

**Lead Research Engineer Document — DP-ForgetBench**  
**Commitment:** Absolute Scientific Transparency  

---

## 1. Scope & Generalization Boundaries

### 1.1 Model & Dataset Boundaries
- The current empirical validation is established on **10-class CIFAR-10** using a compact convolutional neural network with Group Normalization (`SmallGroupNormCNN`, 60,586 parameters).
- While Group Normalization avoids the batch-dependent privacy leakage of Batch Normalization, larger architectures (such as ResNet-18, WideResNet, or Vision Transformers) may exhibit different sensitivity profiles, wider retraining variance, and different loss geometry.

### 1.2 Federation Scale & DP Signal-to-Noise Ratio
- In client-level Differential Privacy, the Gaussian noise added per round is divided by $q \cdot N$ (the expected participating clients per round).
- For small populations ($N=20, q=0.5 \implies 10\text{ clients/round}$), client-level DP at strict budgets ($\epsilon \le 8$) injects noise that overwhelms the gradient signal, causing **Utility Collapse**.
- For larger federations ($N=100, q=0.5 \implies 50\text{ clients/round}$), models sustain healthy learning up to $\epsilon \ge 32.0$ and marginal learning down to $\epsilon = 8.0$.
- Consequently, whether a model collapses or learns under client-level DP is a joint function of **$(\epsilon, N, q, T, |\Theta|)$**, not $\epsilon$ alone.

---

## 2. Adversarial Threat Model & Privacy Auditing

- Our privacy evaluation uses state-of-the-art **Tri-Population Membership Inference Attacks** (calibrated loss probes, shadow retrain references, ROC AUC, and Low-FPR operating points).
- **Limitation**: These attacks represent realistic black-box / gray-box adversaries. We do not claim an empirical impossibility proof against an all-powerful white-box adversary with full trajectory intermediate checkpoint snapshots, except to the extent guaranteed mathematically by the central DP ledger.

---

## 3. Unlearning Baseline Limitations

- **FedEraser**:
  - Requires persistent storage of raw historical client updates on the central server ($93.7\text{ MB}$ for 60k params; gigabytes for ResNet). If the server is compromised, these stored updates compromise client privacy.
  - Requires interactive communication rounds with retained clients after deletion, which is impossible if retained clients are offline or unwilling to participate.
  - Calibration training constitutes data-dependent access (Category B), which incurs additional privacy budget expenditure not accounted for in standard FL unlearning papers.
- **Cached Reconstruction**:
  - Accumulating historical updates without client calibration causes severe catastrophic drift and parameter collapse. It should not be used as an unlearning baseline without calibration.
- **Retained Fine-Tuning**:
  - Highly effective in non-private FL, but requires access to retained clients' raw local data after the unlearning request.

---

## 4. What This Project Does NOT Claim

> [!IMPORTANT]
> 1. We do **NOT** claim: *"DP perfectly deletes data."*
> 2. We do **NOT** claim: *"DP makes unlearning always unnecessary."*
> 3. We do **NOT** claim: *"DP-only is a certified unlearning algorithm."*

### What We DO Claim:
> **Under a formally specified client-level DP contract with verified convergence, doing nothing after a client-deletion request produces a model whose utility and membership risk are statistically and practically indistinguishable from the natural variability inherent in exact retraining.**
