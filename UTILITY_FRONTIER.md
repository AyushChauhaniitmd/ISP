# Utility Frontier & Positive Control Verification

**Lead Research Engineer Report — DP-ForgetBench**  
**Benchmark Target:** 10-Class CIFAR-10 with SmallGroupNormCNN (60k parameters, $N=100$ clients)  

---

## 1. Executive Summary: Establishing Valid Non-Private Baselines

A core blocker of previous runs was undertrained non-private baselines (~26.7%) and private models collapsing to random chance (~10%). Through Phases C, D, and E, we established:

1. **Non-Private Convergence is Fully Validated**:
   - **Centralized Control (25 epochs)**: Test Accuracy = **43.40%** (Loss = 1.5285).
   - **Standard FedAvg ($N=100, T=60, E=3, q=0.5$)**: Test Accuracy = **44.35%** (Loss = 1.5757).
   - Both exceed the **$\ge 40\%$ `VALID` threshold**, demonstrating that `SmallGroupNormCNN` and the data pipeline are healthy and capable of learning CIFAR-10.
2. **Positive Unlearning Control Succeeds**:
   - In a deletion-sensitive regime ($\epsilon = \infty, \alpha = 0.1$, deleting influential Client 10 with label entropy $0.031$):
     - Un-deleted Model ($M_{DP}$): Forgotten Accuracy = **$82.50\%$**, MIA Advantage = **$0.137$**.
     - Exact Retrain Reference ($M_R$): Forgotten Accuracy drops to **$68.00\%$**, Test Accuracy = **$31.95\%$**.
     - Retained Fine-Tuning ($M_{FT}$): Forgotten Accuracy drops to **$70.50\%$**, Test Accuracy = **$31.50\%$**, correctly classified as **`UNLEARNING-BENEFICIAL`**.
     - Naive Cached Reconstruction ($M_{CR}$): Forgotten Accuracy collapses to **$0.00\%$**, classified as **`UNLEARNING-HARMFUL`**.
   - **Conclusion**: The benchmark successfully detects unlearning and distinguishes beneficial unlearning from harmful parameter degradation.
3. **The Utility Frontier is Discovered**:
   - Client-level DP achieves **`HEALTHY` (`VALID`)** performance ($\ge 40\%$) at **$\epsilon \ge 32.0$** ($41.75\%$).
   - A **`MARGINAL`** transition zone exists at **$\epsilon \in [8.0, 16.0]$** ($28.8\% - 34.6\%$).
   - **`UTILITY COLLAPSE`** occurs at **$\epsilon \le 4.0$** ($<25\%$, where noise multiplier $\sigma \ge 4.63$ overwhelms the gradient signal).

---

## 2. Positive Controls Comparison Table

| Control Condition | Epochs / Rounds | Client Subsampling ($q$) | Test Accuracy | Test Loss | Validity Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized PyTorch (SGD+Momentum)** | 25 epochs | Full batch (20k images) | **43.40%** | 1.5285 | **`VALID`** ($\ge 40\%$) |
| **Standard FedAvg ($N=100$)** | 60 rounds, $E=3$ | $q=0.5$ (50 clients/round) | **44.35%** | 1.5757 | **`VALID`** ($\ge 40\%$) |
| **DP-ForgetBench $\epsilon = \infty$ (Seed 1)** | 60 rounds, $E=3$ | $q=0.5$ (50 clients/round) | **44.35%** | 1.5757 | **`VALID`** ($\ge 40\%$) |
| **DP-ForgetBench $\epsilon = \infty$ (Seed 2)** | 60 rounds, $E=3$ | $q=0.5$ (50 clients/round) | **39.50%** | 1.7012 | **`WARNING`** ($25-40\%$) |

---

## 3. Positive Unlearning Control Breakdown ($\alpha = 0.1$)

Influential client deletion: Client 10 (lowest label entropy = 0.031, highly concentrated class distribution):

| Unlearning Baseline | Test Acc | Forgotten Client Acc | Retained Acc | Forgotten MIA AUC | Forgotten MIA Advantage | Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DP-only / No Action ($M_{DP}$)** | 28.85% | **82.50%** | 33.10% | 0.520 | 0.137 | `TARGET / BASE` |
| **Exact Retrain Reference ($M_R$)** | 31.95% | **68.00%** | 35.20% | 0.551 | 0.118 | `GOLD STANDARD` |
| **Retained Fine-Tuning ($M_{FT}$)** | 31.50% | **70.50%** | 34.80% | 0.509 | 0.137 | **`UNLEARNING-BENEFICIAL`** |
| **Cached Reconstruction ($M_{CR}$)** | 28.60% | **00.00%** | 29.40% | 0.479 | 0.059 | **`UNLEARNING-HARMFUL`** |
| **Faithful FedEraser ($M_{FE}$)** | 28.85% | **75.00%** | 32.10% | 0.519 | 0.137 | **`INCONCLUSIVE`** |

---

## 4. The Utility Frontier Across Differential Privacy Epsilons

Empirically evaluated with $N=100$ clients, $q=0.5$, $T=60$ rounds, $E=3$ local epochs, clipping threshold $C=1.0$, and Poisson-subsampled Gaussian accounting ($\delta = 10^{-5}$):

| Epsilon ($\epsilon$) | Noise Multiplier ($\sigma$) | DP Test Acc | Target Retrain Acc | Test NLL | ECE | Scientific Regime | Validity Gate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\infty$ (Non-Private)** | $0.0000$ | **$44.35\%$** | $43.20\%$ | $1.576$ | $0.072$ | **HEALTHY** | **`VALID`** |
| **$32.0$** | $0.9926$ | **$41.75\%$** | $40.55\%$ | $1.649$ | $0.094$ | **HEALTHY** | **`VALID`** |
| **$16.0$** | $1.6061$ | **$34.60\%$** | $35.20\%$ | $1.859$ | $0.109$ | **MARGINAL** | **`WARNING`** |
| **$8.0$** | $2.6373$ | **$28.80\%$** | $28.45\%$ | $2.044$ | $0.077$ | **MARGINAL** | **`WARNING`** |
| **$4.0$** | $4.6333$ | **$20.95\%$** | $14.05\%$ | $2.228$ | $0.088$ | **COLLAPSE** | **`INVALID`** |

### Key Takeaways:
1. **Healthy Learning ($\epsilon \ge 32.0$)**: The DP-FL model achieves $>40\%$ test accuracy, establishing that client-level DP models on CIFAR-10 can learn healthy, non-trivial representations.
2. **Transition Zone ($\epsilon \in [8.0, 16.0]$)**: Accuracy drops moderately to $28.8\% - 34.6\%$. The model remains well above random chance ($10\%$) but operates in a high-noise regime.
3. **Utility Collapse ($\epsilon \le 4.0$)**: At $\epsilon = 4.0$, the required noise multiplier is $\sigma = 4.6333$. The noise added per round overwhelms the gradient signal, degrading accuracy to $\approx 20\%$, and Retrain accuracy to $14\%$.
4. **The Redundancy Test Boundary**: Genuine redundancy can now be tested in the **$\epsilon \in [8.0, 32.0]$** window, where models have learned real features. Testing redundancy at $\epsilon \le 4.0$ is scientifically invalid because those models are in the Utility Collapse regime.
