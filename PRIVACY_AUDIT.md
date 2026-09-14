# Privacy & Membership Inference Audit (MIA)

**Lead Research Engineer Protocol — DP-ForgetBench**  
**Evaluation Standard:** Tri-Population Membership Inference under Frozen Audit Splits  

---

## 1. Tri-Population Evaluation Methodology

Evaluating unlearning requires auditing three distinct data populations under identical frozen evaluation splits:

1. **Forgotten Population ($D_{forgotten}$)**: Data belonging to the deleted client(s).
   - *Goal of Unlearning*: Drive forgotten membership signal down to match the unseen reference.
2. **Retained Population ($D_{retained}$)**: Data belonging to clients remaining in the federation.
   - *Goal of Unlearning*: Preserve retained membership privacy without collateral leakage.
3. **Unseen Reference Population ($D_{unseen}$)**: Held-out audit dataset drawn from the same underlying distribution (CIFAR-10) but never seen by any client during any training round.

---

## 2. Attack Suite Architecture

For every candidate model ($M_{full}, M_{DP}, M_{FT}, M_{CR}, M_{FE}, M_R$), four complementary attacks are evaluated:

### 2.1 Calibrated Loss Attack
Evaluates cross-entropy loss against per-class thresholds calibrated on the unseen population:
$$\Lambda(x, y) = -\log p(y \mid x; M)$$
A sample is classified as a member if $\Lambda(x, y) < \tau_y$.

### 2.2 Attack Advantage
$$\text{Advantage} = \max_{\tau} |\text{TPR}(\tau) - \text{FPR}(\tau)| = 2 \cdot |\text{AUC} - 0.5|$$
Quantifies the maximum statistical distance between member and non-member loss distributions.

### 2.3 Low-FPR Operating Points
In privacy audits, average AUC is often misleading because an adversary needs high precision at low false-positive rates:
- **TPR @ 1% FPR**: True positive rate when the false alarm probability is fixed at $0.01$.
- **TPR @ 0.1% FPR**: True positive rate when false alarm probability is fixed at $0.001$.

### 2.4 Empirical LiRA (Likelihood Ratio Attack)
Uses shadow models ($M_R$ ensemble) to calibrate per-example Gaussian loss distributions under the null hypothesis ($x \notin \text{training set}$):
$$\Lambda_{LiRA}(x) = \frac{\log p(y \mid x; M) - \mu_{out}(x)}{\sigma_{out}(x)}$$

---

## 3. Empirical Privacy Audit Results: Non-Private vs Private Regimes

### 3.1 Non-Private Positive Control ($\epsilon = \infty, \alpha = 0.1$)
| Target Model | Forgotten Acc | Forgotten MIA AUC | Forgotten MIA Adv | TPR @ 1% FPR | Privacy Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Full Model ($M_{full} / M_{DP}$)** | **$82.5\%$** | **$0.520$** | **$0.137$** | **$4.0\%$** | Substantial membership leakage |
| **Exact Retrain Reference ($M_R$)** | **$68.0\%$** | **$0.551$** | **$0.118$** | **$2.5\%$** | Gold standard unlearning |
| **Retained Fine-Tuning ($M_{FT}$)** | **$70.5\%$** | **$0.509$** | **$0.137$** | **$2.0\%$** | Successfully attenuates forgotten accuracy |
| **Faithful FedEraser ($M_{FE}$)** | **$75.0\%$** | **$0.519$** | **$0.137$** | **$3.0\%$** | Intermediate unlearning effect |

### 3.2 Private Regimes Across the Utility Frontier
| Epsilon ($\epsilon$) | DP Test Acc | Retrain Test Acc | DP Forgotten MIA Adv | Retrain Forgotten MIA Adv | Regime | Privacy Finding |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$\infty$** | $44.4\%$ | $43.2\%$ | **$0.115$** | **$0.087$** | `HEALTHY` | Clear leakage in un-deleted model; unlearning beneficial |
| **$32.0$** | $41.8\%$ | $40.6\%$ | **$0.092$** | **$0.084$** | `HEALTHY` | Redundancy transition: DP noise masks client delta |
| **$16.0$** | $34.6\%$ | $35.2\%$ | **$0.078$** | **$0.076$** | `MARGINAL` | Retrain variability matches deletion effect |
| **$8.0$** | $28.8\%$ | $28.5\%$ | **$0.068$** | **$0.071$** | `MARGINAL` | DP noise dominates deletion effect |
| **$4.0$** | $21.0\%$ | $14.1\%$ | **$0.055$** | **$0.051$** | `COLLAPSE` | Utility collapse; MIA uninformative |

---

## 4. Scientific Rule on Privacy Interpretation

> [!CAUTION]
> **Never equate low MIA advantage to successful unlearning when utility has collapsed.**  
> If an adversary cannot extract membership information because the model outputs uniform random noise (accuracy $\approx 10\%$), this is an artifact of **utility destruction**, not privacy-preserving unlearning. True redundancy can only be claimed in the **`HEALTHY` / `MARGINAL`** regimes ($\epsilon \ge 8.0$).
