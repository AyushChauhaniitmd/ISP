# FedEraser Audit & Technical Verification

**Lead Research Engineer Report — DP-ForgetBench**  
**Algorithm Reference:** Liu et al., *"FedEraser: Enabling Efficient Client-Level Data Removal for Federated Learning"*, IEEE TIFS 2021.  

---

## 1. Algorithmic Overview & Implementation Integrity

FedEraser is designed to accelerate client removal in Federated Learning without retraining from scratch ($M_R$). Instead of training from round 0 to round $T$, FedEraser uses cached historical client updates and selectively calibrates the reconstructed global trajectory.

### Step-by-Step Architecture:
1. **Historical Update Caching (Training Phase)**:
   - During primary training, the server caches each client $i$'s parameter update $\Delta w_{i, t}$ for every round $t \in [1, T]$ where client $i$ was selected:
     $$\Delta w_{i, t} = w_{i, t} - w_{t-1}$$
   - These updates are saved in `FederatedHistory.rounds[t].deltas_by_client[client_id]`.
2. **Reconstruction Without Forgotten Client**:
   - When client $u$ requests deletion, the server removes $\Delta w_{u, t}$ from the round aggregations:
     $$\Delta W_t^{retained} = \frac{1}{|S_t \setminus \{u\}|} \sum_{i \in S_t \setminus \{u\}} \Delta w_{i, t}$$
3. **Interactive Client Calibration**:
   - Starting from $t=1$, the server reconstructs the model $\bar{w}_t$.
   - Every $k$ rounds (governed by `calibration_ratio` $\rho$, e.g. $\rho=0.5$), the server transmits intermediate model $\bar{w}_{t-1}$ to retained clients.
   - Retained clients compute a **fresh local update** $\Delta \bar{w}_{i, t}$ on their current local dataset $D_i$.
   - The server computes the calibration alignment factor:
     $$s_t = \frac{\langle \Delta \bar{W}_t, \Delta W_t^{retained} \rangle}{\|\Delta \bar{W}_t\| \cdot \|\Delta W_t^{retained}\|}$$
   - The reconstructed update is rescaled and adjusted toward the new gradient direction.

---

## 2. The DP Post-Processing Audit: Category A vs Category B

A critical question in Differential Privacy benchmarking is whether FedEraser satisfies the DP Post-Processing Theorem:

> **Theorem (DP Post-Processing):** If mechanism $\mathcal{M}: \mathcal{D} \to \mathcal{S}$ satisfies $(\epsilon, \delta)$-DP, then for any arbitrary function $f: \mathcal{S} \to \mathcal{S}'$, $f \circ \mathcal{M}$ also satisfies $(\epsilon, \delta)$-DP **if and only if $f$ has no independent access to the private dataset $D$**.

### Classification:
- **Category A (Pure Post-Processing)**: Operations that compute solely on the released model parameters $\bar{w}_T$ without touching client devices or raw data.
  - *Examples*: Doing nothing (`dp_only_no_action`), parameter pruning, output quantization.
- **Category B (Data-Dependent Access Requiring Separate Accounting)**: Operations that query raw client datasets after the initial release.
  - *FedEraser Status*: **Unambiguously Category B**.
  - **Reason**: In Step 3 (Client Calibration), the server sends intermediate checkpoints to retained clients, and clients compute forward-backward passes on their raw local training examples.
  - **Privacy Implication**: Every calibration round is an active communication round that queries private training data. In a private deployment, these calibration rounds would consume **additional differential privacy budget** ($\epsilon_{calib}$) unless fresh DP noise is added and accounted for.

---

## 3. Storage & Resource Scaling Analysis

### Theoretical Storage Model:
The server-side storage required by FedEraser scales as:
$$\text{Storage (Bytes)} = T \times (q \cdot N) \times |\Theta| \times 4\text{ bytes}$$
where:
- $T$: Number of federated training rounds
- $N$: Total client population size
- $q$: Client subsampling rate per round
- $|\Theta|$: Number of model parameters (float32 = 4 bytes)

### Empirical Measurements from Phase-7 Full Grid:
- **Model**: `SmallGroupNormCNN` ($|\Theta| = 60,586$ parameters $\approx 242.3\text{ KB}$ per update).
- **Observed Storage**:
  - For $T=40$ rounds, $N=20, q=0.5$:
    $$\text{Expected} \approx 40 \times 10 \times 242.3\text{ KB} \approx 96.9\text{ MB}$$
  - **Measured Mean Storage**: **$93.7\text{ MB}$** (Range: $65.6\text{ MB} - 99.1\text{ MB}$).
- **Measured Calibration Runtime**: Mean = **$6.95\text{ seconds}$** per run.

### Storage Scaling Projection for Production Models:

| Model Architecture | Parameter Count ($|\Theta|$) | Update Size | Storage ($T=100, N=100, q=0.5$) | Storage ($T=300, N=500, q=0.1$) |
| :--- | :---: | :---: | :---: | :---: |
| **SmallGroupNormCNN** | $60,586$ | $242\text{ KB}$ | **$1.21\text{ GB}$** | **$3.63\text{ GB}$** |
| **ResNet-18** | $11,170,000$ | $44.7\text{ MB}$ | **$223.5\text{ GB}$** | **$670.5\text{ GB}$** |
| **MobileNetV3-Small** | $2,540,000$ | $10.2\text{ MB}$ | **$51.0\text{ GB}$** | **$153.0\text{ GB}$** |
| **DistilBERT** | $66,000,000$ | $264.0\text{ MB}$ | **$1.32\text{ TB}$** | **$3.96\text{ TB}$** |

**Conclusion on Storage**: Storing un-aggregated historical client updates across hundreds of rounds is an immense server-side liability—both in physical infrastructure cost and because storing raw historical client updates on the server creates an attractive target for server-compromise attacks.

---

## 4. Empirical Performance: Non-Private vs Private FL

| Evaluation Dimension | Non-Private Regime ($\epsilon = \infty$) | Private Regime ($\epsilon \le 8.0$) |
| :--- | :---: | :---: |
| **Utility Restoration** | Matches/tracks target retrain within $2\%$ | Indistinguishable from DP-only ($MUB_{acc} \approx +0.001$) |
| **MIA Leakage Reduction** | Cuts forgotten MIA advantage from $0.115 \to 0.060$ | No advantage over DP-only ($0.072$ vs $0.093$) |
| **Storage Requirement** | Requires ~94 MB historical cache | Requires ~94 MB historical cache |
| **Privacy Compliance** | Viable heuristic unlearning algorithm | **Redundant**: DP-only matches unlearning without storage |

---

## 5. Scientific Recommendation
1. In non-private federated learning, FedEraser is a viable unlearning baseline that significantly outperforms naive cached reconstruction.
2. In client-level private federated learning ($\epsilon \le 8.0$), FedEraser provides **no measurable unlearning benefit** while incurring severe storage overhead and active client-side privacy exposure.
