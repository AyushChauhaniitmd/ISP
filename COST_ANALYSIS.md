# Multi-Dimensional Cost & Resource Analysis

**Lead Research Engineer Report — DP-ForgetBench**  
**Evaluation Scope:** Compute, Communication, Persistent Storage, and Privacy Budget  

---

## 1. Multi-Dimensional Tradeoff Framework

An unlearning algorithm cannot be evaluated on utility and privacy alone; practical viability requires analyzing four orthogonal resource dimensions:

1. **Persistent Server Storage**: Historical model updates stored indefinitely on the server.
2. **Server & Client Compute**: CPU/GPU training hours for unlearning.
3. **Network Communication**: Uplink/downlink bytes transferred between server and clients.
4. **Differential Privacy Budget ($\epsilon_{extra}$)**: Extra privacy budget consumed by unlearning operations.

---

## 2. Resource Tradeoff Matrix Across All 5 Baselines

Evaluated on $N=100$ clients, $T=60$ rounds, $q=0.5$ (50 clients/round), `SmallGroupNormCNN` ($60,586$ parameters = $242.3\text{ KB}$ per update):

| Method | Persistent Server Storage | Unlearning Compute Rounds | Client Communication Bytes | Privacy Category | Practical Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **DP-only / No Action ($M_{DP}$)** | **0 Bytes** | **0 Rounds** (0.0s) | **0 Bytes** | **Category A** (Pure Post-Processing) | **Optimal when $\rho \ll 1$ (DP-dominated)** |
| **Retained Fine-Tuning ($M_{FT}$)** | **0 Bytes** | **4 Rounds** (~3.5s) | **$48.5\text{ MB}$** | **Category B** (Data-dependent on retained) | **Optimal when $\epsilon = \infty$ (Non-private)** |
| **Faithful FedEraser ($M_{FE}$)** | **$93.7\text{ MB}$** | **30 Calib Rounds** (~7.0s) | **$363.5\text{ MB}$** | **Category B** (Data-dependent on retained) | **Redundant under DP; high storage cost** |
| **Cached Reconstruction ($M_{CR}$)** | **$93.7\text{ MB}$** | **0 Rounds** (~0.1s) | **0 Bytes** | **Category A** (Server post-processing) | **Harmful; causes catastrophic drift** |
| **Exact Retrain Reference ($M_R$)** | **0 Bytes** | **60 Rounds** (~28.0s) | **$727.0\text{ MB}$** | **Gold Standard** (Clean retrain) | **Gold-standard ground truth reference** |

---

## 3. Storage Scaling Projections

$$\text{Storage (Bytes)} = T \times (q \cdot N) \times |\Theta| \times 4\text{ bytes}$$

As models and federations scale to realistic production sizes, the storage required by FedEraser and Cached Reconstruction becomes prohibitive:

```
Storage Scaling vs Model Size (N=500, T=200, q=0.1 -> 10,000 updates stored)
-----------------------------------------------------------------------------
SmallGroupNormCNN (60k params):       2.42 GB
MobileNetV3-Small (2.5M params):    100.80 GB
ResNet-18 (11.2M params):           448.00 GB
DistilBERT (66M params):              2.64 TB
```

### Key Takeaway:
Under client-level DP ($\epsilon \le 32$), the unlearning benefit of FedEraser over DP-only is statistically negligible ($MUB_{acc} \approx 0.001$), yet FedEraser imposes **gigabytes to terabytes of server-side historical storage liability**. DP-only completely eliminates this overhead while providing identical empirical unlearning performance.
