# Reproducibility Guide & Experimental Protocol

**Lead Research Engineer Protocol — DP-ForgetBench**  
**Repository Version:** Phase-2 Research-Grade Revision  

---

## 1. System & Runtime Environment

The entire benchmark is built for deterministic execution and verified on the following hardware/software stack:

- **Operating System:** Windows 11 Pro (10.0.26200)
- **Python Version:** 3.12.4 (64-bit AMD64)
- **PyTorch Version:** 2.5.1+cu121 (CUDA 12.1 enabled)
- **Active GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)
- **Deterministic Workspaces:**
  ```powershell
  $env:PYTHONPATH = "src"
  $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
  ```

---

## 2. Seed Architecture & Manifest Hashing

All stochastic elements in DP-ForgetBench are strictly controlled via explicit pseudorandom generator (PRG) hierarchies:

### 2.1 Independent Seed Allocations
| Component | Base Seed Offset | Purpose |
| :--- | :---: | :--- |
| **Data Partitioning** | `seed` | Dirichlet class split and client assignment |
| **Full Model Training** | `seed` | Parameter initialization and Poisson client sampling |
| **Retrain Ensemble ($M_R^k$)** | `seed + 10,000 + k` | Independent initialization & sampling for retrain references |
| **Retained Fine-Tuning** | `seed + 20,000` | Unlearning initialization and client batch ordering |
| **Privacy Auditing (MIA)** | `seed + 30,000` | Bootstrap subsampling of membership probe populations |

### 2.2 Partition Deletion Manifests
Every deletion request produces a SHA256-verified manifest in `partitions/generated/deletion_{hash}.csv`. The hash is computed over the deterministic tuple:
$$\text{SHA256}(\text{client\_id}, \text{fraction}, \text{partition\_indices})$$
ensuring that all compared baselines ($M_{DP}, M_R, M_U, M_{FE}$) operate on the exact same retained/forgotten data slices.

---

## 3. Run ID & Artifact Immutability

To prevent accidental overwriting of experiment results, every run directory is assigned an immutable deterministic ID:

```
cell_eps{eps}_a{alpha}_del{del}_s{seed}_{cfg_hash[:8]}
```

Example:
`results/phase7_factorial/cell_eps8_a1_del100pct_s20260901_b7c1b0b9/`

Inside each run directory, three immutable artifacts are preserved:
1. `run_metadata.json`: Full configuration, code commit hash, platform specs, and validity gate flag.
2. `privacy_ledger.json`: RDP accountant record, noise multiplier, clipping norm, sampling rate, and exact epsilon.
3. `metrics.json`: Test utility, alignment JS divergence, Tri-Population MIAs, and resource costs across all 5 baselines.

---

## 4. Step-by-Step Reproduction Commands

### Step 1: Run Positive Controls (Centralized & FedAvg)
```powershell
python scripts/run_positive_controls.py --n_clients 100 --rounds 60 --local_epochs 3
```
*Expected Result:* Centralized Acc $\approx 43.4\%$, FedAvg Acc $\approx 44.4\%$ (firmly $\ge 40\%$ `VALID`).

### Step 2: Run Positive Unlearning Control
```powershell
python scripts/run_unlearning_control.py
```
*Expected Result:* Detects forgotten accuracy drop on influential client from $82.5\% \to 68.0\%$ (Retrain) and $70.5\%$ (Fine-Tuning), confirming `UNLEARNING-BENEFICIAL`.

### Step 3: Run Utility Frontier Sweep
```powershell
python scripts/run_utility_frontier.py --epsilons inf 32 16 8 4 --seeds 20260901 20260902
```

### Step 4: Run Complete Factorial Grid
```powershell
python scripts/run_phase7_factorial.py --mode full_grid
```

### Step 5: Audit and Summarize Master Results
```powershell
python scripts/check_grid_results.py
```
