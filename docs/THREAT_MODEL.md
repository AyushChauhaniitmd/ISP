# Threat model and privacy-release contract

## Primary experiment

| Item | Contract |
|---|---|
| Protected unit | One complete federated client dataset |
| Adjacency | Add/remove one client dataset |
| Training mechanism | Poisson client sampling; per-client update L2 clipping; Gaussian noise at the server aggregate |
| Accountant | `dp-accounting` RDP accountant for the Poisson-sampled Gaussian mechanism |
| Released object | Final global model only, unless a revised ledger explicitly composes additional releases |
| Attacker | Black-box observer of released model outputs; separate white-box/update attacks are not reported as black-box evidence |
| Secure aggregation | Not implemented in the simulator. The server-side-noise trust assumption is explicit. |

## What the guarantee does not say

- DP does not delete data, recreate a model as if data were absent, or by itself satisfy a deletion request.
- Example-level DP-SGD does not establish the client-level guarantee required by the primary study.
- A post-hoc method accessing retained raw data is not mere post-processing of the released DP model. It needs a separate access/release ledger.
- A privacy ledger is mechanism-specific. Altering sampling, clipping, rounds, noise, model releases, or attack exposure invalidates reused epsilon values.

## Unlearning-method classifications

| Class | Raw private data after request? | Privacy interpretation |
|---|---:|---|
| DP-only/no action | No | Final-model DP ledger applies. Not a deletion method. |
| Pure post-processing | No | DP post-processing preserves the original guarantee. |
| Retained-data fine-tuning | Yes | Separate exposure regime; never present it as DP post-processing. |
| Historical-update FU | Depends on stored updates | Ledger must state whether updates are private/securely aggregated and what is released. |
| Secure/verified FU | Depends on protocol | Reproduce its cryptographic assumptions before comparing privacy or time. |

