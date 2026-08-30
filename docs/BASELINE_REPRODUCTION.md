# Baseline reproduction contract

No method is added to a paper table until it meets this contract.

1. Pin upstream repository URL, commit hash, license, and environment.
2. Reproduce one claimed result on the authors' stated dataset before modifying code.
3. Implement it behind the common interface: full model + request + retained data/update history + declared threat model → unlearned model + cost ledger.
4. Give every method equal data partitions, client schedule where possible, model, stopping rule, validation set, and tuning budget.
5. Include total lifecycle cost: original training, retained checkpoints/update storage, request latency, communication, GPU/CPU time, and memory.
6. Record whether it accesses raw retained data, private updates, public data, or a clean reference model.
7. Compare to exact retraining and independent retraining variability; never only to a weak no-action baseline.

## Baseline order

| Order | Method family | Current state |
|---:|---|---|
| 1 | Exact retraining | Implemented |
| 2 | DP-only/no action | Implemented |
| 3 | Retained-data fine-tuning | Implemented, clearly a baseline only |
| 4 | Cached-update direct accumulation (FedEraser-family control) | Implemented; explicitly not claimed as a full FedEraser reproduction |
| 5 | FedRecover-style DP-aware FU | Pending upstream reproduction |
| 6 | SISA/FedShard design-for-unlearning | Pending method and storage implementation |
| 7 | Starfish/certified client removal | Pending; requires faithful 2PC/system assumptions |
