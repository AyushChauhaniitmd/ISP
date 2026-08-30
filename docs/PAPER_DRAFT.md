# Paper draft scaffold — do not submit with placeholder sections

## Working title

**When Is Explicit Client Unlearning Redundant? A Privacy-Aligned Audit of Differentially Private Federated Learning**

## Abstract

Machine unlearning and differential privacy provide different protections, but empirical studies often conflate record- and client-level privacy or compare approximate unlearning to weak baselines. We introduce DP-ForgetBench, a reproducible evaluation protocol for client deletion in federated learning. The benchmark aligns the protected unit with central client-level DP, compares unchanged DP models and explicit unlearning methods to independent retraining targets, records all privacy releases, and evaluates utility, functional alignment, request lifecycle cost, forgotten-set privacy, and retained-set privacy. **Replace this paragraph with verified public-benchmark results only.**

## Claimed contributions (conditional until evidenced)

1. A privacy-scope-aligned benchmark for client deletion under central DP.
2. A retrain-variability-aware redundancy analysis rather than a single-target comparison.
3. A multi-population privacy audit covering forgotten, retained, and unseen data.
4. Sequential and partial request evaluation with lifecycle accounting.

## Results table template

| Dataset | Client DP `(ε,δ)` | Heterogeneity | Request | Method | Retain utility | JS to retrain | Forget MIA | Retain MIA | Request cost | n seeds |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Placeholder |  |  |  |  |  |  |  |  |  |  |

## Required evidence before submission

- Public CIFAR-10/CIFAR-100 and natural-FL/FEMNIST results; frozen split/deletion manifests.
- Exact retraining, independent target retraining, DP-only, and at least two reproduced published FU baselines.
- Calibrated LiRA/A-LiRA and faithful retained-set tri-class attack evaluation.
- Five exploratory seeds and ten confirmation seeds at selected boundary cells.
- Pre-registered multi-endpoint equivalence/non-inferiority analysis and full raw artifacts.
- Fresh literature and baseline search immediately before submission.

## Forbidden wording until the evidence is complete

- “SOTA”, “certified”, “DP performs deletion”, “universal boundary”, or “privacy guarantee after retained-data fine-tuning”.

