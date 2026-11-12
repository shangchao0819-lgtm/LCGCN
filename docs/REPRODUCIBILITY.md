# Reproducibility and scope

The runnable main plan covers LCGCN, NativeLCGCN, and FastLCGCN: 450 final-200 fits on five datasets, ten supplied splits, and three initialization seeds. It does not provide the six external baselines or complete inductive, filtering, ablation, timing, global-spectrum, paired-inference, or bootstrap experiment suites. Optional internal functions alone do not constitute those complete workflows.

The 2026-11-12 alignment audit compared the package with the manuscript and frozen experimental records. It verified all 450 effective main recipes, all 50 supplied splits and input hashes, local graph equations, hidden-state reconstruction, detached graph construction, and evaluation behavior. Recalculation from the 450 historical records matched the 15 own-model paper-table mean/SD cells. This was historical-record verification, not 450 new training runs.

The audit also compared the frozen and packaged models over five Adam steps for each of three models and three seeds on the CPU. Parameters, gradients, outputs, and final graph hashes matched exactly in those nine small cases.

Seventeen new 200-epoch fits passed checkpoint reload checks: fifteen CUDA model/dataset combinations at split 1/seed 1, Pubmed LCGCN split 4/seed 17 on CUDA, and Citeseer LCGCN split 1/seed 8 on CPU. Fourteen had the same test accuracy as their historical records. Three differed:

| Model | Dataset | Device | New minus historical accuracy (pp) |
|---|---|---|---:|
| LCGCN | DBLP | CUDA | 0.02527678 |
| FastLCGCN | DBLP | CUDA | 0.01011071 |
| LCGCN | Citeseer | CPU | 0.28355388 |

These are fresh-training differences, not failures to reload the newly saved checkpoints. Their complete numerical causes have not been isolated. Dynamic sorting and floating-point execution can affect reproducibility; architectural agreement does not imply identical numerical trajectories. The manuscript also retains historical dynamic reconstruction exceptions. No historical score or checkpoint was replaced based on test accuracy.

Public-release preparation changes documentation, licensing, figure assets, and missing-data guidance, not the audited architecture, numerical recipes, or paper aggregation formula. Third-party input files are excluded from the public repository. Neither this preparation nor the preceding audit reran all 450 fits. The Windows environment was validated locally; consult GitHub Actions for the current Windows/Linux test status. Full static type coverage and cross-device bitwise identity are not claimed.
