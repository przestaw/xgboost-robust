# Mirror of RobustTrees

This repository mirrors [chenhongge/RobustTrees](https://github.com/chenhongge/RobustTrees),
the reference implementation of

> Hongge Chen, Huan Zhang, Duane Boning, Cho-Jui Hsieh,
> "Robust Decision Trees Against Adversarial Examples", ICML 2019,
> [arXiv:1902.10660](https://arxiv.org/abs/1902.10660).

RobustTrees is a fork of XGBoost 0.72 that adds `tree_method=robust_exact`
(updater `robust_grow_colmaker`, source `src/tree/updater_robust_colmaker.cc`)
and the training parameters `robust_eps` and `robust_training_verbose`.

## Branches

| Branch | Content |
| --- | --- |
| `master` | Unchanged copy of upstream RobustTrees `master` (`ed28228`). |
| `mirror-notes` | `master` + this file only. |
| `robust-xgboost-1.7` | Port of the robust split finder onto XGBoost v1.7.6, see `ROBUST_PORT.md` on that branch. |

All code is licensed under the Apache License 2.0 (see `LICENSE`); copyright
remains with the XGBoost contributors and the RobustTrees authors.
