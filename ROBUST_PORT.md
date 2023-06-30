# Robust trees (`tree_method="robust_exact"`) on XGBoost 1.7

This branch is a port of the robust decision tree training of

> Hongge Chen, Huan Zhang, Duane Boning, Cho-Jui Hsieh,
> "Robust Decision Trees Against Adversarial Examples", ICML 2019,
> [arXiv:1902.10660](https://arxiv.org/abs/1902.10660)

from its reference implementation [chenhongge/RobustTrees](https://github.com/chenhongge/RobustTrees)
(a fork of XGBoost 0.72) to the XGBoost 1.7 release line (base: tag `v1.7.6`).
All credit for the algorithm and the original code goes to the authors of the paper
and of RobustTrees; the port only adapts their updater to the XGBoost 1.7 internals.
XGBoost and RobustTrees are licensed under the Apache License 2.0 (see `LICENSE`).

## What was ported

| RobustTrees (0.72) | This branch (1.7.6) |
| --- | --- |
| `src/tree/updater_robust_colmaker.cc`, updater `robust_grow_colmaker` | `src/tree/updater_robust_colmaker.cc`, updater `robust_grow_colmaker` (same name) |
| `tree_method` enum value `robust_exact` = 6 in `src/learner.cc` | `TreeMethod::kRobustExact` = 6, `"robust_exact"`, in `src/gbm/gbtree.h` |
| `robust_exact` -> `updater = "robust_grow_colmaker,prune"` in `src/learner.cc` | same mapping in `GBTree::ConfigureUpdaters` (`src/gbm/gbtree.cc`) |
| `robust_eps` (default 0.3), `robust_training_verbose` (default false) in `TrainParam` | same names and defaults in `RobustColMakerTrainParam` of the robust updater |
| registry link tag in `src/tree/tree_updater.cc` | same |

The robust split search (`EnumerateSplit`) is kept step by step: ascending scan of each
column, threshold candidates `eta = x - eps`, the queues of not yet added and uncertain
points, the four scores (nominal, uncertain points all left, all right, swapped) of
which the minimum is used as the split gain, the final "all statistics" candidate and
moving the chosen threshold to the middle between two feature values.

Adaptations to 1.7 (commented in the source):

- `SplitEvaluator` (0.72) is replaced with `TreeEvaluator`; `ColIterator`/`SparsePage`
  with `SortedCSCPage` batches; `TrainParam::InitAllowUnknown` with `XGBoostParameter`;
  OpenMP loops with `common::ParallelFor`; column sampling with `common::ColumnSampler`.
- The robust parameters live in the updater's own parameter struct, saved in the model
  configuration as `robust_colmaker_train_param`. Passing `robust_eps` with another
  `tree_method` therefore triggers the usual `Parameters: { "robust_eps" } are not used.`
  warning, while `tree_method="robust_exact"` accepts it silently.
- The 0.72 `elastic_net` evaluator never zeroed the gain or weight of a side whose
  hessian is below `min_child_weight`; 1.7 does. The robust search scores partitions
  whose sides are not checked against `min_child_weight`, so the evaluator is given a
  copy of the parameters with `min_child_weight = 0`. The explicit `min_child_weight`
  checks of the search are unchanged.
- Empty columns are skipped (0.72 removed them from the feature set).
- Interaction constraints are honoured as in 1.7 `grow_colmaker`; monotone constraints
  and `max_delta_step` follow the 1.7 evaluator (not tested with `robust_exact`).

Not ported:

- `robust_distcol` (distributed column split maker); 1.7 has no distributed `exact`.
- The `parallel_option != 0` code path (removed from XGBoost before 1.0); the robust
  search always runs the per-feature parallel scan, as with the RobustTrees defaults.
- `splitsample_bynode` (declared in RobustTrees `param.h` but never used).
- GPU support. `robust_exact` is CPU only, single node, in-memory data only.
- The MILP attack and the experiment scripts of RobustTrees.

Random number streams differ between 0.72 and 1.7, so models trained with `subsample`
or `colsample_*` < 1 cannot match RobustTrees models exactly.

## Build (Linux)

Requirements: GCC >= 5 (C++14), CMake >= 3.18, Python 3.10 or 3.11.

```sh
git clone --recursive -b robust-xgboost-1.7 <this repository> xgboost-robust
cd xgboost-robust
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ..
# lib/libxgboost.so now exists; the Python package picks it up instead of rebuilding
python3.11 -m venv .venv && . .venv/bin/activate
pip install "numpy<2" scipy "scikit-learn<1.6"
pip install ./python-package
```

The 1.7 Python package predates NumPy 2 and recent scikit-learn releases: with
NumPy 2.4 some functions (for example `xgb.cv`) fail, and with scikit-learn 1.9 the
scikit-learn estimator checks of `XGBClassifier` fail. Training and prediction with
`XGBClassifier` still worked with those versions, but NumPy 1.26 with scikit-learn
1.5 is the tested combination.

Tests of the port: `python -m pytest tests/python/test_robust_exact.py` (run from the
repository root).

With CMake 4.x add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to the `cmake` call
(the `dmlc-core` submodule declares an old minimum version).

## Usage

```python
from sklearn.datasets import make_moons
from xgboost import XGBClassifier

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)
clf = XGBClassifier(
    tree_method="robust_exact", robust_eps=0.05,
    n_estimators=50, max_depth=3, learning_rate=0.1, random_state=0,
)
clf.fit(X, y)
```

The native interface takes the same keys:
`xgb.train({"tree_method": "robust_exact", "robust_eps": 0.05, ...}, dtrain)`.
`robust_eps` is an absolute distance on the feature values; RobustTrees experiments
used features scaled to [0, 1].

## Reproducing results obtained with RobustTrees 0.72

**Warning: default hyperparameters differ.** In XGBoost 0.72 the scikit-learn
wrapper `XGBClassifier` had its own defaults `max_depth=3`, `learning_rate=0.1`,
`random_state=0`, `n_estimators=100`, `n_jobs=1`. In 1.7 the wrapper defaults are
`None` and the booster defaults apply: `max_depth=6`, `eta=0.3`. A call such as
`XGBClassifier(tree_method="robust_exact", robust_eps=0.05, n_estimators=...)`
therefore trains a different model on 1.7. Pass `max_depth=3, learning_rate=0.1,
random_state=0` explicitly to get the 0.72 behaviour.

Other defaults (for example the evaluation metric) and the model file format also
changed between 0.72 and 1.7; compare `get_xgb_params()` when porting experiments.

Check against RobustTrees: with the native interface and identical parameters
(`max_depth=3`, `eta=0.1`, 50 rounds, `nthread=1`, `base_score=0.5`), RobustTrees
master and this port produced bit-identical predictions and identical tree splits
for `exact` and `robust_exact` with `robust_eps` 0, 0.05 and 0.3 on three small
data sets: `make_moons(200, noise=0.3)`, the same with 15% missing values, and a
6-feature `make_classification` with rounded (tied) values. This is a small check,
not a proof of equivalence on all inputs.

The results of the MSc thesis of Przemysław Stawczyk (2023) were computed with the
original RobustTrees (XGBoost 0.72), not with this port.
