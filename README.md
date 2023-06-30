# Robust trees for XGBoost 1.7 (`tree_method="robust_exact"`)

This repository is an updated implementation, by Przemysław Stawczyk, of
[RobustTrees](https://github.com/chenhongge/RobustTrees) by Hongge Chen, Huan Zhang,
Duane Boning and Cho-Jui Hsieh ("Robust Decision Trees Against Adversarial Examples",
ICML 2019, [arXiv:1902.10660](https://arxiv.org/abs/1902.10660)).

RobustTrees is a fork of XGBoost 0.72 (2018). This version moves its robust tree
updater to the XGBoost 1.7 release line (tag `v1.7.6`), so it builds with current
compilers and runs with Python 3.10/3.11:

- `XGBClassifier(tree_method="robust_exact", robust_eps=...)` and the native
  `xgb.train` interface keep the original names and defaults;
- the robust updater is rewritten for the 1.7 internals and gave bit-identical
  predictions and identical splits to RobustTrees in every tested case;
- tests: `tests/python/test_robust_exact.py`.

**Note:** the scikit-learn wrapper of XGBoost 1.7 has different default
hyperparameters than 0.72 (`max_depth` 6 instead of 3, learning rate 0.3 instead
of 0.1). Pass them explicitly to reproduce models trained with RobustTrees.

This version was made so that the code of the author's MSc thesis on explanations
of adversarially robust tree ensembles (Warsaw University of Technology, 2023) runs
on current Python. All credit for the algorithm and the original code goes to their
authors. XGBoost and RobustTrees are licensed under the Apache License 2.0 (see
`LICENSE`); this version keeps that license.

## Citation

This is the same robust tree building logic as in RobustTrees, carried over to a
newer XGBoost. If you use it, please cite the original work:

H. Chen, H. Zhang, D. Boning, C.-J. Hsieh. Robust Decision Trees Against Adversarial
Examples. ICML 2019, PMLR 97:1122-1131.
<https://proceedings.mlr.press/v97/chen19m.html>

```bibtex
@inproceedings{chen2019robust,
  title     = {Robust Decision Trees Against Adversarial Examples},
  author    = {Chen, Hongge and Zhang, Huan and Boning, Duane and Hsieh, Cho-Jui},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {97},
  pages     = {1122--1131},
  year      = {2019},
  publisher = {PMLR}
}
```

Build instructions, details and known limitations: [ROBUST_PORT.md](ROBUST_PORT.md).

Branches: `robust-xgboost-1.7` (this version), `master` (RobustTrees, unchanged),
`mirror-notes` (RobustTrees plus a description).

The upstream XGBoost README follows.

---
<img src=https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png width=135/>  eXtreme Gradient Boosting
===========
[![Build Status](https://xgboost-ci.net/job/xgboost/job/master/badge/icon)](https://xgboost-ci.net/blue/organizations/jenkins/xgboost/activity)
[![Build Status](https://img.shields.io/travis/dmlc/xgboost.svg?label=build&logo=travis&branch=master)](https://travis-ci.org/dmlc/xgboost)
[![XGBoost-CI](https://github.com/dmlc/xgboost/workflows/XGBoost-CI/badge.svg?branch=master)](https://github.com/dmlc/xgboost/actions)
[![Documentation Status](https://readthedocs.org/projects/xgboost/badge/?version=latest)](https://xgboost.readthedocs.org)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![CRAN Status Badge](http://www.r-pkg.org/badges/version/xgboost)](http://cran.r-project.org/web/packages/xgboost)
[![PyPI version](https://badge.fury.io/py/xgboost.svg)](https://pypi.python.org/pypi/xgboost/)
[![Conda version](https://img.shields.io/conda/vn/conda-forge/py-xgboost.svg)](https://anaconda.org/conda-forge/py-xgboost)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue)](https://optuna.org)
[![Twitter](https://img.shields.io/badge/@XGBoostProject--_.svg?style=social&logo=twitter)](https://twitter.com/XGBoostProject)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/dmlc/xgboost/badge)](https://api.securityscorecards.dev/projects/github.com/dmlc/xgboost)

[Community](https://xgboost.ai/community) |
[Documentation](https://xgboost.readthedocs.org) |
[Resources](demo/README.md) |
[Contributors](CONTRIBUTORS.md) |
[Release Notes](NEWS.md)

XGBoost is an optimized distributed gradient boosting library designed to be highly ***efficient***, ***flexible*** and ***portable***.
It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework.
XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
The same code runs on major distributed environment (Kubernetes, Hadoop, SGE, MPI, Dask) and can solve problems beyond billions of examples.

License
-------
© Contributors, 2021. Licensed under an [Apache-2](https://github.com/dmlc/xgboost/blob/master/LICENSE) license.

Contribute to XGBoost
---------------------
XGBoost has been developed and used by a group of active community members. Your help is very valuable to make the package better for everyone.
Checkout the [Community Page](https://xgboost.ai/community).

Reference
---------
- Tianqi Chen and Carlos Guestrin. [XGBoost: A Scalable Tree Boosting System](http://arxiv.org/abs/1603.02754). In 22nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2016
- XGBoost originates from research project at University of Washington.

Sponsors
--------
Become a sponsor and get a logo here. See details at [Sponsoring the XGBoost Project](https://xgboost.ai/sponsors). The funds are used to defray the cost of continuous integration and testing infrastructure (https://xgboost-ci.net).

## Open Source Collective sponsors
[![Backers on Open Collective](https://opencollective.com/xgboost/backers/badge.svg)](#backers) [![Sponsors on Open Collective](https://opencollective.com/xgboost/sponsors/badge.svg)](#sponsors)

### Sponsors
[[Become a sponsor](https://opencollective.com/xgboost#sponsor)]

<a href="https://www.nvidia.com/en-us/" target="_blank"><img src="https://raw.githubusercontent.com/xgboost-ai/xgboost-ai.github.io/master/images/sponsors/nvidia.jpg" alt="NVIDIA" width="72" height="72"></a>
<a href="https://www.intel.com/" target="_blank"><img src="https://images.opencollective.com/intel-corporation/2fa85c1/logo/256.png" width="72" height="72"></a>
<a href="https://getkoffie.com/?utm_source=opencollective&utm_medium=github&utm_campaign=xgboost" target="_blank"><img src="https://images.opencollective.com/koffielabs/f391ab8/logo/256.png" width="72" height="72"></a>

### Backers
[[Become a backer](https://opencollective.com/xgboost#backer)]

<a href="https://opencollective.com/xgboost#backers" target="_blank"><img src="https://opencollective.com/xgboost/backers.svg?width=890"></a>
