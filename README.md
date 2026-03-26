# MRMRS

[![PyPI](https://img.shields.io/pypi/v/mrmrs)](https://pypi.org/project/mrmrs/)
[![Tests](https://github.com/JanLubojacky/mrmrs/actions/workflows/test.yml/badge.svg)](https://github.com/JanLubojacky/mrmrs/actions/workflows/test.yml)

Blazing-fast feature selection using the mRMR (minimum Redundancy - Maximum Relevance) algorithm.

- **100-500x faster** than [mrmr-selection](https://github.com/smazzanti/mrmr)
- **Works directly on Polars DataFrames** — zero-copy, no conversions needed
- **Sub-5ms import time** — no heavy dependencies, just a Rust binary
- **Written in Rust** with SIMD acceleration and incremental redundancy updates

## Installation

```bash
uv add mrmrs
```

or

```bash
pip install mrmrs
```

## Quick Start
```python
import polars as pl
from mrmrs import mrmr, Feature

df: pl.DataFrame = pl.read_csv("dataset.csv")
y: pl.Series = pl.read_csv("target.csv")["target"]

selected_features: list[Feature] = mrmr(
    x=df,
    y=y,
    number_of_features=50,
    task_type="classification" # or "regression"
)

selected_feature_names: list[str] = [feature.name for feature in selected_features]
subset_df = df.select(selected_feature_names)
```

## Supported Methods

**Relevance**
- F-statistic from one-way ANOVA (classification)
- Pearson's correlation coefficient (regression)

**Redundancy**
- Pearson's correlation coefficient

## Benchmarks

Benchmarked against the [mrmr-selection](https://github.com/smazzanti/mrmr) package on synthetic data generated with `sklearn.datasets.make_classification` (1000 samples, classification task). Each configuration was run 5 times, reporting mean +/- std.

mrmrs achieves its speed through Rust with SIMD (scalar fallback when unavailable), zero-copy Polars Arrow memory, and an incremental redundancy update that avoids recomputation each round.

**Hardware:** AMD Ryzen 5 5600 6-Core (12 threads), 32 GB RAM

### Scaling total features (selecting 50 features)

| Features | mrmrs (s) | mrmr_selection (s) | Speedup |
|----------|-----------|---------------------|---------|
| 500 | 0.009 ± 0.001 | 2.286 ± 0.602 | **198x** |
| 1,000 | 0.016 ± 0.002 | 3.181 ± 0.041 | **234x** |
| 5,000 | 0.069 ± 0.005 | 18.242 ± 0.042 | **269x** |
| 10,000 | 0.170 ± 0.005 | 30.680 ± 0.130 | **184x** |
| 20,000 | 0.357 ± 0.010 | 56.938 ± 1.506 | **155x** |
| 50,000 | 0.963 ± 0.007 | 136.115 ± 1.191 | **143x** |

### Scaling selected features (from 5,000 total features)

| Selected (K) | mrmrs (s) | mrmr_selection (s) | Speedup |
|-------------|-----------|---------------------|---------|
| 10 | 0.048 ± 0.001 | 4.201 ± 0.112 | **89x** |
| 50 | 0.070 ± 0.001 | 18.473 ± 0.218 | **263x** |
| 100 | 0.097 ± 0.002 | 36.183 ± 0.076 | **380x** |
| 200 | 0.149 ± 0.005 | 71.431 ± 0.622 | **519x** |

The speedup increases with larger K because mrmrs calculates redundancies incrementally in parallel while mrmr_selection recomputes them from scratch each round.

To reproduce: `uv run --package mrmrs-bench python bench/benchmark.py`

## Development
**Requirements**
- uv, cargo, maturin

1. **create uv venv**
```sh
uv sync
```
2. **activate uv venv**
```sh
source .venv/bin/activate
```
3. **install this pkg to this uv venv in release mode**
```sh
maturin develop --release --uv
```
alternatively if you enter this repository while you have another venv activated and run the same command maturin will install this pkg to that environment

## Running tests
- currently there is no separation between the rust crate and the python bindings so this requries the venv to be activated
```sh
cargo test
```

## References
- [mrmr-selection](https://github.com/smazzanti/mrmr)
- [Numerically stable parallel computation of (co-)variance](https://dl.acm.org/doi/10.1145/3221269.3223036)
- [Minimum Redundancy Feature Selection from Microarray Gene Expression Data (o.g. mRMR paper)](https://ranger.uta.edu/~chqding/papers/gene_select.pdf)
- [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://arxiv.org/pdf/1908.05376)
