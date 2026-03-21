# MRMRS

Python library that implements the mRMR algorithm for feature selection. mRMR means "minimum Redundancy - Maximum Relevance". It is a feature selection algorithm that calculates the relevance of each feature to the target variable and also the redundance between selected features to provide a "minimal optimal" feature set that is highly relevant to the target variable while minimizing the redundance between the selected features.

Currently the following methods are implemented (with plans to add more):

**relevance**
- f-statistic from one-way ANOVA for classification
- pearsons correlation coefficient for regression

**redundance**
- pearsons correlation

This is a re-implementation of the [existing mRMR package](https://github.com/smazzanti/mrmr) that sadly does not seem to be mainined anymore.

This package is written in rust and uses the polars api from rust so it only works on polars dataframes (pandas dataframes can be converted to polars easily for the purposes of using them with this package.)

## How to use
```python
import polars as pl
from mrmrs import mrmr

df: pl.DataFrame = pl.read_csv("dataset.csv")
y: pl.Series = pl.read_csv("target.csv")["target"]

selected_features: list[mrmr.Feature] = mrmr(
    X=df, 
    y=y, 
    number_of_features=50,
    task_type="classification" # or "regression"
)

selected_feature_names: list[str] = [feature.name for feature in selected_features]
subset_df = df.select(selected_feature_names)
```

## Installation

```bash
```

or using pip you can do

```bash
```

currently wheels are available for macs with arm64 chips and linux with x86_64. On other platforms you will need the rust compiler installed to build the package.

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

## Performance
The original mRMR package is quite slow. This package
- makes no clones and is very efficient about memory with the polars apache arrow memory model
- calculates redundancies incrementally in each round in parallel
- is a binary with minimal dependencies so it imports in ms even on cold imports (mRMR takes ~20s on cold imports and 700ms on warm imports for me on M4 chip) while this package takes under 5 ms in both cases
- The actual mRMR algorithm runs faster, a lot faster. Testing on toy datasets created by `sklearns.datsets.make_classification` with 5k-80k features, 1000 samples and 100 selected features this package was 80-280x faster than the mrmr-selection package, the speedup is better on larger feature sets, for the one with 80k features, this package took 3s while mrmr-selection took 14 minutes!

## Resources
- [Minimum Redundancy Feature Selection from Microarray Gene Expression Data (o.g. paper)](https://ranger.uta.edu/~chqding/papers/gene_select.pdf)
- [Maximum Relevance and Minimum Redundancy Feature Selection Methods for a Marketing Machine Learning Platform](https://arxiv.org/pdf/1908.05376)
