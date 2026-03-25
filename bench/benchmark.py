import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from mrmr import mrmr_classif
from sklearn.datasets import make_classification

import mrmrs

DATA_DIR = Path(__file__).parent / "data"


def generate_data(
    n_samples: int, n_features: int
) -> tuple[pl.DataFrame, pl.Series, pd.DataFrame, pd.Series]:
    DATA_DIR.mkdir(exist_ok=True)
    x_path = DATA_DIR / f"X_{n_samples}x{n_features}.parquet"
    y_path = DATA_DIR / f"y_{n_samples}x{n_features}.parquet"

    if x_path.exists() and y_path.exists():
        print(f"  Loading cached data from {DATA_DIR} ...")
        X_pl = pl.read_parquet(x_path)
        y_pl = pl.read_parquet(y_path).to_series()
        X_pd = X_pl.to_pandas()
        y_pd = y_pl.to_pandas()
        return X_pl, y_pl, X_pd, y_pd

    X_np, y_np = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(10, n_features // 10),
        n_redundant=max(5, n_features // 20),
        random_state=42,
    )
    columns = [f"f_{i}" for i in range(n_features)]

    X_pd = pd.DataFrame(X_np, columns=columns)
    y_pd = pd.Series(y_np, name="target")

    X_pl = pl.from_pandas(X_pd)
    y_pl = pl.from_pandas(y_pd)

    X_pl.write_parquet(x_path)
    y_pl.to_frame().write_parquet(y_path)
    print(f"  Saved data to {DATA_DIR}")

    return X_pl, y_pl, X_pd, y_pd


def bench_mrmrs(X_pl: pl.DataFrame, y_pl: pl.Series, k: int) -> float:
    start = time.perf_counter()
    mrmrs.mrmr(X_pl, y_pl, k, "classification")
    return time.perf_counter() - start


def bench_mrmr_selection(X_pd: pd.DataFrame, y_pd: pd.Series, k: int) -> float:
    start = time.perf_counter()
    mrmr_classif(X_pd, y_pd, K=k)
    return time.perf_counter() - start


def run_benchmark(
    n_features: int, k: int, n_runs: int = 5, n_samples: int = 1000
) -> dict:
    print(f"  Generating data: {n_samples} samples x {n_features} features ...")
    X_pl, y_pl, X_pd, y_pd = generate_data(n_samples, n_features)

    times_mrmrs = []
    times_mrmr_sel = []

    for i in range(n_runs):
        print(f"  Run {i + 1}/{n_runs} ...", end=" ", flush=True)

        t = bench_mrmrs(X_pl, y_pl, k)
        times_mrmrs.append(t)
        print(f"mrmrs={t:.3f}s", end=" ", flush=True)

        t = bench_mrmr_selection(X_pd, y_pd, k)
        times_mrmr_sel.append(t)
        print(f"mrmr_selection={t:.3f}s")

    mrmrs_mean = np.mean(times_mrmrs)
    mrmrs_std = np.std(times_mrmrs)
    sel_mean = np.mean(times_mrmr_sel)
    sel_std = np.std(times_mrmr_sel)

    return {
        "n_features": n_features,
        "k": k,
        "mrmrs_mean": mrmrs_mean,
        "mrmrs_std": mrmrs_std,
        "mrmr_sel_mean": sel_mean,
        "mrmr_sel_std": sel_std,
        "speedup": sel_mean / mrmrs_mean,
    }


def print_results(results: list[dict], title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f" {title}")
    print(f"{'=' * 80}")
    print(
        f"{'Features':>10} {'K':>5} {'mrmrs (s)':>18} {'mrmr_selection (s)':>22} {'Speedup':>10}"
    )
    print(f"{'-' * 80}")
    for r in results:
        print(
            f"{r['n_features']:>10} {r['k']:>5} "
            f"{r['mrmrs_mean']:>8.3f} ± {r['mrmrs_std']:<7.3f} "
            f"{r['mrmr_sel_mean']:>8.3f} ± {r['mrmr_sel_std']:<7.3f} "
            f"{r['speedup']:>8.1f}x"
        )


def main() -> None:
    print("Benchmark: mrmrs vs mrmr_selection")
    print(f"Each configuration is run 5 times, reporting mean ± std\n")

    # Benchmark 1: Scaling total features, fixed K=50
    print("=" * 80)
    print(" Benchmark 1: Selecting 50 features from increasing feature sets")
    print("=" * 80)
    feature_counts = [500, 1_000, 5_000, 10_000, 20_000, 50_000]
    results_1 = []
    for n_features in feature_counts:
        print(f"\n[{n_features} features, K=50]")
        result = run_benchmark(n_features, k=50)
        results_1.append(result)

    # Benchmark 2: Scaling K, fixed 5000 features
    print("\n" + "=" * 80)
    print(" Benchmark 2: Selecting increasing features from 5000 total")
    print("=" * 80)
    k_values = [10, 50, 100, 200]
    results_2 = []
    for k in k_values:
        print(f"\n[5000 features, K={k}]")
        result = run_benchmark(5_000, k=k)
        results_2.append(result)

    # Summary tables
    print_results(results_1, "Benchmark 1: Scaling total features (K=50)")
    print_results(results_2, "Benchmark 2: Scaling selected features (total=5000)")


if __name__ == "__main__":
    main()
