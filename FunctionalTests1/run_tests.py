from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats
import functions as f

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
N = 100_000
SEED = 1234


def read(name):
    return pd.read_csv(DATA / name, float_precision="round_trip")


def matrix(name):
    return read(name).to_numpy(dtype=float)


def check_matrix(a, psd=True, psd_atol=1e-9):
    assert np.isfinite(a).all(), "Nonfinite matrix"
    assert np.allclose(a, a.T, atol=1e-12, rtol=1e-10), "Matrix is not symmetric"
    if psd:
        assert np.linalg.eigvalsh(a)[0] >= -psd_atol, "Matrix is not PSD within tolerance"


def calculate(group, i):
    simulation = None
    if group == 1:
        x = matrix("test1.csv")
        if i in (1, 3):
            a = f.missing_cov(x, skip_missing=i == 1)
        else:
            a = f.missing_corr(x, skip_missing=i == 2)
            assert np.allclose(np.diag(a), 1, atol=1e-12, rtol=1e-10)
        check_matrix(a, psd=i <= 2)
    elif group == 2:
        x = matrix("test2.csv")
        if i == 1:
            a = f.ew_cov(x, .97)
        elif i == 2:
            a = f.ew_corr(x, .94)
            assert np.allclose(np.diag(a), 1, atol=1e-12, rtol=1e-10)
        else:
            a = f.mixed_ew_cov(x)
            assert np.allclose(np.diag(a), np.diag(f.ew_cov(x, .97)), atol=1e-12, rtol=1e-10)
        check_matrix(a)
    elif group == 3:
        # these earlier outputs are the instructor-specified inputs
        x = matrix("testout_1.3.csv" if i in (1, 3) else "testout_1.4.csv")
        a = f.near_psd(x) if i <= 2 else f.higham_nearestPSD(x)
        check_matrix(a, psd_atol=1e-9*np.max(np.diag(x)))
        assert np.allclose(np.diag(a), np.diag(x), atol=1e-12, rtol=1e-10)
        if i in (2, 4):
            assert np.allclose(np.diag(a), 1, atol=1e-12, rtol=1e-10)
    elif group == 4:
        x = matrix("testout_3.1.csv")
        a = f.chol_psd(x)
        assert np.array_equal(a, np.tril(a)) and (np.diag(a) >= 0).all()
        assert np.allclose(a @ a.T, x, atol=1e-12, rtol=1e-10), "Cholesky reconstruction failed"
    elif group == 5:
        name = "test5_1.csv" if i == 1 else "test5_2.csv" if i in (2, 5) else "test5_3.csv"
        x = matrix(name)
        if i == 5:
            b, values, total = f.pca_factors(x, .99)
            assert len(values) == 2, "Expected two retained PCA factors"
            assert values.sum()/total >= .99 and values[:-1].sum()/total < .99
            target = b @ b.T
            draws = f.simulate_pca(x, N, pctExp=.99, seed=SEED)
        else:
            target = f.near_psd(x) if i == 3 else f.higham_nearestPSD(x) if i == 4 else x
            check_matrix(target, psd_atol=1e-9*np.max(np.diag(x)))
            assert np.allclose(np.diag(target), np.diag(x), atol=1e-12, rtol=1e-10)
            repair = f.higham_nearestPSD if i == 4 else f.near_psd
            draws = f.simulateNormal(N, x, seed=SEED, fix_method=repair)
        assert draws.shape == (N, 5)
        a = np.cov(draws, rowvar=False, ddof=1)
        check_matrix(a)
        simulation = target, draws
    elif group == 6:
        prices = read("test6.csv")
        out = f.return_calculate(prices, "ARITHMETIC" if i == 1 else "LOG")
        assert out.columns.tolist() == prices.columns.tolist()
        assert out.Date.tolist() == prices.Date.iloc[1:].tolist()
        assert len(out) == len(prices)-1
        return out, simulation
    elif group == 7:
        if i == 1:
            x = matrix("test7_1.csv")[:, 0]
            p = f.fit_normal(x)
            assert p["sigma"] == np.std(x, ddof=1)
        elif i in (2, 4):
            x = matrix("test7_2.csv")[:, 0]
            p = f.fit_general_t(x)
        elif i == 3:
            x = read("test7_3.csv")
            p = f.fit_regression_t(x.y.to_numpy(), x.drop(columns="y").to_numpy())
            assert p["mu"] == 0 and len(p) == 7
        else:
            x = matrix("test7_5.csv")[:, 0]
            p = f.fit_nig_moments(x) if i == 5 else f.fit_NIG_mle(x)
            assert p["delta"] > 0 and p["alpha"] > abs(p["beta"])
            if i == 5:
                d = stats.norminvgauss(p["alpha"]*p["delta"], p["beta"]*p["delta"], loc=p["mu"], scale=p["delta"])
                sample = [x.mean(), x.var(ddof=1), stats.skew(x), stats.kurtosis(x)]
                assert np.allclose(d.stats(moments="mvsk"), sample, atol=1e-10, rtol=1e-9)
        assert np.isfinite(list(p.values())).all()
        if "nu" in p:
            assert p["sigma"] >= 1e-6-1e-12 and p["nu"] >= 2.0001-1e-10
        if i == 4:
            ll = stats.t.logpdf(x, p["nu"], loc=p["mu"], scale=p["sigma"]).sum()
            p = {"AICC": f.AICc(float(ll), 3, len(x))}
        return pd.DataFrame([p]), simulation
    return pd.DataFrame(a, columns=[f"x{j+1}" for j in range(a.shape[1])]), simulation


def main():
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for group, count in [(1, 4), (2, 3), (3, 4), (4, 1), (5, 5), (6, 2), (7, 6)]:
        for i in range(1, count + 1):
            test = f"{group}.{i}"
            name = f"testout_{test}.csv" if group <= 5 else f"testout{group}_{i}.csv"
            atol, rtol = 1e-12, 1e-10
            if group == 3:
                atol, rtol = 1e-9, 1e-9
            elif group == 4:
                atol, rtol = 1e-7, 1e-10
            elif group == 7 and i in (2, 3, 6):
                atol, rtol = 1e-10, 1e-8
            elif group == 7 and i == 4:
                atol, rtol = 1e-7, 1e-10
            row = {"test": test, "status": "FAIL", "output": name,
                   "comparison": "statistical" if group == 5 else "numerical",
                   "max_absolute_error": np.nan, "max_relative_error": np.nan,
                   "absolute_tolerance": atol if group != 5 else np.nan,
                   "relative_tolerance": rtol if group != 5 else np.nan}
            try:
                out, simulation = calculate(group, i)
                out.to_csv(RESULTS / name, index=False, lineterminator="\n", float_format="%.17g")
                actual = pd.read_csv(RESULTS / name, float_precision="round_trip")
                expected = read(name)
                assert actual.columns.tolist() == expected.columns.tolist(), "Column names/order differ"
                assert actual.shape == expected.shape, "Output dimensions differ"
                if "Date" in expected.columns:
                    assert actual.Date.tolist() == expected.Date.tolist(), "Dates differ"
                a = actual.drop(columns="Date", errors="ignore").to_numpy(dtype=float)
                b = expected.drop(columns="Date", errors="ignore").to_numpy(dtype=float)
                assert np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite output"
                error = np.abs(a - b)
                row["max_absolute_error"] = float(error.max())
                nonzero = b != 0
                row["max_relative_error"] = float((error[nonzero]/np.abs(b[nonzero])).max()) if nonzero.any() else 0.
                passed = np.all(error <= atol + rtol*np.abs(b))
                if group == 5:
                    target, draws = simulation
                    v = np.diag(target)
                    se = np.sqrt((target**2 + np.outer(v, v)) / (N - 1))
                    # two independent sample covariances have sqrt(2) times the SE
                    row["max_target_z"] = float(np.max(np.abs(a-target)/se))
                    row["max_reference_target_z"] = float(np.max(np.abs(b-target)/se))
                    row["max_reference_difference_z"] = float(np.max(error/(np.sqrt(2)*se)))
                    row["max_mean_z"] = float(np.max(np.abs(draws.mean(axis=0))/np.sqrt(v/N)))
                    row["z_threshold"] = 5.
                    row["seed"] = SEED
                    row["sample_size"] = N
                    passed = max(row["max_target_z"], row["max_reference_target_z"],
                                 row["max_reference_difference_z"], row["max_mean_z"]) <= 5
                row["status"] = "PASS" if passed else "FAIL"
                row["details"] = "" if passed else "Comparison tolerance exceeded"
            except Exception as exc:
                row["details"] = str(exc) or type(exc).__name__
            rows.append(row)
            print(f"{test:5} {row['status']:4}  abs={row['max_absolute_error']:.6g}  "
                  f"rel={row['max_relative_error']:.6g}  {row['details']}")
    pd.DataFrame(rows).to_csv(RESULTS / "comparison_report.csv", index=False,
                            lineterminator="\n", float_format="%.17g")
    passed = sum(row["status"] == "PASS" for row in rows)
    print(f"\n{passed} / 25 passed")
    return 0 if passed == 25 else 1


if __name__ == "__main__":
    sys.exit(main())
