# Functional Tests 1

Python implementation of FINTECH 545 Tests 1.1–7.6 (25 tests).

## Run

From the repository root:

```powershell
python -m pip install -r FunctionalTests1/requirements.txt
python FunctionalTests1/run_tests.py
```

Tested with Python 3.10.6, NumPy 2.1.3, pandas 2.2.3, and SciPy 1.15.3.
The runner prints PASS/FAIL for each test, writes the output CSVs and
`results/comparison_report.csv`, and exits nonzero if any test fails.
Paths are relative to the script, so the working directory does not matter.

## Files

```text
FunctionalTests1/
├── README.md
├── requirements.txt
├── run_tests.py
├── functions.py
├── data/
└── results/
```

`functions.py` contains the calculations. `run_tests.py` reads the inputs,
checks a few mathematical properties, and compares the results.
`data/` contains the 35 unchanged instructor input/expected-output CSVs.
`results/` contains 25 calculated CSVs and the comparison report.
The output filenames and column order match the instructor files; no CSV index
column is written. `Assignment1/` is not used or modified.

## Conventions

| Tests | Calculation and conventions |
|---|---|
| 1.1–1.4 | Listwise/pairwise covariance and Pearson correlation. Covariance divides by n−1; pairwise correlation uses each pair's shared rows for both standard deviations. |
| 2.1–2.3 | Normalized exponential weights, newest row weighted most, weighted centering, no degrees-of-freedom correction. Test 2.3 uses variances at lambda=.97 and correlations at lambda=.94, following the code rather than its reversed comment. |
| 3.1–3.4 | near_psd eigenvalue clipping and Higham/Dykstra projections in correlation space, then restore original variances. Higham uses identity weights, distance-change tolerance 1e-9, minimum eigenvalue >−1e-9, and at most 100 iterations. |
| 4.1 | Lower PSD Cholesky; residual pivots in [−1e-8,0] become zero. Check reconstruction. |
| 5.1–5.5 | 100,000 zero-mean draws, sample covariance (ddof=1), appropriate repair. PCA sorts eigenvalues descending, keeps values >=1e-8, and retains the minimum factors explaining .99 of total variance. This input retains two factors. |
| 6.1–6.2 | Arithmetic/log price returns. Keep input dates from row 2 onward and preserve asset column order; no initial missing-return row. |
| 7.1–7.4 | Normal uses sample SD. Generalized t uses location and scale, with sigma>=1e-6 and nu>=2.0001. Regression fixes error location at zero and estimates a separate intercept. AICc uses k=3 for the univariate t. |
| 7.5–7.6 | NIG closed-form moment inversion and SciPy MLE. Moments use variance ddof=1 and uncorrected skewness/excess kurtosis. Both return mu,alpha,beta,delta; SciPy conversion is delta=scale, alpha=a/delta, beta=b/delta, mu=loc. |

The t fits use scaled data, analytic likelihood derivatives, and four fixed
starting degrees of freedom. Only converged solutions are accepted. NIG MLE
uses SciPy's default optimizer settings and checks its convergence flag.
The moment fit checks the admissible NIG region.

Tests 3 and 4 read earlier expected matrices only where the instructor specifies
those matrices as inputs. Functions do not read files or use expected answers.

## Comparisons

Column names/order, dimensions, and dates must match exactly. Numeric comparisons
use `abs(actual-expected) <= atol + rtol*abs(expected)`:

| Cases | atol | rtol |
|---|---:|---:|
| 1.*, 2.*, 6.*, 7.1, 7.5 | 1e-12 | 1e-10 |
| 3.* | 1e-9 | 1e-9 |
| 4.1 | 1e-7 | 1e-10 |
| 7.2, 7.3, 7.6 | 1e-10 | 1e-8 |
| 7.4 | 1e-7 | 1e-10 |

Higham tolerates tiny negative eigenvalues at its stopping threshold. Cholesky
allows square-root roundoff at a zero pivot; reconstruction is also checked at
atol=1e-12, rtol=1e-10. Optimizer tolerances allow trailing-digit differences.

For simulations, each test starts a local NumPy PCG64 generator with seed 1234.
Python and Julia streams differ, so the same statistical rule is used for all
five tests. With target covariance C and N=100000:

```text
SE_ij = sqrt((C_ij^2 + C_ii*C_jj)/(N-1))
```

Generated and instructor covariances are each checked against the target using
SE. Their difference is checked using sqrt(2)*SE, and simulated means using
sqrt(C_ii/N). All four maximum standardized errors must be <=5. The target is
the original, repaired, or PCA-truncated covariance as appropriate. These are
local statistical checks, not an instructor-published grading threshold.
The seed and threshold are unchanged from the previous implementation.

All 25 tests pass locally: 20 deterministic and 5 statistical. The report lists
absolute/relative discrepancies and all four simulation checks. Relative errors
exclude zero expected values, but those entries still receive an absolute check.

## Reference

Algorithms and original CSVs come from
[FinTech-545-Fall2026](https://github.com/dompazz/FinTech-545-Fall2026/tree/add6d02a8a3960bfa0bc8ed8ea2f45a5834f3459),
commit `add6d02a8a3960bfa0bc8ed8ea2f45a5834f3459`:
`testfiles/test_setup.jl`, `testfiles/data/`, and the covariance, simulation,
return, and fitting functions in `library/`. The actual generation code and
committed CSVs take precedence over comments or stale workbook filenames.
