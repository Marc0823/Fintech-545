# Functional Tests 1

Python implementation of FINTECH 545 Tests 1.1–7.6.

## Run

From the repository root:

```powershell
python -m pip install -r FunctionalTests1/requirements.txt
python FunctionalTests1/run_tests.py
```

## Files

- `functions.py`
  Functions for covariance, PSD repair, simulation, returns, and distribution fitting.

- `run_tests.py`
  Runs Tests 1.1–7.6 and compares calculated results with the provided expected outputs.

- `data/`
  Input and expected-output files from the course repository.

- `results/`
  Created when `run_tests.py` is executed and contains calculated outputs and the comparison report.

## Topics

- Tests 1–2: covariance, correlation, missing data, and exponentially weighted covariance.
- Tests 3–4: PSD matrix repair and Cholesky factorization.
- Test 5: normal and PCA simulation.
- Test 6: arithmetic and log returns.
- Test 7: Normal, Student-t, regression-t, NIG fitting, and AICc.

The implementation uses Python with NumPy, pandas, and SciPy.
