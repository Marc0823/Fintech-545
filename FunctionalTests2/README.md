# Functional Tests 2

Python implementation of the remaining 25 cases (8.1 through 13.9) in the course test workbook.

## Run from the repository root

```bash
python -m pip install -r FunctionalTests2/requirements.txt
python FunctionalTests2/run_tests.py
```

## Coverage

| Tests | Calculation |
| --- | --- |
| 8.1–8.6 | Normal, Student-t, and simulated 95% VaR and ES |
| 9.1 | Gaussian copula portfolio risk by holding and in total |
| 10.1–10.4 | Risk budgets and constrained maximum Sharpe portfolios |
| 11.1–11.2 | Stock and factor return and volatility attribution |
| 12.1–12.3 | European values and Greeks, American values and Greeks, discrete cash dividends |
| 13.1–13.9 | Multivariate t fit, copula fitting and selection, tail dependence, portfolio risk |

## Files and comparisons

functions.py contains the calculations. run_tests.py reads the input CSVs and compares each output to the corresponding course benchmark. A failed comparison returns a nonzero exit code. The results directory contains one calculated CSV per case and comparison_report.csv with the tolerances used.

For deterministic cases the relative tolerance is 0.001 with an absolute floor of 1e-7. The two 10,000-draw simulations allow 0.005 in return units. The two 100,000-draw portfolio simulations allow 3.5% relative error. Python and Julia generate different draws even when seeded. These comparison thresholds are project choices.

Course data and numerical conventions come from [the course test files](https://github.com/dompazz/FinTech-545-Fall2026/tree/main/testfiles).
