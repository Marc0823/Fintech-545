# FinTech 545 - Assignment 2

The complete written solution is **Assignment2.pdf**. It answers Problems 1-5
with Predict, Fit, and Reconcile sections and labels every lettered question.

## Environment

Calculations were run with Python **3.10.6**, NumPy **2.1.3**, pandas **2.2.3**,
SciPy **1.15.3**, and matplotlib **3.9.2**. Only these four scientific packages
are needed to run the calculations. PDF rendering additionally uses Quarto,
a Jupyter Python kernel, and a LaTeX installation with pdfLaTeX.

Install the Python dependencies:

```bash
python -m pip install numpy==2.1.3 pandas==2.2.3 scipy==1.15.3 matplotlib==3.9.2
python -m pip install "ipykernel>=6.29" nbclient nbformat PyYAML
```

Install the standalone [Quarto CLI](https://quarto.org/docs/get-started/) and
place it on PATH. Quarto **1.10.19**, ipykernel **6.31.0**, and TinyTeX (TeX Live
**2026**) were used here. A Python package named
`quarto` is not a replacement for the CLI. If no LaTeX distribution is installed,
Quarto can install TinyTeX:

```bash
quarto install tinytex
```

## Run

From this `Assignment2` folder:

```bash
python assignment2.py
quarto render assignment2.qmd
```

The first command finishes all calculations and checks in one script. The
second executes the report's small formatting cells and produces
`Assignment2.pdf`. If Quarto selects another Python installation, set
`QUARTO_PYTHON` to the Python executable used for the installation above. In
PowerShell, for example:

```powershell
$env:QUARTO_PYTHON = (Get-Command python).Source
quarto render assignment2.qmd
```

## Files

- `assignment2.py`: all numerical work and the three requested figures.
- `assignment2.qmd`: report source; numbers and tables read `results.json`.
- `Assignment2.pdf`: complete rendered solution.
- `problem1.csv` through `problem5.csv`: unchanged course input data.
- `results.json`: generated full-precision results and validation status.
- `problem4_daily_loglik.csv`: generated Gaussian/t likelihood contribution
  for every observation, their difference, and both non-extreme indicators.
- `figures/`: generated return plot, bond P&L distributions, and rank-pair plots.

Rerunning calculations overwrites only the generated results and figures.
Rendering overwrites the PDF; Quarto may also create a temporary cache folder.

## Conventions

- Seed 545; exactly 100,000 draws for each requested copula and factor-model
  simulation. Copulas share normal inputs; t mixing shocks use seed 546.
- Loss-convention VaR is minus the lower P&L quantile. Negative bond VaR means
  a quantile gain and is deliberately not made positive or floored at zero.
- Historical quantiles and ES follow `library/RiskStats.jl`; all requested
  tail sizes are integer order statistics. The PDF gives the exact convention.
- Sample variance uses `n-1`; normal MLE uses `n`. Skewness and excess kurtosis
  use uncorrected standardized central moments. EW variance uses normalized
  finite weights and removes the weighted mean as in `library/ewCov.jl`.
- The Problem 2 visual regime split is days 1-460 versus 461-500, chosen before
  fitting. The original prediction is retained and its errors are reconciled.
- Copula comparison fixes selected margins and the Kendall-based R. The course
  convention therefore counts 0 free Gaussian parameters and 1 free t parameter;
  the profile uses the exact two-stage 200-point course grid.
- Problem 4 non-extreme days use own empirical rank tails. Fitted-CDF tail
  classification is also reported as a sensitivity check. The row-952 outlier
  is retained in all fits and risk calculations.
- Problem 5 reports OLS intercepts but simulates zero-mean stock returns as
  required. Its residual covariance uses `n-1` for consistency with direct
  sample covariance; `n-2` residual error SDs are also reported for reference.

The source snapshot is
[`dompazz/FinTech-545-Fall2026`, commit 6f104966e891a3326898d51e291c8a0153b454f7](https://github.com/dompazz/FinTech-545-Fall2026/tree/6f104966e891a3326898d51e291c8a0153b454f7).
