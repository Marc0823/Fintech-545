# Assignment 1 - Univariate and Multivariate Statistics

FinTech 545 - Quantitative Risk Management

This folder contains the final written submission, executable analysis code,
source data, numerical output, figures, and the editable report source for
Assignment 1 in FinTech 545. The submitted written report is `Assignment1.pdf`;
`report.md` preserves its approved source text.

## Python and dependencies

The results were verified with Python 3.10.6 and these package versions:

- numpy 2.1.3
- pandas 2.2.3
- scipy 1.15.3
- statsmodels 0.14.4
- matplotlib 3.9.2

From the repository root, install the verified versions with:

```powershell
python -m pip install numpy==2.1.3 pandas==2.2.3 scipy==1.15.3 statsmodels==0.14.4 matplotlib==3.9.2
```

## Run the assignment

From the repository root, run:

```powershell
python Assignment1/code/assignment1.py
```

The script resolves paths from its own location, so it does not depend on the
current working directory. It validates all five CSV schemas, missing values,
and finite numeric values before calculating anything. If a required CSV is
missing, it stops with the exact missing path instead of fabricating data.

The command prints all labeled numerical results to the console and also writes
the same complete output to `Assignment1/results/results.txt`. It overwrites the
generated figures and result transcript deterministically on each run.

## Directory structure

```text
Assignment1/
|-- Assignment1.pdf
|-- data/
|   |-- problem1.csv
|   |-- problem2.csv
|   |-- problem3.csv
|   |-- problem4.csv
|   `-- problem5.csv
|-- figures/
|   |-- problem1_distribution.png
|   |-- problem2_scatter.png
|   |-- problem2_residuals.png
|   |-- problem3_pairs.png
|   |-- problem4_conditional.png
|   |-- problem5_series.png
|   |-- problem5_acf.png
|   `-- problem5_pacf.png
|-- results/
|   `-- results.txt
|-- code/
|   `-- assignment1.py
|-- report.md
`-- README.md
```

`problem1_distribution.png` is an optional diagnostic. Every plot explicitly
requested by the assignment is also generated. The Problem 2 scatter, Problem 3
pair plots, and Problem 5 series/ACF/PACF diagnostics are generated in the code
before their corresponding model fits or correlation calculations.

## Statistical conventions

### Variance and covariance

- Problem 1 sample variance uses `numpy.var(..., ddof=1)`, the `n-1`
  denominator.
- Problem 4 sample covariance and the bucket-defining sample standard deviation
  use `ddof=1`.
- The fitted Normal regression error scale in Problem 2 is a maximum-likelihood
  estimate, so `sigma = sqrt(SSE/n)`. This differs from the conventional OLS
  residual variance used for standard errors, which divides SSE by `n-2` for an
  intercept and one slope.

### Skewness and kurtosis

- Skewness uses `scipy.stats.skew(bias=False)`, which applies SciPy's
  finite-sample bias correction.
- Kurtosis uses `scipy.stats.kurtosis(fisher=True, bias=False)`. `fisher=True`
  means excess kurtosis, so the Normal reference value is zero. Raw kurtosis is
  printed only as a check and equals excess kurtosis plus three.

### Student-t regression

The fitted error is parameterized exactly as:

```python
error ~ scipy.stats.t(df=nu, loc=0, scale=scale)
```

`scale` is SciPy's t-distribution scale parameter, not the error standard
deviation. The optimizer uses `log(scale)` and `log(nu-2)`, which enforces
`scale > 0` and `nu > 2`. It uses a fixed deterministic set of starting values
for `nu` and retains the converged solution with the largest likelihood.

### AICc

The script calculates AIC and AICc manually:

```text
AIC  = 2*k - 2*log(L)
AICc = AIC + (2*k^2 + 2*k)/(n-k-1)
```

For Problem 2, `k=3` for the Normal regression (`alpha`, `beta`, `sigma`) and
`k=4` for the Student-t regression (`alpha`, `beta`, `scale`, `nu`). For Problem
5, `k` includes the constant, every AR or MA coefficient, and the innovation
variance.

### ACF, PACF, and AR/MA models

- The ACF uses `statsmodels.tsa.stattools.acf(adjusted=False, fft=True)`.
- The PACF uses `statsmodels.tsa.stattools.pacf(method="ywmle")`.
- Both diagnostic plots use the assignment's constant significance band
  `+/- 1.96/sqrt(n)`.
- Every AR/MA candidate uses `statsmodels.tsa.arima.model.ARIMA` with `d=0`,
  `trend="c"`, exact Gaussian state-space likelihood, and explicit stationary
  initialization. Stationarity is enforced for AR terms and invertibility is
  enforced for MA terms. In this statsmodels parameterization, the reported
  `const` is the constant trend (mean-level) parameter, not the intercept in an
  AR recursion.
- The same 500 observations enter every candidate. The script reports the
  likelihood, parameter count, AIC, and manually calculated AICc for each fit.
