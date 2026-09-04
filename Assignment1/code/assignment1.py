"""Reproduce every numerical result and figure required for Assignment 1.

Run this file from the repository root with:

    python Assignment1/code/assignment1.py

All paths are resolved relative to this file, so the command also works from a
different current working directory.  The calculations intentionally spell out
statistical conventions where libraries commonly differ.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# The non-interactive backend makes figure generation reproducible on machines
# without a display (for example, a grader's CI environment).
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ASSIGNMENT_DIR / "data"
FIGURE_DIR = ASSIGNMENT_DIR / "figures"
RESULTS_DIR = ASSIGNMENT_DIR / "results"
RESULTS_FILE = RESULTS_DIR / "results.txt"

EXPECTED_COLUMNS = {
    1: ["x"],
    2: ["x", "y"],
    3: ["x1", "x2", "x3", "x4"],
    4: ["x1", "x2"],
    5: ["x"],
}

# A compact, consistent figure style suitable for insertion into a report.
plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


class Reporter:
    """Print results to the console while retaining the same text for a file."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def line(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def section(self, problem_number: int) -> None:
        if self.lines:
            self.line()
        title = f"PROBLEM {problem_number}"
        self.line(title)
        self.line("-" * len(title))

    def save(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def load_problem(problem_number: int) -> pd.DataFrame:
    """Load and strictly validate one assignment CSV.

    The assignment specifies the complete schema for each file.  Failing fast
    prevents a misspelled or missing column from silently changing a result.
    """

    path = DATA_DIR / f"problem{problem_number}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required data file is missing: {path}")

    frame = pd.read_csv(path)
    expected = EXPECTED_COLUMNS[problem_number]
    if list(frame.columns) != expected:
        raise ValueError(
            f"Unexpected columns in {path.name}: found {list(frame.columns)}, "
            f"expected {expected}"
        )
    if frame.empty:
        raise ValueError(f"Required data file is empty: {path}")

    # Convert explicitly so non-numeric text cannot be silently omitted by a
    # statistical function.
    frame = frame.apply(pd.to_numeric, errors="raise")
    if frame.isna().any().any():
        counts = frame.isna().sum().to_dict()
        raise ValueError(f"Missing values found in {path.name}: {counts}")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values found in {path.name}")
    return frame


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save and close a figure using one consistent output convention."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    destination = FIGURE_DIR / filename
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def aicc(log_likelihood: float, parameter_count: int, sample_size: int) -> tuple[float, float]:
    """Return AIC and the assignment's finite-sample AIC correction."""

    if sample_size <= parameter_count + 1:
        raise ValueError("AICc is undefined when n <= k + 1")
    aic = 2.0 * parameter_count - 2.0 * log_likelihood
    correction = (2.0 * parameter_count**2 + 2.0 * parameter_count) / (
        sample_size - parameter_count - 1
    )
    return aic, aic + correction


def run_problem_1(frame: pd.DataFrame, report: Reporter) -> dict[str, float]:
    """Compute sample moments and assess the fitted Normal's left tail."""

    report.section(1)
    report.line(f"Data check: columns={list(frame.columns)}, n={len(frame)}, missing=0")
    x = frame["x"].to_numpy(dtype=float)
    n = x.size

    # np.var(ddof=1) uses the unbiased n-1 denominator requested in the prompt.
    sample_mean = float(np.mean(x))
    sample_variance = float(np.var(x, ddof=1))
    sample_sd = float(np.sqrt(sample_variance))

    # scipy.stats bias=False applies its finite-sample bias corrections.
    # fisher=True reports excess kurtosis, so a Normal distribution has value 0.
    sample_skewness = float(stats.skew(x, bias=False))
    excess_kurtosis = float(stats.kurtosis(x, fisher=True, bias=False))
    raw_kurtosis = excess_kurtosis + 3.0

    normal_quantile_01 = float(stats.norm.ppf(0.01, loc=sample_mean, scale=sample_sd))
    actual_below = int(np.count_nonzero(x < normal_quantile_01))
    expected_below = 0.01 * n
    actual_percent = 100.0 * actual_below / n

    report.line("Moment conventions:")
    report.line("  Variance: sample variance with ddof=1 (n-1 denominator).")
    report.line("  Skewness: scipy.stats.skew(bias=False).")
    report.line("  Kurtosis: scipy.stats.kurtosis(fisher=True, bias=False), i.e. excess kurtosis.")
    report.line(f"Sample mean: {sample_mean:.10f}")
    report.line(f"Sample variance: {sample_variance:.10f}")
    report.line(f"Sample standard deviation: {sample_sd:.10f}")
    report.line(f"Bias-corrected standardized skewness: {sample_skewness:.10f}")
    report.line(f"Bias-corrected excess kurtosis: {excess_kurtosis:.10f}")
    report.line(f"Raw kurtosis (excess + 3, verification only): {raw_kurtosis:.10f}")
    report.line("Fitted Normal by moment matching:")
    report.line(f"  location (sample mean): {sample_mean:.10f}")
    report.line(f"  variance (sample variance): {sample_variance:.10f}")
    report.line(f"  scale (sqrt of sample variance): {sample_sd:.10f}")
    report.line(f"Fitted Normal 1% quantile: {normal_quantile_01:.10f}")
    report.line(f"Actual observations below fitted 1% quantile: {actual_below}")
    report.line(f"Expected observations under fitted Normal (n * 0.01): {expected_below:.2f}")
    report.line(f"Actual empirical percentage below threshold: {actual_percent:.4f}%")

    # This optional figure is useful for checking the tail mismatch visually.
    grid = np.linspace(float(np.min(x)), float(np.max(x)), 600)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(x, bins="fd", density=True, color="#6baed6", alpha=0.65, edgecolor="white")
    ax.plot(
        grid,
        stats.norm.pdf(grid, loc=sample_mean, scale=sample_sd),
        color="#cb181d",
        linewidth=2.0,
        label="Moment-matched Normal",
    )
    ax.axvline(
        normal_quantile_01,
        color="#54278f",
        linestyle="--",
        linewidth=1.8,
        label="Fitted Normal 1% quantile",
    )
    ax.set(title="Problem 1: Sample distribution and fitted Normal", xlabel="x", ylabel="Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    figure_path = save_figure(fig, "problem1_distribution.png")
    report.line(f"Figure saved: {figure_path.relative_to(ASSIGNMENT_DIR)}")

    return {
        "mean": sample_mean,
        "variance": sample_variance,
        "skewness": sample_skewness,
        "excess_kurtosis": excess_kurtosis,
        "normal_q01": normal_quantile_01,
        "actual_below": float(actual_below),
        "expected_below": expected_below,
    }


@dataclass(frozen=True)
class StudentTFit:
    alpha: float
    beta: float
    scale: float
    nu: float
    log_likelihood: float
    converged: bool
    optimizer_message: str


def fit_student_t_regression(x: np.ndarray, y: np.ndarray, ols_params: np.ndarray) -> StudentTFit:
    """Fit y = alpha + beta*x + t-error by deterministic multi-start MLE.

    The SciPy parameterization is used directly:
        error ~ scipy.stats.t(df=nu, loc=0, scale=scale)

    Optimizing log(scale) and log(nu - 2) enforces scale > 0 and nu > 2.
    Fixed starting degrees of freedom make the multi-start search deterministic.
    """

    alpha_start, beta_start = map(float, ols_params)
    residual_start = y - alpha_start - beta_start * x
    scale_start = max(float(np.std(residual_start, ddof=0)), np.finfo(float).eps)

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float]:
        alpha, beta, log_scale, log_nu_minus_two = parameters
        return (
            float(alpha),
            float(beta),
            float(np.exp(log_scale)),
            float(2.0 + np.exp(log_nu_minus_two)),
        )

    def negative_log_likelihood(parameters: np.ndarray) -> float:
        alpha, beta, scale, nu = unpack(parameters)
        residuals = y - alpha - beta * x
        value = -np.sum(stats.t.logpdf(residuals, df=nu, loc=0.0, scale=scale))
        return float(value) if np.isfinite(value) else np.inf

    # Bounds on transformed parameters prevent numerical overflow without
    # materially restricting any plausible fit for this data.
    bounds = [
        (None, None),
        (None, None),
        (np.log(1e-10), np.log(1e10)),
        (np.log(1e-8), np.log(1e6)),
    ]
    candidates = []
    for starting_nu in (3.0, 5.0, 10.0, 30.0):
        start = np.array(
            [alpha_start, beta_start, np.log(scale_start), np.log(starting_nu - 2.0)]
        )
        result = minimize(
            negative_log_likelihood,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 20_000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if np.isfinite(result.fun):
            candidates.append(result)

    if not candidates:
        raise RuntimeError("Student-t regression optimization produced no finite candidate")
    best = min(candidates, key=lambda result: float(result.fun))
    if not best.success:
        raise RuntimeError(f"Student-t regression failed to converge: {best.message}")

    alpha, beta, scale, nu = unpack(best.x)
    return StudentTFit(
        alpha=alpha,
        beta=beta,
        scale=scale,
        nu=nu,
        log_likelihood=-float(best.fun),
        converged=bool(best.success),
        optimizer_message=str(best.message),
    )


def run_problem_2(frame: pd.DataFrame, report: Reporter) -> dict[str, float]:
    """Fit OLS, Normal-error MLE, and Student-t-error MLE regressions."""

    report.section(2)
    report.line(f"Data check: columns={list(frame.columns)}, n={len(frame)}, missing=0")
    x = frame["x"].to_numpy(dtype=float)
    y = frame["y"].to_numpy(dtype=float)
    n = x.size

    # Required pre-fit visualization: it is generated before any model below.
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(x, y, s=26, alpha=0.72, color="#2171b5", edgecolors="none")
    ax.set(title="Problem 2: y against x (before fitting)", xlabel="x", ylabel="y")
    ax.grid(alpha=0.18)
    scatter_path = save_figure(fig, "problem2_scatter.png")
    report.line(f"Pre-fit scatter figure saved before model fitting: {scatter_path.relative_to(ASSIGNMENT_DIR)}")

    design = sm.add_constant(x, has_constant="add")
    ols_result = sm.OLS(y, design).fit(cov_type="nonrobust")
    ols_alpha, ols_beta = map(float, ols_result.params)
    ols_se_alpha, ols_se_beta = map(float, ols_result.bse)

    # For Gaussian regression, the MLE of alpha and beta is OLS.  The MLE of
    # sigma uses SSE/n (ddof=0), not the OLS residual variance SSE/(n-2).
    normal_alpha, normal_beta = ols_alpha, ols_beta
    normal_residuals = y - normal_alpha - normal_beta * x
    normal_sigma = float(np.sqrt(np.mean(normal_residuals**2)))
    normal_log_likelihood = float(
        np.sum(stats.norm.logpdf(normal_residuals, loc=0.0, scale=normal_sigma))
    )

    student_fit = fit_student_t_regression(x, y, ols_result.params)
    student_residuals = y - student_fit.alpha - student_fit.beta * x

    normal_k = 3  # alpha, beta, sigma
    student_k = 4  # alpha, beta, scale, nu
    normal_aic, normal_aicc = aicc(normal_log_likelihood, normal_k, n)
    student_aic, student_aicc = aicc(student_fit.log_likelihood, student_k, n)

    normal_q95 = float(stats.norm.ppf(0.95, loc=0.0, scale=normal_sigma))
    normal_q995 = float(stats.norm.ppf(0.995, loc=0.0, scale=normal_sigma))
    student_q95 = float(stats.t.ppf(0.95, df=student_fit.nu, loc=0.0, scale=student_fit.scale))
    student_q995 = float(stats.t.ppf(0.995, df=student_fit.nu, loc=0.0, scale=student_fit.scale))

    report.line("OLS (conventional non-robust covariance; residual variance uses SSE/(n-2)):")
    report.line(f"  alpha: {ols_alpha:.10f}")
    report.line(f"  beta: {ols_beta:.10f}")
    report.line(f"  SE(alpha): {ols_se_alpha:.10f}")
    report.line(f"  SE(beta): {ols_se_beta:.10f}")
    report.line("Normal-error maximum likelihood:")
    report.line("  Parameterization: error ~ Normal(loc=0, scale=sigma); sigma MLE uses SSE/n.")
    report.line(f"  alpha: {normal_alpha:.10f}")
    report.line(f"  beta: {normal_beta:.10f}")
    report.line(f"  sigma: {normal_sigma:.10f}")
    report.line(f"  maximized log-likelihood: {normal_log_likelihood:.10f}")
    report.line("Student-t-error maximum likelihood:")
    report.line("  Parameterization: error ~ scipy.stats.t(df=nu, loc=0, scale=scale).")
    report.line("  Constraints: scale > 0 and nu > 2 via log transformations.")
    report.line(f"  alpha: {student_fit.alpha:.10f}")
    report.line(f"  beta: {student_fit.beta:.10f}")
    report.line(f"  scale (not the error SD): {student_fit.scale:.10f}")
    report.line(f"  degrees of freedom nu: {student_fit.nu:.10f}")
    report.line(f"  maximized log-likelihood: {student_fit.log_likelihood:.10f}")
    report.line(f"  optimizer converged: {student_fit.converged}")
    report.line(f"  optimizer message: {student_fit.optimizer_message}")
    report.line("AICc convention: AIC=2k-2log(L); AICc=AIC+(2k^2+2k)/(n-k-1).")
    report.line(f"  Normal: k={normal_k}, AIC={normal_aic:.10f}, AICc={normal_aicc:.10f}")
    report.line(f"  Student-t: k={student_k}, AIC={student_aic:.10f}, AICc={student_aicc:.10f}")
    report.line(f"  AICc difference (Normal - Student-t): {normal_aicc - student_aicc:.10f}")
    preferred = "Normal" if normal_aicc < student_aicc else "Student-t"
    report.line(f"  Minimum-AICc model: {preferred}")
    report.line("Positive upper-tail quantiles of the zero-centered fitted error distributions:")
    report.line(f"  95% Normal: {normal_q95:.10f}")
    report.line(f"  95% Student-t: {student_q95:.10f}")
    report.line(f"  99.5% Normal: {normal_q995:.10f}")
    report.line(f"  99.5% Student-t: {student_q995:.10f}")

    # The OLS residual histogram provides a common empirical reference.  The
    # fitted regression lines differ only slightly; both zero-centered fitted
    # error densities are overlaid using their own fitted error parameters.
    ols_residuals = np.asarray(ols_result.resid, dtype=float)
    lower = min(float(np.min(ols_residuals)), float(stats.t.ppf(0.002, student_fit.nu) * student_fit.scale))
    upper = max(float(np.max(ols_residuals)), float(stats.t.ppf(0.998, student_fit.nu) * student_fit.scale))
    grid = np.linspace(lower, upper, 800)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(
        ols_residuals,
        bins="fd",
        density=True,
        color="#9ecae1",
        alpha=0.68,
        edgecolor="white",
        label="OLS residuals",
    )
    ax.plot(
        grid,
        stats.norm.pdf(grid, loc=0.0, scale=normal_sigma),
        color="#cb181d",
        linewidth=2.0,
        label="Fitted Normal error",
    )
    ax.plot(
        grid,
        stats.t.pdf(grid, df=student_fit.nu, loc=0.0, scale=student_fit.scale),
        color="#238b45",
        linewidth=2.0,
        label="Fitted Student-t error",
    )
    ax.set(title="Problem 2: Residual distribution and fitted error densities", xlabel="Residual", ylabel="Density")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    residual_path = save_figure(fig, "problem2_residuals.png")
    report.line(f"Residual figure saved: {residual_path.relative_to(ASSIGNMENT_DIR)}")

    return {
        "ols_alpha": ols_alpha,
        "ols_beta": ols_beta,
        "ols_se_alpha": ols_se_alpha,
        "ols_se_beta": ols_se_beta,
        "normal_sigma": normal_sigma,
        "normal_aicc": normal_aicc,
        "student_alpha": student_fit.alpha,
        "student_beta": student_fit.beta,
        "student_scale": student_fit.scale,
        "student_nu": student_fit.nu,
        "student_aicc": student_aicc,
        "normal_q95": normal_q95,
        "student_q95": student_q95,
        "normal_q995": normal_q995,
        "student_q995": student_q995,
    }


def run_problem_3(frame: pd.DataFrame, report: Reporter) -> dict[str, object]:
    """Plot every pair, then compare Pearson and Spearman correlations."""

    report.section(3)
    report.line(f"Data check: columns={list(frame.columns)}, n={len(frame)}, missing=0")
    columns = list(frame.columns)
    pairs = list(itertools.combinations(columns, 2))

    # Required pre-fit diagnostic: all six pair plots are generated before the
    # correlation matrices are computed below.
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    for ax, (left, right) in zip(axes.flat, pairs):
        ax.scatter(frame[left], frame[right], s=17, alpha=0.62, color="#2171b5", edgecolors="none")
        ax.set(xlabel=left, ylabel=right, title=f"{left} vs {right}")
        ax.grid(alpha=0.16)
    fig.suptitle("Problem 3: Pairwise scatter plots (before correlations)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pairs_path = save_figure(fig, "problem3_pairs.png")
    report.line(f"Pairwise scatter figure saved before correlations: {pairs_path.relative_to(ASSIGNMENT_DIR)}")

    pearson = frame.corr(method="pearson")
    spearman = frame.corr(method="spearman")
    gaps = sorted(
        [
            (left, right, abs(float(pearson.loc[left, right]) - float(spearman.loc[left, right])))
            for left, right in pairs
        ],
        key=lambda item: item[2],
        reverse=True,
    )

    matrix_format = lambda value: f"{value: .6f}"
    report.line("Pearson correlation matrix:")
    for line in pearson.to_string(float_format=matrix_format).splitlines():
        report.line(f"  {line}")
    report.line("Spearman correlation matrix:")
    for line in spearman.to_string(float_format=matrix_format).splitlines():
        report.line(f"  {line}")
    report.line("Absolute Pearson-Spearman gaps, largest to smallest:")
    for rank, (left, right, gap) in enumerate(gaps, start=1):
        report.line(
            f"  {rank}. {left}/{right}: Pearson={pearson.loc[left, right]:.10f}, "
            f"Spearman={spearman.loc[left, right]:.10f}, gap={gap:.10f}"
        )
    largest_left, largest_right, largest_gap = gaps[0]
    report.line(f"Largest-gap pair: {largest_left}/{largest_right} (gap={largest_gap:.10f})")

    return {
        "pearson": pearson,
        "spearman": spearman,
        "largest_pair": (largest_left, largest_right),
        "largest_gap": largest_gap,
    }


def run_problem_4(frame: pd.DataFrame, report: Reporter) -> dict[str, object]:
    """Compute and evaluate the Normal conditional model for x2 given x1."""

    report.section(4)
    report.line(f"Data check: columns={list(frame.columns)}, n={len(frame)}, missing=0")
    x1 = frame["x1"].to_numpy(dtype=float)
    x2 = frame["x2"].to_numpy(dtype=float)
    n = x1.size

    mu1, mu2 = float(np.mean(x1)), float(np.mean(x2))
    # np.cov(ddof=1) uses the requested n-1 sample covariance convention.
    covariance = np.cov(np.vstack([x1, x2]), ddof=1)
    sigma11 = float(covariance[0, 0])
    sigma12 = float(covariance[0, 1])
    sigma21 = float(covariance[1, 0])
    sigma22 = float(covariance[1, 1])

    conditional_variance = sigma22 - sigma21 * sigma12 / sigma11
    remaining_variance_factor = conditional_variance / sigma22
    variance_reduction = 1.0 - remaining_variance_factor
    beta_conditional = sigma21 / sigma11
    conditional_sd = float(np.sqrt(conditional_variance))

    conditional_mean = mu2 + beta_conditional * (x1 - mu1)
    lower_band = conditional_mean - 1.96 * conditional_sd
    upper_band = conditional_mean + 1.96 * conditional_sd
    inside = (x2 >= lower_band) & (x2 <= upper_band)

    # A simple OLS slope with an intercept must equal Cov(x1,x2)/Var(x1), up to
    # floating-point rounding.
    ols_check = sm.OLS(x2, sm.add_constant(x1, has_constant="add")).fit()
    ols_intercept, ols_slope = map(float, ols_check.params)
    expected_intercept = mu2 - beta_conditional * mu1

    overall_inside = int(np.count_nonzero(inside))
    overall_coverage = overall_inside / n
    x1_sample_sd = float(np.std(x1, ddof=1))
    distance = np.abs(x1 - mu1)
    bucket_masks = [
        distance <= x1_sample_sd,
        (distance > x1_sample_sd) & (distance <= 2.0 * x1_sample_sd),
        distance > 2.0 * x1_sample_sd,
    ]
    bucket_labels = [
        "|x1-mu1| <= 1*sd(x1)",
        "1*sd(x1) < |x1-mu1| <= 2*sd(x1)",
        "|x1-mu1| > 2*sd(x1)",
    ]
    bucket_results: list[tuple[str, int, int, float]] = []
    for label, mask in zip(bucket_labels, bucket_masks):
        count = int(np.count_nonzero(mask))
        inside_count = int(np.count_nonzero(inside & mask))
        coverage = inside_count / count if count else float("nan")
        bucket_results.append((label, count, inside_count, coverage))

    report.line("Convention: means use 1/n; covariance and sd(x1) use ddof=1 (n-1 denominator).")
    report.line(f"mu1: {mu1:.10f}")
    report.line(f"mu2: {mu2:.10f}")
    report.line("Sample covariance matrix [x1, x2]:")
    report.line(f"  [[{sigma11:.10f}, {sigma12:.10f}],")
    report.line(f"   [{sigma21:.10f}, {sigma22:.10f}]]")
    report.line("Conditioning x2 on x1: Sigma11=Var(x1), Sigma22=Var(x2), Sigma21=Cov(x2,x1).")
    report.line(f"Conditional variance Sigma22 - Sigma21^2/Sigma11: {conditional_variance:.10f}")
    report.line(f"Conditional standard deviation: {conditional_sd:.10f}")
    report.line(f"Remaining-variance factor conditional_variance/Sigma22: {remaining_variance_factor:.10f}")
    report.line(f"Percentage variance reduction: {100.0 * variance_reduction:.6f}%")
    report.line(f"Conditional-mean coefficient beta_cond=Sigma21/Sigma11: {beta_conditional:.10f}")
    report.line(f"OLS slope from x2 ~ intercept + x1: {ols_slope:.10f}")
    report.line(f"Absolute slope verification difference: {abs(beta_conditional - ols_slope):.3e}")
    report.line(f"Conditional-mean intercept mu2-beta_cond*mu1: {expected_intercept:.10f}")
    report.line(f"OLS intercept verification: {ols_intercept:.10f}")
    report.line("95% band convention: conditional mean +/- 1.96 * conditional SD (constant width).")
    report.line(
        f"Overall band coverage: {overall_inside}/{n} = {100.0 * overall_coverage:.6f}%"
    )
    report.line("Coverage by distance-from-mean bucket (sd(x1) uses ddof=1):")
    for index, (label, count, inside_count, coverage) in enumerate(bucket_results, start=1):
        report.line(
            f"  Bucket {index} [{label}]: n={count}, inside={inside_count}, "
            f"coverage={100.0 * coverage:.6f}%"
        )

    grid = np.linspace(float(np.min(x1)), float(np.max(x1)), 500)
    grid_mean = mu2 + beta_conditional * (grid - mu1)
    grid_lower = grid_mean - 1.96 * conditional_sd
    grid_upper = grid_mean + 1.96 * conditional_sd
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.scatter(x1, x2, s=16, alpha=0.44, color="#3182bd", edgecolors="none", label="Observations")
    ax.plot(grid, grid_mean, color="#cb181d", linewidth=2.2, label="Conditional expectation")
    ax.plot(grid, grid_lower, color="#54278f", linestyle="--", linewidth=1.5, label="95% band")
    ax.plot(grid, grid_upper, color="#54278f", linestyle="--", linewidth=1.5)
    ax.fill_between(grid, grid_lower, grid_upper, color="#9e9ac8", alpha=0.13)
    ax.set(title="Problem 4: Conditional expectation and constant-width 95% band", xlabel="x1", ylabel="x2")
    ax.legend(frameon=False)
    ax.grid(alpha=0.18)
    conditional_path = save_figure(fig, "problem4_conditional.png")
    report.line(f"Conditional-distribution figure saved: {conditional_path.relative_to(ASSIGNMENT_DIR)}")

    return {
        "covariance": covariance,
        "conditional_variance": conditional_variance,
        "remaining_variance_factor": remaining_variance_factor,
        "variance_reduction": variance_reduction,
        "beta_conditional": beta_conditional,
        "overall_coverage": overall_coverage,
        "bucket_results": bucket_results,
    }


def correlation_stem_figure(values: np.ndarray, band: float, title: str, ylabel: str) -> plt.Figure:
    """Create an ACF/PACF stem plot with the assignment's constant band."""

    lags = np.arange(values.size)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    markerline, stemlines, baseline = ax.stem(lags, values, basefmt=" ")
    plt.setp(markerline, marker="o", markersize=4.5, color="#2171b5")
    plt.setp(stemlines, color="#2171b5", linewidth=1.3)
    baseline.set_visible(False)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axhline(band, color="#cb181d", linestyle="--", linewidth=1.2, label=r"$\pm 1.96/\sqrt{n}$")
    ax.axhline(-band, color="#cb181d", linestyle="--", linewidth=1.2)
    ax.set(title=title, xlabel="Lag", ylabel=ylabel, xlim=(-0.8, values.size - 0.2))
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.18)
    return fig


@dataclass(frozen=True)
class ArmaCandidate:
    name: str
    coefficients: dict[str, float]
    log_likelihood: float
    parameter_count: int
    nobs: int
    aic: float
    aicc: float
    warnings: tuple[str, ...]


def fit_arma_candidate(x: np.ndarray, name: str, order: tuple[int, int, int]) -> ArmaCandidate:
    """Fit one stationary/invertible ARIMA candidate under a common convention."""

    model = ARIMA(
        x,
        order=order,
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
    )
    # Explicit stationary initialization is used for every candidate rather than
    # allowing model-specific automatic initialization choices.
    model.initialize_stationary()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit()

    warning_messages = tuple(
        f"{item.category.__name__}: {item.message}"
        for item in caught
        if issubclass(item.category, (Warning,))
    )
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(f"{name} emitted a convergence warning: {warning_messages}")
    if not bool(result.mle_retvals.get("converged", True)):
        raise RuntimeError(f"{name} optimizer did not converge: {result.mle_retvals}")

    parameter_count = len(result.params)
    manual_aic, manual_aicc = aicc(float(result.llf), parameter_count, int(result.nobs))
    coefficients = {
        parameter_name: float(value)
        for parameter_name, value in zip(result.param_names, result.params)
    }
    return ArmaCandidate(
        name=name,
        coefficients=coefficients,
        log_likelihood=float(result.llf),
        parameter_count=parameter_count,
        nobs=int(result.nobs),
        aic=manual_aic,
        aicc=manual_aicc,
        warnings=warning_messages,
    )


def run_problem_5(frame: pd.DataFrame, report: Reporter) -> dict[str, object]:
    """Generate pre-fit diagnostics, then compare AR and MA orders by AICc."""

    report.section(5)
    report.line(f"Data check: columns={list(frame.columns)}, n={len(frame)}, missing=0")
    x = frame["x"].to_numpy(dtype=float)
    n = x.size
    max_lag = 40
    significance_band = 1.96 / np.sqrt(n)

    # All three diagnostics are computed and saved before fitting any AR/MA
    # candidate.  ACF uses its usual 1/n normalization (adjusted=False).  PACF
    # uses Yule-Walker without the finite-n adjustment (method='ywmle').
    acf_values = acf(x, nlags=max_lag, fft=True, adjusted=False)
    pacf_values = pacf(x, nlags=max_lag, method="ywmle")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(np.arange(1, n + 1), x, color="#2171b5", linewidth=1.0)
    ax.set(title="Problem 5: Time series (before fitting)", xlabel="Observation", ylabel="x")
    ax.grid(alpha=0.18)
    series_path = save_figure(fig, "problem5_series.png")

    acf_path = save_figure(
        correlation_stem_figure(acf_values, significance_band, "Problem 5: Sample ACF (before fitting)", "ACF"),
        "problem5_acf.png",
    )
    pacf_path = save_figure(
        correlation_stem_figure(pacf_values, significance_band, "Problem 5: Sample PACF (before fitting)", "PACF"),
        "problem5_pacf.png",
    )

    report.line("Pre-fit diagnostics were generated before fitting candidate models.")
    report.line(f"  Series figure: {series_path.relative_to(ASSIGNMENT_DIR)}")
    report.line(f"  ACF figure: {acf_path.relative_to(ASSIGNMENT_DIR)}")
    report.line(f"  PACF figure: {pacf_path.relative_to(ASSIGNMENT_DIR)}")
    report.line("ACF convention: statsmodels acf(adjusted=False, fft=True).")
    report.line("PACF convention: statsmodels pacf(method='ywmle').")
    report.line(f"Significance band: +/- 1.96/sqrt(n) = +/- {significance_band:.10f}")
    report.line("First 10 nonzero-lag sample ACF values:")
    for lag in range(1, 11):
        report.line(f"  lag {lag:2d}: {acf_values[lag]: .10f}")
    report.line("First 10 nonzero-lag sample PACF values:")
    for lag in range(1, 11):
        report.line(f"  lag {lag:2d}: {pacf_values[lag]: .10f}")

    significant_acf = [lag for lag in range(1, 21) if abs(acf_values[lag]) > significance_band]
    significant_pacf = [lag for lag in range(1, 21) if abs(pacf_values[lag]) > significance_band]
    report.line(f"Significant ACF lags among 1-20 by this band: {significant_acf}")
    report.line(f"Significant PACF lags among 1-20 by this band: {significant_pacf}")

    # This rule uses only the already-computed ACF/PACF diagnostics.  It is
    # evaluated before the model loop, so fitted likelihoods/AICc cannot affect
    # the stated pre-fit suggestion.
    if (
        abs(pacf_values[1]) > significance_band
        and abs(pacf_values[2]) > significance_band
        and np.all(np.abs(pacf_values[3:9]) <= significance_band)
    ):
        prefit_suggestion = "AR(2)"
        prefit_reason = (
            "PACF has strong lags 1 and 2 followed by lags 3-8 inside the band, "
            "while the ACF decays/oscillates; isolated later spikes can occur in sampling."
        )
    else:
        prefit_suggestion = "No unambiguous automatic cutoff"
        prefit_reason = "The early-lag cutoff rule was not satisfied; inspect the saved ACF/PACF figures."
    report.line(f"Pre-fit cutoff/decay suggestion (ACF/PACF only): {prefit_suggestion}")
    report.line(f"Pre-fit diagnostic basis: {prefit_reason}")

    candidate_specs = [
        ("AR(1)", (1, 0, 0)),
        ("AR(2)", (2, 0, 0)),
        ("AR(3)", (3, 0, 0)),
        ("MA(1)", (0, 0, 1)),
        ("MA(2)", (0, 0, 2)),
        ("MA(3)", (0, 0, 3)),
    ]
    candidates = [fit_arma_candidate(x, name, order) for name, order in candidate_specs]

    report.line("AR/MA fitting convention:")
    report.line("  statsmodels.tsa.arima.model.ARIMA with d=0 and trend='c' for every model.")
    report.line("  The reported 'const' is statsmodels' constant trend (mean-level) parameter, not an AR-recursion intercept.")
    report.line("  Exact Gaussian state-space log likelihood with explicit stationary initialization.")
    report.line("  Stationarity enforced for AR terms; invertibility enforced for MA terms.")
    report.line("  k includes the constant, all AR/MA coefficients, and innovation variance sigma2.")
    report.line("  AIC and AICc are calculated manually from llf, k, and result.nobs.")
    report.line("Candidate fits:")
    for candidate in candidates:
        report.line(f"  {candidate.name}")
        for parameter_name, value in candidate.coefficients.items():
            report.line(f"    {parameter_name}: {value:.10f}")
        report.line(f"    log-likelihood: {candidate.log_likelihood:.10f}")
        report.line(f"    k: {candidate.parameter_count}")
        report.line(f"    nobs: {candidate.nobs}")
        report.line(f"    AIC: {candidate.aic:.10f}")
        report.line(f"    AICc: {candidate.aicc:.10f}")
        report.line(f"    captured warnings: {len(candidate.warnings)}")
        for warning_message in candidate.warnings:
            report.line(f"      {warning_message}")

    selected = min(candidates, key=lambda candidate: candidate.aicc)
    report.line(f"Minimum-AICc model: {selected.name} (AICc={selected.aicc:.10f})")
    report.line(f"Pre-fit suggestion matched AICc selection: {prefit_suggestion == selected.name}")

    by_name = {candidate.name: candidate for candidate in candidates}
    ar2 = by_name["AR(2)"]
    ar3 = by_name["AR(3)"]
    report.line("Focused AR(2) versus AR(3) comparison:")
    report.line(
        "  AR(2) AR coefficients: "
        + ", ".join(
            f"{name}={value:.10f}" for name, value in ar2.coefficients.items() if name.startswith("ar.")
        )
    )
    report.line(
        "  AR(3) AR coefficients: "
        + ", ".join(
            f"{name}={value:.10f}" for name, value in ar3.coefficients.items() if name.startswith("ar.")
        )
    )
    report.line(f"  AR(3) third-lag coefficient: {ar3.coefficients['ar.L3']:.10f}")
    report.line(f"  AR(2) log-likelihood: {ar2.log_likelihood:.10f}")
    report.line(f"  AR(3) log-likelihood: {ar3.log_likelihood:.10f}")
    report.line(f"  AR(2) AICc: {ar2.aicc:.10f}")
    report.line(f"  AR(3) AICc: {ar3.aicc:.10f}")

    return {
        "significance_band": significance_band,
        "acf": acf_values,
        "pacf": pacf_values,
        "prefit_suggestion": prefit_suggestion,
        "candidates": candidates,
        "selected": selected.name,
    }


def main() -> None:
    """Load all data, run all problems, and save a transcript of the output."""

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all files before starting calculations.  If any is missing or invalid,
    # the run stops rather than producing a partial result set.
    frames = {problem: load_problem(problem) for problem in range(1, 6)}

    report = Reporter()
    run_problem_1(frames[1], report)
    run_problem_2(frames[2], report)
    run_problem_3(frames[3], report)
    run_problem_4(frames[4], report)
    run_problem_5(frames[5], report)

    report.line()
    report.line("REPRODUCIBILITY INFORMATION")
    report.line("---------------------------")
    report.line(f"NumPy version: {np.__version__}")
    report.line(f"pandas version: {pd.__version__}")
    report.line(f"SciPy version: {scipy.__version__}")
    report.line(f"statsmodels version: {statsmodels.__version__}")
    report.line(f"matplotlib version: {matplotlib.__version__}")
    report.save(RESULTS_FILE)
    print(f"\nComplete numerical output saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
