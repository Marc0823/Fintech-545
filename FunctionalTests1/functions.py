import numpy as np
import pandas as pd
from scipy import stats, optimize, special


# Covariance and correlation
def missing_cov(x, skip_missing=True, correlation=False):
    x = np.asarray(x, dtype=float)
    if skip_missing:
        x = x[~np.isnan(x).any(axis=1)]
    out = np.empty((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(i + 1):
            pair = x[:, [i, j]]
            pair = pair[~np.isnan(pair).any(axis=1)]
            if len(pair) < 2:
                raise ValueError("Not enough observations for covariance")
            centered = pair - pair.mean(axis=0)
            c = centered.T @ centered / (len(pair) - 1)
            value = c[0, 1]
            if correlation:
                value /= np.sqrt(c[0, 0] * c[1, 1])
            out[i, j] = out[j, i] = value
    return out


def missing_corr(x, skip_missing=True):
    return missing_cov(x, skip_missing, correlation=True)


def exponential_weights(n, lam):
    # newest observation gets the largest weight
    w = (1 - lam) * lam ** np.arange(n - 1, -1, -1)
    return w / w.sum()


def ew_cov(x, lam):
    w = exponential_weights(len(x), lam)
    centered = np.sqrt(w)[:, None] * (x - w @ x)
    return centered.T @ centered


def ew_corr(x, lam):
    c = ew_cov(x, lam)
    d = np.diag(1 / np.sqrt(np.diag(c)))
    return d @ c @ d


def mixed_ew_cov(x):
    d = np.diag(np.sqrt(np.diag(ew_cov(x, .97))))
    return d @ ew_corr(x, .94) @ d


# PSD repair: work in correlation space, then restore variances
def correlation_space(a):
    a = (a + a.T) / 2
    if np.allclose(np.diag(a), 1, atol=0, rtol=np.sqrt(np.finfo(float).eps)):
        return a.copy(), None
    sd = np.sqrt(np.diag(a))
    d = np.diag(1 / sd)
    return d @ a @ d, sd


def near_psd(a):
    c, sd = correlation_space(a)
    values, vectors = np.linalg.eigh(c)
    # set negative eigenvalues to zero
    values = np.maximum(values, 0)
    t = 1 / np.sqrt((vectors * vectors) @ values)
    b = np.diag(t) @ vectors @ np.diag(np.sqrt(values))
    out = b @ b.T
    if sd is not None:
        out = np.diag(sd) @ out @ np.diag(sd)
    return out


def higham_nearestPSD(a):
    y, sd = correlation_space(a)
    original = y.copy()
    correction = np.zeros_like(y)
    previous_distance = np.inf
    for _ in range(100):
        residual = y - correction
        values, vectors = np.linalg.eigh(residual)
        x = vectors @ np.diag(np.maximum(values, 0)) @ vectors.T
        correction = x - residual
        y = x.copy()
        np.fill_diagonal(y, 1)
        distance = np.sum((y - original) ** 2)
        if abs(distance - previous_distance) < 1e-9 and np.linalg.eigvalsh(y)[0] > -1e-9:
            if sd is not None:
                y = np.diag(sd) @ y @ np.diag(sd)
            return y
        previous_distance = distance
    raise RuntimeError("Higham did not converge in 100 iterations")


def chol_psd(a):
    a = (a + a.T) / 2
    root = np.zeros_like(a)
    for j in range(len(a)):
        pivot = a[j, j] - root[j, :j] @ root[j, :j]
        if -1e-8 <= pivot <= 0:
            pivot = 0.0
        if pivot < 0:
            raise np.linalg.LinAlgError("Negative Cholesky pivot")
        root[j, j] = np.sqrt(pivot)
        if root[j, j] != 0:
            for i in range(j + 1, len(a)):
                root[i, j] = (a[i, j] - root[i, :j] @ root[j, :j]) / root[j, j]
    if not np.allclose(root @ root.T, a, atol=1e-8, rtol=1e-10):
        raise np.linalg.LinAlgError("PSD factor does not reconstruct the matrix")
    return root


# Simulation
def simulateNormal(nsim, covariance, seed=1234, fix_method=near_psd):
    c = (covariance + covariance.T) / 2
    try:
        root = np.linalg.cholesky(c)
    except np.linalg.LinAlgError:
        try:
            root = chol_psd(c)
        except np.linalg.LinAlgError:
            root = chol_psd(fix_method(c))
    rng = np.random.Generator(np.random.PCG64(seed))
    normals = rng.standard_normal((len(c), nsim))
    return (root @ normals).T


def pca_factors(covariance, pctExp=.99):
    a = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(a)
    values, vectors = values[::-1], vectors[:, ::-1]
    total = values.sum()
    positive = np.flatnonzero(values >= 1e-8)
    if pctExp < 1:
        count = min(int(np.searchsorted(np.cumsum(values[positive]) / total, pctExp)) + 1, len(positive))
        positive = positive[:count]
    retained = values[positive]
    loadings = vectors[:, positive] @ np.diag(np.sqrt(retained))
    return loadings, retained, total


def simulate_pca(covariance, nsim, pctExp=.99, seed=1234):
    b, values, _ = pca_factors(covariance, pctExp)
    rng = np.random.Generator(np.random.PCG64(seed))
    normals = rng.standard_normal((len(values), nsim))
    return (b @ normals).T


# Returns
def return_calculate(prices, method="ARITHMETIC"):
    assets = [name for name in prices.columns if name != "Date"]
    p = prices[assets].to_numpy(dtype=float)
    ratios = p[1:] / p[:-1]
    if method == "ARITHMETIC":
        values = ratios - 1
    elif method == "LOG":
        values = np.log(ratios)
    else:
        raise ValueError("method must be ARITHMETIC or LOG")
    out = pd.DataFrame(values, columns=assets)
    out["Date"] = prices.Date.iloc[1:].to_numpy()
    return out.loc[:, prices.columns]


# Distribution fitting
def fit_normal(x):
    return {"mu": float(x.mean()), "sigma": float(x.std(ddof=1))}


def t_mle(y, design):
    # scale the data to help the optimizer; convert back afterward
    n, p = design.shape
    location, response_scale = float(y.mean()), float(y.std(ddof=1))
    z = (y - location) / response_scale
    column_scale = np.sqrt(np.mean(design * design, axis=0))
    xs = design / column_scale
    beta0 = np.linalg.lstsq(xs, z, rcond=None)[0]
    residual0 = z - xs @ beta0
    lower_log_scale = np.log(1e-6 / response_scale)

    def objective(theta):
        beta, log_scale, nu = theta[:p], theta[p], theta[p + 1]
        scale = np.exp(log_scale)
        residual = z - xs @ beta
        z2 = (residual / scale) ** 2
        log_term = np.log1p(z2 / nu)
        ll = (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2)
              - .5 * np.log(nu * np.pi) - log_scale - .5 * (nu + 1) * log_term)
        # derivatives of the negative log likelihood
        q = (nu + 1) * residual / (nu * scale * scale + residual * residual)
        grad_beta = -xs.T @ q
        grad_log_scale = n - np.sum((nu + 1) * z2 / (nu + z2))
        dll_nu = (.5 * (special.digamma((nu + 1) / 2) - special.digamma(nu / 2) - 1 / nu)
                  - .5 * log_term + .5 * (nu + 1) * z2 / (nu * (nu + z2)))
        return -float(ll.sum()), np.r_[grad_beta, grad_log_scale, -dll_nu.sum()]

    bounds = [(None, None)] * p + [(lower_log_scale, None), (2.0001, None)]
    candidates = []
    for nu0 in (3.0, 5.0, 10.0, 30.0):
        scale0 = max(np.std(residual0, ddof=1) * np.sqrt((nu0 - 2) / nu0), np.exp(lower_log_scale))
        theta0 = np.r_[beta0, np.log(scale0), nu0]
        result = optimize.minimize(objective, theta0, jac=True, method="L-BFGS-B", bounds=bounds,
                                   options={"maxiter": 10000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 100})
        if result.success and np.isfinite(result.fun):
            candidates.append(result)
    if not candidates:
        raise RuntimeError("Student-t fit did not converge")
    best = min(candidates, key=lambda r: r.fun)
    beta = best.x[:p] * response_scale / column_scale
    beta[0] += location
    sigma, nu = float(np.exp(best.x[p]) * response_scale), float(best.x[p + 1])
    return beta, sigma, nu


def fit_general_t(x):
    beta, sigma, nu = t_mle(x, np.ones((len(x), 1)))
    return {"mu": float(beta[0]), "sigma": sigma, "nu": nu}


def fit_regression_t(y, x):
    design = np.column_stack([np.ones(len(y)), x])
    beta, sigma, nu = t_mle(y, design)
    out = {"mu": 0.0, "sigma": sigma, "nu": nu, "Alpha": beta[0]}
    out.update({f"B{i}": beta[i] for i in range(1, len(beta))})
    return out


def fit_nig_moments(x):
    m, v = x.mean(), x.var(ddof=1)
    s, k = stats.skew(x, bias=True), stats.kurtosis(x, fisher=True, bias=True)
    if k <= 0 or k <= (5 / 3) * s * s:
        raise ValueError("NIG requires excess kurtosis > (5/3)*skewness^2")
    t = s * s / k
    rho2 = t / (3 - 4 * t)
    rho = np.sign(s) * np.sqrt(rho2)
    D = 3 * (1 + 4 * rho2) / k
    alpha = np.sqrt(D / (v * (1 - rho2) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1 - rho2)
    delta = D / gamma
    mu = m - delta * beta / gamma
    return {"mu": float(mu), "alpha": float(alpha), "beta": float(beta), "delta": float(delta)}


def fit_NIG_mle(x):
    def checked_optimizer(func, x0, args=(), disp=0):
        # same SciPy defaults, with a convergence check
        best, _, _, _, warning = optimize.fmin(func, x0, args=args, disp=disp, full_output=True)
        if warning != 0:
            raise RuntimeError("NIG MLE did not converge")
        return best

    a, b, mu, delta = stats.norminvgauss.fit(x, optimizer=checked_optimizer)
    # convert scipy NIG parameters to course notation
    return {"mu": float(mu), "alpha": float(a/delta), "beta": float(b/delta), "delta": float(delta)}


def AIC(loglik, k):
    return -2 * loglik + 2 * k


def AICc(loglik, k, n):
    return AIC(loglik, k) + 2 * k * (k + 1) / (n - k - 1)
