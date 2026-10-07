"""FinTech 545 Assignment 2: run with python assignment2.py.

Calculations follow Weeks 1-5 and the course library at commit
6f104966e891a3326898d51e291c8a0153b454f7. Report prose is in assignment2.qmd;
it reads results.json, so displayed results are calculated, not transcribed.
"""
from pathlib import Path
from itertools import combinations
import json
import platform
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.special import gammaln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
SEED = 545
NSIM = 100_000
RECENT_DAYS = 40  # Chosen from the return plot before fitting, not optimized.
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})
results = {"seed": SEED, "nsim": NSIM, "course_commit":
           "6f104966e891a3326898d51e291c8a0153b454f7",
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scipy": scipy.__version__,
                        "matplotlib": matplotlib.__version__}}


def moments(x):
    """Sample variance; population-standardized skew and excess kurtosis (Julia convention)."""
    x = np.asarray(x)
    centered = x - x.mean()
    m2 = np.mean(centered**2)
    return {"mean": x.mean(), "variance": x.var(ddof=1), "sd": x.std(ddof=1),
            "skewness": np.mean(centered**3)/m2**1.5,
            "excess_kurtosis": np.mean(centered**4)/m2**2 - 3}


def historical_risk(pnl, alpha=0.05):
    """Course RiskStats.jl order statistics; negative VaR means a quantile gain.

    q = average of floor(n*alpha)th and ceil(n*alpha)th ordered P&Ls (1-based).
    Every n*alpha here is an integer; ES is the mean of P&Ls at or below q.
    """
    x = np.sort(np.asarray(pnl))
    lo, hi = int(np.floor(len(x)*alpha)), int(np.ceil(len(x)*alpha))
    assert lo >= 1
    q = (x[lo-1] + x[hi-1])/2
    var, es = -q, -x[x <= q].mean()
    assert np.isfinite([var, es]).all() and es >= var - 1e-8
    return {"VaR": var, "ES": es}


def near_psd(a):
    """Course Rebonato-Jackel: clip eigenvalues at zero, restore unit diagonal."""
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, 0)
    b = vecs * np.sqrt(vals)
    b = b / np.sqrt(np.sum(b*b, axis=1))[:, None]
    return b @ b.T


def higham(a, tol=1e-12, max_iter=1000):
    """Unweighted Higham/Dykstra, as in library/simulate.jl; tighter tolerance."""
    y = a.copy()
    delta = np.zeros_like(a)
    previous_distance = np.inf
    for iteration in range(1, max_iter+1):
        r = (y-delta + (y-delta).T)/2
        vals, vecs = np.linalg.eigh(r)
        x = (vecs * np.maximum(vals, 0)) @ vecs.T
        delta = x-r
        y = x.copy()
        np.fill_diagonal(y, 1)
        distance = np.sum((y-a)**2)  # Course wgtNorm is squared Frobenius.
        if (abs(distance-previous_distance) < tol and
                np.linalg.eigvalsh(y)[0] > -tol):
            return y, iteration
        previous_distance = distance
    raise RuntimeError("Higham did not converge")


def fit_t(x):
    """Location-scale Student t MLE; standardize only for optimizer conditioning.

    Three starting df values guard against a poor local fit. The estimates here
    are all above the course's nu > 2 restriction; no restriction binds.
    """
    x = np.asarray(x)
    center, unit = x.mean(), x.std(ddof=0)
    z = (x-center)/unit
    candidates = [stats.t.fit(z, start, loc=0, scale=1) for start in (4, 12, 40)]
    pars = max(candidates, key=lambda p: stats.t.logpdf(z, *p).sum())
    nu, loc, scale = pars[0], center + unit*pars[1], unit*pars[2]
    assert nu > 2 and scale > 0
    return {"nu": nu, "location": loc, "scale": scale,
            "loglik": stats.t.logpdf(x, nu, loc=loc, scale=scale).sum()}


def aicc(ll, k, n):
    return -2*ll + 2*k + 2*k*(k+1)/(n-k-1)


def copula_loglik(u, r, nu=None):
    """Per-observation log copula density, with the core densities explicit."""
    d = u.shape[1]
    logdet = np.linalg.slogdet(r)[1]
    y = stats.norm.ppf(u) if nu is None else stats.t.ppf(u, nu)
    quadratic = np.sum(y * np.linalg.solve(r, y.T).T, axis=1)
    if nu is None:
        return -.5*logdet - .5*(quadratic-np.sum(y*y, axis=1))
    joint = (gammaln((nu+d)/2) - gammaln(nu/2)
             - d/2*np.log(nu*np.pi) - .5*logdet
             - (nu+d)/2*np.log1p(quadratic/nu))
    marginal = (gammaln((nu+1)/2) - gammaln(nu/2)
                - .5*np.log(nu*np.pi) - (nu+1)/2*np.log1p(y*y/nu))
    return joint - marginal.sum(axis=1)


def rank_uniforms(x):
    return stats.rankdata(x, axis=0, method="average")/(len(x)+1)


def joint_tails(u, scale=1):
    rows = []
    for i, j in combinations(range(3), 2):
        rows.append({"pair": f"X{i+1}-X{j+1}",
                     "lower": np.sum((u[:, i] <= .025) & (u[:, j] <= .025))*scale,
                     "upper": np.sum((u[:, i] >= .975) & (u[:, j] >= .975))*scale})
    return rows


# Problem 1: counts precede correlations. Asset order matches the stated weights.
names1 = ["IDX", "A", "B", "C", "D"]
d1 = pd.read_csv(BASE / "problem1.csv")[names1]
observed = d1.notna().astype(int)
counts1 = (observed.T @ observed).to_numpy()
complete1 = d1.dropna()
cc = complete1.corr().to_numpy()
pw = d1.corr().to_numpy()
sd1 = d1.std(ddof=1).to_numpy()
w1 = np.array([1, -.4, -.3, -.2, -.1])
cov1 = pw * np.outer(sd1, sd1)
rj = near_psd(pw)
hm, iterations = higham(pw)
p1 = {"names": names1, "counts": counts1, "complete_n": len(complete1),
      "std_full": sd1, "pairwise_covariance": cov1,
      "tracking_weights": w1, "matrices": {}, "higham_iterations": iterations}
for name, matrix in [("Complete case", cc), ("Pairwise", pw),
                     ("Rebonato-Jackel", rj), ("Higham", hm)]:
    eigen = np.linalg.eigvalsh(matrix)
    entry = {"matrix": matrix, "eigenvalues": eigen, "min_eigenvalue": eigen[0],
             "distance_from_pairwise": np.linalg.norm(matrix-pw, "fro"),
             "tracking_variance_full_sd": w1 @ (matrix*np.outer(sd1, sd1)) @ w1}
    if name in ("Complete case", "Pairwise"):
        try:
            root = np.linalg.cholesky(matrix)
            entry["cholesky"] = "Succeeded"
            entry["cholesky_error"] = np.max(np.abs(root@root.T-matrix))
        except np.linalg.LinAlgError:
            entry["cholesky"] = "Failed: matrix is not positive definite"
    assert np.allclose(matrix, matrix.T, atol=1e-13)
    assert np.allclose(np.diag(matrix), 1, atol=1e-12)
    if name != "Pairwise":
        assert eigen[0] >= -1e-10
    p1["matrices"][name] = entry
p1["higham_changes"] = sorted([
    {"pair": f"{names1[i]}-{names1[j]}", "count": counts1[i,j],
     "before": pw[i,j], "after": hm[i,j], "change": hm[i,j]-pw[i,j]}
    for i,j in combinations(range(5), 2)], key=lambda z: abs(z["change"]), reverse=True)
p1["estimator_gap"] = np.linalg.norm(pw-cc, "fro")
p1["repair_method_gap"] = np.linalg.norm(rj-hm, "fro")
assert np.linalg.norm(hm-pw, "fro") <= np.linalg.norm(rj-pw, "fro") + 1e-10
results["p1"] = p1


# Problem 2: mean removed globally; EW variance also removes its weighted mean.
d2 = pd.read_csv(BASE / "problem2.csv")
raw2 = d2.Price.pct_change(fill_method=None).dropna().to_numpy()
x2 = raw2 - raw2.mean()
n2 = len(x2)
z95 = -stats.norm.ppf(.05)
p2 = {"n": n2, "removed_mean": raw2.mean(), "moments": moments(x2),
      "raw_moments": moments(raw2), "recent_days": RECENT_DAYS, "ew": {}}
for lam in (.94, .97):
    weights = (1-lam)*lam**np.arange(n2-1, -1, -1)
    weights /= weights.sum()
    assert np.isclose(weights.sum(), 1) and np.all(np.diff(weights) > 0)
    ewmean = weights@x2
    sigma = np.sqrt(weights@((x2-ewmean)**2))
    neff = 1/np.sum(weights**2)
    p2["ew"][str(lam)] = {"lambda": lam, "n_eff": neff,
        "n_eff_infinite": (1+lam)/(1-lam), "half_life": np.log(.5)/np.log(lam),
        "recent_weight": weights[-RECENT_DAYS:].sum(), "weighted_mean": ewmean,
        "sigma": sigma, "weight_sum": weights.sum(), "VaR": 1e6*z95*sigma,
        "se_sigma": sigma/np.sqrt(2*neff), "se_VaR": 1e6*z95*sigma/np.sqrt(2*neff)}
t2 = fit_t(x2)
p2["t_fit"] = t2
p2["var"] = {"Normal, equal weight": 1e6*z95*x2.std(ddof=1),
             "Normal, EW 0.97": p2["ew"]["0.97"]["VaR"],
             "Normal, EW 0.94": p2["ew"]["0.94"]["VaR"],
             "Student t MLE": -1e6*stats.t.ppf(.05, t2["nu"], loc=t2["location"], scale=t2["scale"]),
             "Historical": historical_risk(1e6*x2)["VaR"]}
p2["var_order"] = sorted(p2["var"], key=p2["var"].get)
p2["regimes"] = [{"days": f"1-{n2-RECENT_DAYS}", "n": n2-RECENT_DAYS, **moments(x2[:-RECENT_DAYS])},
                 {"days": f"{n2-RECENT_DAYS+1}-{n2}", "n": RECENT_DAYS, **moments(x2[-RECENT_DAYS:])}]
prob = np.array([q["n"]/n2 for q in p2["regimes"]])
regvar = np.array([q["variance"] for q in p2["regimes"]])
p2["normal_variance_mixture_excess"] = 3*(prob@(regvar**2))/(prob@regvar)**2-3
p2["ew_var_gap"] = p2["ew"]["0.94"]["VaR"]-p2["ew"]["0.97"]["VaR"]
p2["se_gap_independence_benchmark"] = np.hypot(p2["ew"]["0.94"]["se_VaR"], p2["ew"]["0.97"]["se_VaR"])
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(d2.Day.iloc[1:], 100*x2, lw=.7, color="#245c7d")
ax.axhline(0, color="gray", lw=.6)
ax.axvspan(n2-RECENT_DAYS+.5, n2+.5, color="#e5b268", alpha=.25, label="Visual split: final 40 days")
ax.set(title="Demeaned arithmetic returns", xlabel="Return day", ylabel="Daily return (%)")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(FIG / "problem2_returns.png")
plt.close(fig)
results["p2"] = p2


# Problem 3: retain one-year expected gains; do not demean these scenarios.
d3 = pd.read_csv(BASE / "problem3.csv")
bad3 = d3[["A", "B"]] < -.20
p3 = {"n": len(d3), "moments": {c: moments(d3[c]) for c in ["A", "B"]},
      "loss_counts": {"A": bad3.A.sum(), "B": bad3.B.sum(),
                      "either": bad3.any(axis=1).sum(), "both": bad3.all(axis=1).sum()}, "risk": {}}
positions3 = {"1M A": 1e6*d3.A, "1M B": 1e6*d3.B,
              "2M A": 2e6*d3.A, "1M A + 1M B": 1e6*(d3.A+d3.B)}
for name, pnl in positions3.items():
    p3["risk"][name] = {"5%": historical_risk(pnl), "1%": historical_risk(pnl, .01),
                       "normal_VaR_5": -pnl.mean()+z95*pnl.std(ddof=1),
                       "mean_pnl": pnl.mean(), "sd_pnl": pnl.std(ddof=1)}
p3["subadditivity"] = {}
for measure in ["VaR", "ES"]:
    left = p3["risk"]["1M A + 1M B"]["5%"][measure]
    right = p3["risk"]["1M A"]["5%"][measure] + p3["risk"]["1M B"]["5%"][measure]
    p3["subadditivity"][measure] = {"combined": left, "sum_standalone": right, "holds": bool(left <= right)}
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharex=True, sharey=True)
for ax, key, title in zip(axes, ["2M A", "1M A + 1M B"], [r"\$2M in A", r"\$1M in A + \$1M in B"]):
    ax.hist(positions3[key]/1e6, bins=np.linspace(-1.8,.35,95), color="#245c7d",
            weights=np.full(len(d3), 100/len(d3)))
    ax.set(title=title, xlabel="One-year P&L ($ millions)", yscale="log")
    ax.grid(axis="y", alpha=.15)
axes[0].set_ylabel("Scenarios per bin (%) - log scale")
fig.suptitle("Bond portfolio P&L distributions", y=1.01)
fig.tight_layout()
fig.savefig(FIG / "problem3_pnl.png", bbox_inches="tight")
plt.close(fig)
results["p3"] = p3


# Problem 4: descriptive ranks are separate from uniforms through fitted margins.
d4 = pd.read_csv(BASE / "problem4.csv")
x4 = d4.to_numpy()
n4 = len(x4)
urank = rank_uniforms(x4)
p4 = {"n": n4, "moments": {c: moments(d4[c]) for c in d4}, "outlier_check": {},
      "rank_tail_counts": joint_tails(urank), "independent_expected_count": n4*.025**2,
      "rank_uniform_range": [urank.min(), urank.max()], "margins": {}}
for c in d4:
    i = np.argmax(np.abs(d4[c]-d4[c].mean()))
    p4["outlier_check"][c] = {"row": int(i+1), "value": d4[c].iloc[i],
                              "without": moments(d4[c].drop(i))}
fig, axes = plt.subplots(1,3,figsize=(9.3,3.15))
for ax, (i,j) in zip(axes, combinations(range(3), 2)):
    lo = (urank[:,i] <= .025) & (urank[:,j] <= .025)
    hi = (urank[:,i] >= .975) & (urank[:,j] >= .975)
    ax.scatter(urank[:,i],urank[:,j],s=6,alpha=.5,color="#245c7d",edgecolors="none")
    ax.scatter(urank[lo|hi,i],urank[lo|hi,j],s=14,color="#b4483c",edgecolors="none")
    for value in [.025,.975]:
        ax.axvline(value,color="gray",lw=.5,ls=":")
        ax.axhline(value,color="gray",lw=.5,ls=":")
    ax.set(title=f"X{i+1} / X{j+1}", xlabel=f"X{i+1} rank / (n+1)",
           ylabel=f"X{j+1} rank / (n+1)", xlim=(0,1), ylim=(0,1))
fig.suptitle("Rank uniforms: all three asset pairs",y=1.01)
fig.tight_layout()
fig.savefig(FIG / "problem4_ranks.png",bbox_inches="tight")
plt.close(fig)

u4 = np.empty_like(x4)
fitted_margins = []
for j, c in enumerate(d4):
    mu, sigma = stats.norm.fit(x4[:,j])  # Normal MLE uses divisor n.
    ll = stats.norm.logpdf(x4[:,j], mu, sigma).sum()
    normal = {"location": mu, "scale": sigma, "loglik": ll, "k": 2, "AICc": aicc(ll,2,n4)}
    student = fit_t(x4[:,j])
    student.update({"k": 3, "AICc": aicc(student["loglik"],3,n4)})
    choose_t = student["AICc"] < normal["AICc"]
    selected = "Student t" if choose_t else "Normal"
    p4["margins"][c] = {"normal": normal, "t": student, "selected": selected}
    dist = (stats.t(student["nu"], loc=student["location"], scale=student["scale"])
            if choose_t else stats.norm(mu, sigma))
    fitted_margins.append(dist)
    u4[:,j] = dist.cdf(x4[:,j])
clip_lo, clip_hi = np.finfo(float).eps, 1-np.finfo(float).eps
p4["cdf_clip_count"] = np.sum((u4 <= clip_lo) | (u4 >= clip_hi))
u4 = np.clip(u4, clip_lo, clip_hi)
assert np.all((u4 > 0) & (u4 < 1))
tau = np.eye(3)
for i,j in combinations(range(3), 2):
    tau[i,j] = tau[j,i] = stats.kendalltau(u4[:,i],u4[:,j]).statistic
r4 = np.sin(np.pi*tau/2)
np.fill_diagonal(r4, 1)
p4["R_min_eigenvalue_before"] = np.linalg.eigvalsh(r4)[0]
# Course fix_correlation: repair only if necessary, then ensure positive definite.
if np.linalg.eigvalsh(r4)[0] < -1e-8:
    r4, _ = higham(r4)
r4 = (r4+r4.T)/2
if np.linalg.eigvalsh(r4)[0] < 1e-8:
    r4 = (r4 + 1e-8*np.eye(3))/(1+1e-8)
p4.update({"tau": tau, "R": r4, "R_eigenvalues": np.linalg.eigvalsh(r4),
           "fitted_uniform_range": [u4.min(), u4.max()]})

# Exact course profile_nu convention: 200 theta points, then 200-point refinement.
theta = np.linspace(.01,.49,200)
ll_grid = np.array([copula_loglik(u4,r4,1/t).sum() for t in theta])
imax = int(np.argmax(ll_grid))
fine = np.linspace(theta[max(imax-1,0)],theta[min(imax+1,199)],200)
ll_fine = np.array([copula_loglik(u4,r4,1/t).sum() for t in fine])
nu4 = 1/fine[np.argmax(ll_fine)]
g_day = copula_loglik(u4,r4)
t_day = copula_loglik(u4,r4,nu4)
p4["copulas"] = {}
for name, daily, k, nu in [("Gaussian",g_day,0,None), ("Student t",t_day,1,nu4)]:
    ll = daily.sum()
    p4["copulas"][name] = {"loglik": ll, "nu": nu, "k": k,
                            "AICc": aicc(ll,k,n4), "BIC": -2*ll+k*np.log(n4)}

# Shared correlated normals reduce simulation noise in the comparison; the t
# vector gets one shared chi-square shock per day, not separate shocks per asset.
rng = np.random.default_rng(SEED)
z4 = rng.standard_normal((NSIM,3)) @ np.linalg.cholesky(r4).T
chi = np.random.default_rng(SEED+1).chisquare(nu4,NSIM)
ug = np.clip(stats.norm.cdf(z4),clip_lo,clip_hi)
ut = np.clip(stats.t.cdf(z4/np.sqrt(chi[:,None]/nu4),nu4),clip_lo,clip_hi)
p4["risk"] = {}
p4["simulated_tail_counts"] = {}
p4["simulated_uniform_ranges"] = {}
for name, usim in [("Gaussian",ug), ("Student t",ut)]:
    assert usim.shape == (NSIM,3) and np.all((usim>0)&(usim<1))
    simulated = np.column_stack([dist.ppf(usim[:,j]) for j,dist in enumerate(fitted_margins)])
    assert np.isfinite(simulated).all()
    pnl = simulated.sum(axis=1)*1e6
    p4["risk"][name] = {"5%": historical_risk(pnl), "1%": historical_risk(pnl,.01)}
    # Re-rank each simulated margin: exactly the same own-tail definition as (b).
    p4["simulated_tail_counts"][name] = joint_tails(rank_uniforms(simulated),n4/NSIM)
    p4["simulated_uniform_ranges"][name] = [usim.min(),usim.max()]
pnl_history = 1e6*x4.sum(axis=1)
p4["risk"]["Historical"] = {"5%": historical_risk(pnl_history), "1%": historical_risk(pnl_history,.01)}
p4["risk_change"] = [{"level": a, "measure": m,
    "difference": p4["risk"]["Student t"][a][m]-p4["risk"]["Gaussian"][a][m],
    "percent": 100*(p4["risk"]["Student t"][a][m]/p4["risk"]["Gaussian"][a][m]-1)}
    for a in ["5%","1%"] for m in ["VaR","ES"]]
diff = t_day-g_day
# "Own outer 5%" means empirical ranks; also report the fitted-CDF sensitivity.
nonextreme = np.all((urank>.05)&(urank<.95),axis=1)
nonextreme_cdf = np.all((u4>.05)&(u4<.95),axis=1)
p4["likelihood_contributions"] = {
    "total": diff.sum(), "nonextreme_count": nonextreme.sum(),
    "nonextreme": diff[nonextreme].sum(), "extreme": diff[~nonextreme].sum(),
    "nonextreme_percent": 100*diff[nonextreme].sum()/diff.sum(),
    "fitted_cdf_nonextreme_count": nonextreme_cdf.sum(),
    "fitted_cdf_nonextreme": diff[nonextreme_cdf].sum(),
    "fitted_cdf_nonextreme_percent": 100*diff[nonextreme_cdf].sum()/diff.sum()}
daily_table = pd.DataFrame({"observation": np.arange(1,n4+1), "loglik_gaussian": g_day,
                           "loglik_t": t_day, "difference_t_minus_gaussian": diff,
                           "non_extreme_rank": nonextreme, "non_extreme_fitted_cdf": nonextreme_cdf})
daily_table.to_csv(BASE / "problem4_daily_loglik.csv", index=False)
i,j = max(combinations(range(3),2),key=lambda pair:r4[pair])
rho = r4[i,j]
p4["tail_dependence"] = {"pair": f"X{i+1}-X{j+1}", "rho": rho, "nu": nu4,
    "t": 2*stats.t.cdf(-np.sqrt((nu4+1)*(1-rho)/(1+rho)),nu4+1), "Gaussian": 0}
results["p4"] = p4


# Problem 5: OLS intercepts are reported, but risk simulation has zero means as
# instructed. Equivalently, simulate alpha + beta*M + e and subtract fitted mean.
d5 = pd.read_csv(BASE / "problem5.csv")
returns5 = d5[["MKT","A","B"]].pct_change(fill_method=None).dropna().to_numpy()
market = returns5[:,0]
stocks = returns5[:,1:]
design = np.column_stack([np.ones(len(market)),market])
coef = np.linalg.lstsq(design,stocks,rcond=None)[0]
residual = stocks-design@coef
# Use n-1 for residual covariance, consistent with the sample covariance of
# returns. This gives the exact in-sample OLS covariance decomposition. n-2
# regression error estimates are also reported, but are not used in simulation.
res_cov = np.cov(residual,rowvar=False,ddof=1)
market_var = market.var(ddof=1)
sample_cov = np.cov(stocks,rowvar=False,ddof=1)
model_cov = np.outer(coef[1],coef[1])*market_var+res_cov
assert np.allclose(model_cov,sample_cov,rtol=1e-12,atol=1e-15)
p5 = {"n": len(market), "alpha": coef[0], "beta": coef[1],
      "residual_sd": np.sqrt(np.diag(res_cov)),
      "residual_sd_n_minus_2": np.sqrt(np.sum(residual**2,axis=0)/(len(market)-2)),
      "residual_correlation": np.corrcoef(residual,rowvar=False)[0,1],
      "market_sd": np.sqrt(market_var), "residual_covariance": res_cov,
      "sample_covariance": sample_cov, "model_covariance": model_cov,
      "covariance_max_error": np.max(np.abs(model_cov-sample_cov)),
      "market_residual_covariance": np.cov(np.column_stack([market,residual]),rowvar=False)[0,1:],
      "risk": {}}
rng = np.random.default_rng(SEED)
shocks = rng.standard_normal((NSIM,3))
market_sim = shocks[:,0]*np.sqrt(market_var)
exposures = {"P1": np.array([1e6,1e6]), "P2": np.array([1e6,-1e6])}
for scenario, cov in [("Full residual covariance",res_cov),
                      ("Independent residuals",np.diag(np.diag(res_cov)))]:
    eps_sim = shocks[:,1:] @ np.linalg.cholesky(cov).T
    sim_returns = market_sim[:,None]*coef[1]+eps_sim
    assert sim_returns.shape == (NSIM,2) and np.isfinite(sim_returns).all()
    p5["risk"][scenario] = {name: historical_risk(sim_returns@w)["VaR"] for name,w in exposures.items()}
p5["risk"]["Delta normal"] = {name:z95*np.sqrt(w@sample_cov@w) for name,w in exposures.items()}
p5["risk"]["Independent analytic"] = {
    name:z95*np.sqrt(w@(np.outer(coef[1],coef[1])*market_var+np.diag(np.diag(res_cov)))@w)
    for name,w in exposures.items()}
p5["effects"] = {}
for name,w in exposures.items():
    full = p5["risk"]["Full residual covariance"][name]
    independent = p5["risk"]["Independent residuals"][name]
    p5["effects"][name] = {"shortcut_change": independent-full,
        "shortcut_percent": 100*(independent/full-1),
        "simulation_minus_delta": full-p5["risk"]["Delta normal"][name],
        "simulation_delta_percent": 100*(full/p5["risk"]["Delta normal"][name]-1),
        "market_variance": market_var*(w@coef[1])**2,
        "residual_diagonal_variance": w@np.diag(np.diag(res_cov))@w,
        "residual_cross_variance": 2*w[0]*w[1]*res_cov[0,1]}
results["p5"] = p5


# Small mathematical checks, not a test framework.
for matrix in [cov1,res_cov,sample_cov,model_cov,r4]:
    assert np.allclose(matrix,matrix.T,atol=1e-13)
for matrix in [res_cov,sample_cov,model_cov,r4]:
    assert np.linalg.eigvalsh(matrix)[0] >= -1e-10
assert p3["subadditivity"]["ES"]["holds"]
assert np.allclose(2*np.array(list(p3["risk"]["1M A"]["5%"].values())),
                   list(p3["risk"]["2M A"]["5%"].values()))
assert np.isfinite(g_day).all() and np.isfinite(t_day).all()
assert len(daily_table) == n4 and np.isclose(daily_table.difference_t_minus_gaussian.sum(),diff.sum())
for name in ["problem2_returns.png","problem3_pnl.png","problem4_ranks.png"]:
    assert (FIG / name).is_file()
results["checks"] = {"status": "All checks passed", "simulation_rows_per_model": NSIM,
                     "figures": 3, "daily_likelihood_rows": n4}


def json_number(obj):
    if isinstance(obj,np.ndarray):
        return obj.tolist()
    if isinstance(obj,np.generic):
        return obj.item()
    raise TypeError(type(obj).__name__)


(BASE / "results.json").write_text(json.dumps(results,indent=2,default=json_number,allow_nan=False),encoding="utf-8")
print("P1 tracking variance:",p1["matrices"]["Pairwise"]["tracking_variance_full_sd"])
print("P2 VaR:",p2["var"])
print("P3 historical 5%:",{key:val["5%"] for key,val in p3["risk"].items()})
print("P4 copulas:",p4["copulas"])
print("P4 portfolio risk:",p4["risk"])
print("P5 VaR:",p5["risk"])
print("All checks passed. Wrote results.json, problem4_daily_loglik.csv, and 3 figures.")
