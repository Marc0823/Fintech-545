"""FINTECH 545 Assignment 3. Run with python assignment3.py."""
from datetime import date
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import optimize, stats


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
FIGURES = BASE / "figures"
TAILS = (0.05, 0.01)
SEED = 545
NSIM = 1_000_000
DAYS = 255
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})


def gbsm(sign, stock, strike, time, rate, carry, vol, greeks=False):
    """Call sign +1, put sign -1. Theta is per year of time elapsed."""
    width = vol * np.sqrt(time)
    d1 = (np.log(stock / strike) + (carry + vol**2 / 2) * time) / width
    d2 = d1 - width
    a, b = np.exp((carry - rate) * time), np.exp(-rate * time)
    n1, n2 = stats.norm.cdf(sign * d1), stats.norm.cdf(sign * d2)
    value = sign * (stock * a * n1 - strike * b * n2)
    if not greeks:
        return value
    density = stats.norm.pdf(d1)
    delta = sign * a * n1
    gamma = a * density / (stock * width)
    theta = (-stock * a * density * vol / (2 * np.sqrt(time))
             - sign * (carry - rate) * stock * a * n1
             - sign * rate * strike * b * n2)
    vega = stock * a * density * np.sqrt(time)
    return value, delta, gamma, theta, vega


def moments(values):
    x = np.asarray(values)
    sd = x.std(ddof=1)
    return [x.mean(), sd**2, stats.skew(x), stats.kurtosis(x)]


def aicc(log_likelihood, k, n):
    return -2 * log_likelihood + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def empirical_risk(pnl):
    """Course RiskStats.jl convention, including the floor/ceil midpoint."""
    x = np.sort(pnl)
    risk = []
    for alpha in TAILS:
        count = alpha * len(x)
        q = (x[int(np.floor(count)) - 1] + x[int(np.ceil(count)) - 1]) / 2
        risk.extend([-q, -x[x <= q].mean()])
    return risk


def normal_risk(mean, sd):
    risk = []
    for alpha in TAILS:
        z = stats.norm.ppf(alpha)
        risk.extend([-mean - sd * z, -mean + sd * stats.norm.pdf(z) / alpha])
    return risk


def ols(y, factors):
    """OLS with an intercept and conventional homoskedastic standard errors."""
    n = len(y)
    x = np.column_stack([np.ones(n), factors])
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ coef
    sse = resid @ resid
    p = len(coef)
    se = np.sqrt(sse / (n - p) * np.diag(np.linalg.inv(x.T @ x)))
    ll = -n / 2 * (1 + np.log(2 * np.pi * sse / n))
    r2 = 1 - sse / ((y - y.mean()) @ (y - y.mean()))
    return dict(coef=coef, se=se, alpha_t=coef[0] / se[0], r2=r2,
                aicc=aicc(ll, p + 1, n)), resid


def chain_inputs():
    stock, rate, q = 182.40, 0.0425, 0.006
    time = (date(2026, 12, 18) - date(2026, 11, 13)).days / 365
    chain = pd.read_csv(DATA / "problem1.csv").set_index(["Strike", "Type"]).sort_index()
    assert len(chain) == 36 and chain.index.is_unique and (chain.Ask >= chain.Bid).all()
    chain["Mid"] = (chain.Bid + chain.Ask) / 2
    call, put = (chain.xs(kind, level="Type") for kind in ("Call", "Put"))
    strike = call.index.to_numpy()
    fair = stock * np.exp(-q * time) - strike * np.exp(-rate * time)
    gap = call.Mid - put.Mid - fair
    band = call.Ask - call.Bid + put.Ask - put.Bid
    parity = pd.DataFrame(dict(gap=gap, band=band, flag=gap.abs() > band))
    return stock, rate, q, time, call, put, parity


def book_inputs():
    prices = pd.read_csv(DATA / "problem3.csv", parse_dates=["Date"])
    assert len(prices) == 501 and prices.Date.is_monotonic_increasing
    assert prices.Date.is_unique and (prices.ABC > 0).all()
    books = pd.read_csv(DATA / "problem3_portfolios.csv")
    stock = prices.ABC.iloc[-1]
    returns = prices.ABC.pct_change().dropna().to_numpy()
    greeks = {}
    for name, positions in books.groupby("Portfolio", sort=False):
        value = np.zeros(5)
        for position in positions.itertuples(index=False):
            if position.Type == "Stock":
                unit = [stock, 1., 0., 0., 0.]
            else:
                sign = 1 if position.Type == "Call" else -1
                unit = gbsm(sign, stock, position.Strike, position.DaysToExpiry / DAYS,
                            0.0425, 0.0425, position.ImpliedVol, greeks=True)
            value += position.Holding * np.asarray(unit)
        greeks[name] = value[:4]
    return stock, returns, books, greeks


def monthly_inputs():
    monthly = pd.read_csv(DATA / "problem4.csv", parse_dates=["Month"]).set_index("Month")
    assert len(monthly) == 240 and monthly.index.is_unique and np.isfinite(monthly).all().all()
    rf = 1.03**(1 / 12) - 1
    summary = {}
    for name in monthly:
        x = monthly[name]
        mean, var, skew, kurtosis = moments(x)
        summary[name] = [mean, var, skew, kurtosis,
                         np.sqrt(12) * (mean - rf) / np.sqrt(var),
                         np.sqrt(var / len(x))]
    putw = monthly.PUTW
    fourth = (putw - putw.mean())**4
    worst = putw.idxmin()
    remaining = putw.drop(worst)
    impact = dict(date=worst.date().isoformat(), value=putw.loc[worst],
                  fourth_share=fourth.loc[worst] / fourth.sum(),
                  remaining_moments=moments(remaining),
                  remaining_sharpe=np.sqrt(12) * (remaining.mean() - rf) / remaining.std())
    return monthly, rf, summary, impact


def factor_inputs():
    def read_factors(filename):
        frame = pd.read_csv(DATA / filename, dtype={"Date": "string"})
        frame.columns = frame.columns.str.strip()
        assert frame.Date.str.fullmatch(r"\d{8}").all()
        frame.Date = pd.to_datetime(frame.Date, format="%Y%m%d")
        frame = frame.set_index("Date").sort_index()
        assert frame.index.is_unique and np.isfinite(frame).all().all()
        assert not frame.isin([-99.99, -999]).any().any()
        return frame / 100

    ff = read_factors("F-F_Research_Data_Factors_daily.CSV")
    mom = read_factors("F-F_Momentum_Factor_daily.CSV")
    factors = ff.join(mom, how="inner")
    funds = pd.read_csv(DATA / "problem5.csv", parse_dates=["Date"]).set_index("Date")
    aligned = funds.join(factors, how="left", validate="one_to_one")
    assert aligned.index.is_monotonic_increasing and np.isfinite(aligned).all().all()
    diagnostics = {}
    for name in funds:
        y = aligned[name] - aligned.RF
        capm, resid = ols(y, aligned[["Mkt-RF"]])
        corr = [np.corrcoef(resid, aligned[factor])[0, 1] for factor in ("SMB", "HML", "Mom")]
        diagnostics[name] = dict(mean=y.mean(), sd=y.std(),
                                 sharpe=np.sqrt(DAYS) * y.mean() / y.std(),
                                 capm=capm, corr=corr)
    return aligned, factors, mom, diagnostics


def problem1():
    stock, rate, q, time, call, put, parity = chain_inputs()
    rows = []
    for strike in parity.index:
        row = dict(strike=strike, **parity.loc[strike].to_dict())
        for kind, sign, quote in [("Call", 1, call), ("Put", -1, put)]:
            fair = stock * np.exp(-q * time) - strike * np.exp(-rate * time)
            lower = max(sign * fair, 0.)
            upper = stock * np.exp(-q * time) if sign == 1 else strike * np.exp(-rate * time)
            fitted = {}
            for price_type in ("Bid", "Mid", "Ask"):
                price = quote.at[strike, price_type]
                if not lower <= price < upper:
                    fitted[price_type] = None
                    continue
                if price == lower:
                    fitted[price_type] = 0.
                    continue

                def residual(vol):
                    return gbsm(sign, stock, strike, time, rate, rate - q, vol) - price

                iv = optimize.brentq(residual, 1e-8, 8, xtol=1e-13)
                assert abs(residual(iv)) < 1e-8
                fitted[price_type] = iv
            mid_vol = fitted["Mid"]
            vega = gbsm(sign, stock, strike, time, rate, rate - q, mid_vol, greeks=True)[4]
            row[kind] = dict(**fitted, vega=vega, cent=0.01 / vega,
                             lower=lower, bid_price=quote.at[strike, "Bid"])
        rows.append(row)

    fig, ax = plt.subplots(figsize=(7.1, 3.2), layout="constrained")
    strikes = [row["strike"] for row in rows]
    for kind, color, marker in [("Call", "#174571", "o"), ("Put", "#9b4b2b", "s")]:
        iv = [row[kind]["Mid"] for row in rows]
        ax.plot(strikes, iv, color=color, marker=marker, markersize=4, label=f"{kind} mid IV")
    flagged = [row for row in rows if row["flag"]]
    for row in flagged:
        ax.axvline(row["strike"], color="0.45", linewidth=0.9, linestyle="--")
    ax.axvspan(212.5, 227, color="0.93", zorder=0)
    ax.set(xlabel="Strike ($)", ylabel="Annualized implied volatility", xlim=(138, 227))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "problem1_iv.pdf")
    plt.close(fig)
    return dict(time=time, rows=rows)


def option_tree(stock, strike, time, rate, vol, n, dividends=()):
    """Return European call/put and American call/put using the same CRR grid.

    A cash dividend splits the lattice into independent subtrees. Each row in
    a segment is a starting stock price from a distinct dividend history.
    """
    dt = time / n
    u = np.exp(vol * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(rate * dt) - d) / (u - d)
    discount = np.exp(-rate * dt)
    assert 0 < p < 1
    signs = np.array([1, -1])

    def rollback(spots, remaining, events):
        steps = events[0][0] if events else remaining
        # At the event node, continuation uses the ex-dividend stock. Exercise
        # at the event uses the stock immediately before the cash payment.
        terminal = spots[:, None] * u**(2 * np.arange(steps + 1) - steps)
        exercise = np.maximum((terminal[..., None] - strike) * signs, 0.)
        if events:
            _, dividend = events[0]
            next_spots = terminal - dividend
            assert next_spots.min() > 0
            tail_events = [(step - steps, amount) for step, amount in events[1:]]
            values = rollback(next_spots.ravel(), remaining - steps, tail_events)
            values = values.reshape(*terminal.shape, 4)
            values[..., 2:] = np.maximum(values[..., 2:], exercise)
        else:
            values = np.concatenate([exercise, exercise], axis=-1)
        for level in range(steps - 1, -1, -1):
            values = discount * ((1 - p) * values[:, :-1] + p * values[:, 1:])
            nodes = spots[:, None] * u**(2 * np.arange(level + 1) - level)
            exercise = np.maximum((nodes[..., None] - strike) * signs, 0.)
            values[..., 2:] = np.maximum(values[..., 2:], exercise)
        return values[:, 0]

    return rollback(np.atleast_1d(stock), n, dividends)[0]


def problem2():
    stock, strike, time, rate, vol = 48.50, 50., 0.5, 0.045, 0.30
    div_times = np.array([0.10, 0.40])
    amounts = np.array([0.25, 1.50])
    thresholds = strike * (1 - np.exp(-rate * (time - div_times)))
    pv = amounts @ np.exp(-rate * div_times)
    n = 400

    def with_div(steps, spot=stock, dividends=amounts):
        assert steps % 5 == 0
        events = list(zip([steps // 5, 4 * steps // 5], dividends))
        return option_tree(spot, strike, time, rate, vol, steps, events)

    base = option_tree(stock, strike, time, rate, vol, n)
    analytic = [gbsm(sign, stock, strike, time, rate, rate, vol) for sign in (1, -1)]
    adjusted = [gbsm(sign, stock - pv, strike, time, rate, rate, vol) for sign in (1, -1)]
    convergence = {steps: with_div(steps) for steps in [25, 50, 100, 105, 200, 205, 400, 405]}
    central = convergence[n]
    assert np.allclose(base[0], base[2], atol=1e-10)
    assert np.all(central[2:] >= central[:2])
    assert abs(central[0] - central[1] - (stock - pv - strike * np.exp(-rate*time))) < 1e-9
    differences = []
    for h in (1, 0.1, 0.01, 0.001):
        up, down = with_div(n, stock + h)[2:], with_div(n, stock - h)[2:]
        delta = (up - down) / (2 * h)
        gamma = (up - 2 * central[2:] + down) / h**2
        differences.append([h, delta[0], gamma[0], delta[1], gamma[1]])

    dividend_sens = []
    h = 0.01
    for bump in np.eye(2) * h:
        dividend_sens.append((with_div(n, dividends=amounts + bump)
                              - with_div(n, dividends=amounts - bump)) / (2 * h))

    def exercise_advantage(spot):
        continuation = option_tree(spot - amounts[1], strike, time - div_times[1],
                                   rate, vol, n // 5)[0]
        return spot - strike - continuation

    boundary = optimize.brentq(exercise_advantage, strike, 100)
    exact_boundary = optimize.brentq(
        lambda s: s - strike - gbsm(1, s - amounts[1], strike, 0.1, rate, rate, vol),
        strike, 100)
    return dict(n=n, thresholds=thresholds, pv=pv, no_div=[*analytic, *base[2:]],
                no_div_tree=base, adjusted=adjusted, with_div=central,
                convergence=convergence, differences=differences, dividend_sens=dividend_sens,
                boundary=boundary, exact_boundary=exact_boundary)


def problem3():
    stock, returns, books, greeks = book_inputs()
    horizon, rate = 10, 0.0425
    mean, sd = stats.norm.fit(returns)
    standardized = (returns - mean) / sd
    df, loc, scale = stats.t.fit(standardized)
    t_mean, t_scale = mean + loc * sd, scale * sd
    assert df > 2
    normal_ll = stats.norm.logpdf(returns, mean, sd).sum()
    t_ll = stats.t.logpdf(returns, df, t_mean, t_scale).sum()
    fit = dict(Normal=[mean, sd, aicc(normal_ll, 2, len(returns))],
               Student=[t_mean, t_scale, df, aicc(t_ll, 3, len(returns))])

    def revalue(positions, terminal):
        total = np.zeros_like(terminal, dtype=float)
        for position in positions.itertuples(index=False):
            if position.Type == "Stock":
                value = terminal
            else:
                sign = 1 if position.Type == "Call" else -1
                time = (position.DaysToExpiry - horizon) / DAYS
                value = gbsm(sign, terminal, position.Strike, time, rate, rate, position.ImpliedVol)
            total += position.Holding * value
        return total

    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((NSIM, horizon))
    # Independent daily chi-square scales produce independent Student-t draws.
    # Sharing z across distributions reduces noise in the comparison.
    normal_daily = z * sd
    t_daily = z * (t_scale / np.sqrt(rng.chisquare(df, z.shape) / df))
    assert min(normal_daily.min(), t_daily.min()) > -1
    terminal = {"Normal": stock * np.prod(1 + normal_daily, axis=1),
                "Student": stock * np.prod(1 + t_daily, axis=1)}
    scenarios = np.linspace(min(np.quantile(x, .0005) for x in terminal.values()),
                            max(np.quantile(x, .9995) for x in terminal.values()), 401)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), layout="constrained", sharex=True)
    result = {}
    for (name, positions), ax in zip(books.groupby("Portfolio", sort=False), axes.flat):
        value, delta, gamma, theta = greeks[name]
        elapsed = horizon / DAYS
        flat = revalue(positions, np.array(stock)) - value
        delta_risk = normal_risk(0, abs(delta) * stock * sd * np.sqrt(horizon))
        full_risk, quadratic_risk, mean_se = {}, {}, {}
        for model, final_stock in terminal.items():
            pnl = revalue(positions, final_stock) - value
            move = final_stock - stock
            quadratic = delta * move + 0.5 * gamma * move**2 + theta * elapsed
            full_risk[model] = [pnl.mean(), *empirical_risk(pnl)]
            quadratic_risk[model] = [quadratic.mean(), *empirical_risk(quadratic)]
            mean_se[model] = pnl.std(ddof=1) / np.sqrt(NSIM)
        result[name] = dict(value=value, delta=delta, gamma=gamma, theta=theta,
                            flat=flat, delta_risk=delta_risk, full=full_risk,
                            quadratic=quadratic_risk, mean_se=mean_se)
        ax.plot(scenarios, revalue(positions, scenarios) - value, color="#174571", label="Full revaluation")
        ax.plot(scenarios, delta * (scenarios - stock), color="#9b4b2b", linestyle="--", label="Delta only")
        ax.axhline(0, color="0.65", linewidth=0.6)
        ax.set_title(name)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.supxlabel("ABC price after 10 trading days ($)")
    fig.supylabel("10-day P&L ($)")
    fig.savefig(FIGURES / "problem3_pnl.pdf")
    plt.close(fig)
    return dict(stock=stock, moments=moments(returns), fit=fit, books=result)


def problem4():
    monthly, rf, summary, impact = monthly_inputs()
    returns = monthly.to_numpy()
    count = returns.shape[1]
    ones = np.ones(count)
    putw = np.eye(count)[-1]

    def tangencies(x):
        mean = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        excess = mean - rf
        free = np.linalg.solve(cov, excess)
        # Some resamples have no finite positive-Sharpe unconstrained
        # tangency with sum(w) = 1. Keep their constrained solutions.
        weights = [free / free.sum() if free.sum() > 0 else np.full(count, np.nan)]

        # Scale the risky weights so excess @ z = 1. Minimizing variance
        # on this slice is equivalent to maximizing positive Sharpe.
        for capped in (False, True):
            vertices = np.eye(count)
            if capped:
                vertices = np.vstack([vertices[:-1], 0.8 * vertices[:-1] + 0.2 * putw])
            start = vertices[np.argmax(vertices @ excess)]
            best_excess = start @ excess
            if best_excess <= 0:
                weights.append(np.full(count, np.nan))
                continue
            scaled_excess = excess / best_excess
            scaled_cov = cov / np.diag(cov).max()
            constraints = [optimize.LinearConstraint(scaled_excess, 1, 1)]
            if capped:
                constraints.append(optimize.LinearConstraint(putw - 0.2 * ones, -np.inf, 0))
            solution = optimize.minimize(lambda z: z @ scaled_cov @ z,
                                         start,
                                         jac=lambda z: 2 * scaled_cov @ z,
                                         bounds=[(0, None)] * count, constraints=constraints,
                                         method="SLSQP", options={"ftol": 1e-11, "maxiter": 250})
            if not solution.success:
                raise RuntimeError(solution.message)
            w = solution.x / solution.x.sum()
            assert w.min() >= -1e-10 and (not capped or w[-1] <= 0.2 + 1e-9)
            weights.append(w)
        return mean, cov, np.array(weights)

    mean, cov, weights = tangencies(returns)
    assert np.isfinite(weights).all()
    portfolios = {}
    for name, w in zip(("Unconstrained", "Long only", "PUTW capped"), weights):
        pnl = returns @ w
        mean_p, sd_p = pnl.mean(), pnl.std(ddof=1)
        portfolios[name] = dict(w=w, mean=mean_p, sd=sd_p,
                                sharpe=np.sqrt(12) * (mean_p - rf) / sd_p,
                                historical=empirical_risk(pnl), normal=normal_risk(mean_p, sd_p))

    def min_variance(target=None):
        constraints = [optimize.LinearConstraint(ones, 1, 1)]
        if target is not None:
            constraints.append(optimize.LinearConstraint(100 * mean, 100 * target, 100 * target))
        scaled_cov = 10_000 * cov
        result = optimize.minimize(lambda w: w @ scaled_cov @ w, ones / count,
                                   jac=lambda w: 2 * scaled_cov @ w, bounds=[(0, 1)] * count,
                                   constraints=constraints, method="SLSQP", options={"ftol": 1e-12})
        if not result.success:
            raise RuntimeError(result.message)
        assert abs(result.x.sum() - 1) < 1e-9
        return result.x

    gmv = min_variance()
    targets = np.linspace(gmv @ mean, mean.max(), 120)
    frontier = np.array([min_variance(target) for target in targets])
    assert np.allclose(frontier @ mean, targets, atol=1e-9)
    sigma = np.sqrt(12 * np.einsum("ij,jk,ik->i", frontier, cov, frontier))
    fig, ax = plt.subplots(figsize=(7.1, 3.55), layout="constrained")
    ax.plot(sigma, 12 * targets, color="#174571", label="Long-only efficient frontier")
    tangent = portfolios["Long only"]
    cml_x = np.array([0., max(sigma)])
    ax.plot(cml_x, 12 * rf + tangent["sharpe"] * cml_x, color="#9b4b2b",
            linestyle="--", label="Capital market line")
    asset_sigma = np.sqrt(12 * np.diag(cov))
    ax.scatter(asset_sigma, 12 * mean, color="0.3", s=19, zorder=3)
    for name, x, y in zip(monthly.columns, asset_sigma, 12 * mean):
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.scatter(np.sqrt(12) * tangent["sd"], 12 * tangent["mean"], marker="D", s=27,
               color="#174571", zorder=4, label="Long-only tangency")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Annualized volatility", ylabel="Annualized arithmetic mean return",
           xlim=(0, asset_sigma.max() * 1.13))
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "problem4_frontier.pdf")
    plt.close(fig)

    excess = monthly - rf
    capm, _ = ols(excess.PUTW, excess[["EQ"]])
    down = excess.loc[excess.EQ < -0.05]
    down_capm, _ = ols(down.PUTW, down[["EQ"]])
    rng = np.random.default_rng(SEED + 1)
    boot = []
    for indices in rng.integers(0, len(returns), size=(1000, len(returns))):
        _, _, w = tangencies(returns[indices])
        boot.append(w)
    intervals = np.nanquantile(boot, [0.05, 0.95], axis=0).transpose(1, 2, 0)
    undefined = np.isnan(np.asarray(boot)[:, :, 0]).sum(axis=0)
    return dict(rf=rf, summary=summary, impact=impact, portfolios=portfolios, capm=capm,
                down=down_capm, down_n=len(down), intervals=intervals,
                undefined=undefined)


def problem5():
    aligned, factors, mom, diagnostics = factor_inputs()
    end = factors.index.max()
    start = end - pd.DateOffset(years=10)
    recent = factors.loc[(factors.index > start) & (factors.index <= end)]
    columns = ["Mkt-RF", "SMB", "HML", "Mom"]
    sample_premia = aligned[columns].mean().to_numpy()
    recent_premia = recent[columns].mean().to_numpy()
    recent_rf = recent.RF.mean()
    worst = mom.Mom.idxmin()
    shock = mom.at[worst, "Mom"]
    market = aligned["Mkt-RF"]
    funds = {}
    for name, diagnostic in diagnostics.items():
        y = aligned[name] - aligned.RF
        models = {"CAPM": diagnostic["capm"],
                  "FF3": ols(y, aligned[columns[:3]])[0],
                  "Carhart": ols(y, aligned[columns])[0]}
        alpha, *beta = models["Carhart"]["coef"]
        expected = recent_rf + beta @ recent_premia
        loss = -shock * beta[-1]
        # Omitted-factor intercepts come from projecting each omitted factor
        # on the market with a constant; the market loading also changes.
        auxiliary = [ols(aligned[col], market)[0]["coef"][0] for col in columns[1:]]
        alpha_attribution = np.array(beta[1:]) * auxiliary
        assert np.isclose(models["CAPM"]["coef"][0], alpha + alpha_attribution.sum())
        realized = aligned[name]
        funds[name] = dict(mean=diagnostic["mean"], sd=diagnostic["sd"],
                           sharpe=diagnostic["sharpe"], corr=diagnostic["corr"], models=models,
                           annual_no_alpha=DAYS * expected, annual_with_alpha=DAYS * (expected + alpha),
                           stress_dollars=loss * 10_000_000, stress_sd=loss / realized.std(ddof=1),
                           realized=DAYS * realized.mean(),
                           realized_geo=np.expm1(DAYS * np.log1p(realized).mean()),
                           alpha_attribution=DAYS * alpha_attribution)
    return dict(n=len(aligned), sample_start=aligned.index.min().date().isoformat(),
                sample_end=aligned.index.max().date().isoformat(),
                recent_start=recent.index.min().date().isoformat(), recent_end=end.date().isoformat(),
                recent_n=len(recent), sample_premia=DAYS * sample_premia,
                recent_premia=DAYS * recent_premia, recent_rf=DAYS * recent_rf,
                shock_date=worst.date().isoformat(), shock=shock, funds=funds)


def main():
    FIGURES.mkdir(exist_ok=True)
    results = {}
    for number, calculate in enumerate((problem1, problem2, problem3, problem4, problem5), 1):
        results[f"problem{number}"] = calculate()
        print(f"Problem {number} complete.", flush=True)
    # Convert NumPy arrays and scalars for the full-precision result file.
    with (BASE / "results.json").open("w", encoding="utf-8") as out:
        json.dump(results, out, default=lambda x: x.tolist(), indent=2, allow_nan=False)
        out.write("\n")
    print(f"Numerical results and figures written to {BASE}")


if __name__ == "__main__":
    main()
