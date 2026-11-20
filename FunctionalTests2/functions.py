import numpy as np
import pandas as pd
from scipy import optimize, stats


def fit_t(sample):
    # Standardization makes the three-parameter optimization better conditioned.
    center, scale = np.mean(sample), np.std(sample)
    df, loc, width = stats.t.fit((sample - center) / scale)
    return stats.t(df, center + loc * scale, width * scale)


def empirical_risk(values):
    cutoff = np.quantile(values, 0.05)
    return -cutoff, -values[values <= cutoff].mean()


def portfolio_risk(returns, holdings):
    pnl = returns * holdings.to_numpy()
    pnl = np.column_stack([pnl, pnl.sum(axis=1)])
    rows = []
    for column, (stock, value) in enumerate(list(holdings.items()) + [('Total', holdings.sum())]):
        var, es = empirical_risk(pnl[:, column])
        rows.append([stock, var, es, var / value, es / value])
    return pd.DataFrame(rows, columns=['Stock', 'VaR95', 'ES95', 'VaR95_Pct', 'ES95_Pct'])


def optimize_weights(cov, budget=None, excess=None, bounds=(0, 1)):
    n = len(cov)
    if budget is not None:
        # The first order conditions give x_i (C x)_i = budget_i.
        initial = np.sqrt(budget / np.diag(cov))
        result = optimize.minimize(
            lambda x: 0.5 * x @ cov @ x - budget @ np.log(x), initial,
            jac=lambda x: cov @ x - budget / x, bounds=[(1e-10, None)] * n,
            method='L-BFGS-B', options={'ftol': 1e-15, 'gtol': 1e-10})
    else:
        result = optimize.minimize(
            lambda w: -(w @ excess) / np.sqrt(w @ cov @ w), np.full(n, 1/n),
            bounds=[bounds] * n, constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
            method='SLSQP', options={'ftol': 1e-13})
    if not result.success:
        raise ValueError(result.message)
    return result.x / result.x.sum()


def attribution(assets, weights, factors=None, beta=None):
    stock_only = factors is None
    if stock_only:
        factors, beta = assets, np.eye(len(weights))
    daily, contributions = [], []
    for ret, factor in zip(assets.to_numpy(), factors.to_numpy()):
        portfolio = weights @ ret
        parts = (weights @ beta) * factor
        daily.append(portfolio)
        contributions.append([*parts, portfolio - parts.sum()])
        weights = weights * (1 + ret) / (1 + portfolio)
    daily, parts = np.array(daily), np.array(contributions)
    raw = np.column_stack([factors, parts[:, -1], daily])
    total = np.expm1(np.log1p(raw).sum(axis=0))
    def log_ratio(x):
        return np.divide(np.log1p(x), x, out=np.ones_like(x), where=x != 0)
    linked = (log_ratio(daily) / log_ratio(total[-1])) @ parts
    vol = daily.std(ddof=1)
    component_vol = (daily - daily.mean()) @ parts / ((len(daily) - 1) * vol)
    result = pd.DataFrame([total, [*linked, total[-1]], [*component_vol, vol]],
                          columns=[*factors.columns, 'Alpha', 'Portfolio'])
    result.insert(0, 'Value', ['TotalReturn', 'Return Attribution', 'Vol Attribution'])
    return result.drop(columns='Alpha') if stock_only else result


def european(sign, spot, strike, time, rate, carry, vol):
    root = np.sqrt(time)
    d1 = (np.log(spot / strike) + (carry + vol**2 / 2) * time) / (vol * root)
    d2 = d1 - vol * root
    growth, discount = np.exp((carry - rate)*time), np.exp(-rate*time)
    delta = sign * growth * stats.norm.cdf(sign*d1)
    bond = strike * discount * stats.norm.cdf(sign*d2)
    value = spot * delta - sign * bond
    vega = spot * growth * stats.norm.pdf(d1) * root
    theta = -vega * vol / (2*time) - (carry-rate)*spot*delta - sign*rate*bond
    return [value, delta, vega/(spot**2 * vol*time), vega, sign*time*bond, theta]


def american(sign, spot, strike, time, rate, carry, vol, steps=500, dividends=()):
    dt = time / steps
    up = np.exp(vol * np.sqrt(dt))
    prob = (np.exp(carry*dt) - 1/up) / (up - 1/up)
    disc = np.exp(-rate*dt)
    # Price several subtrees together. A cash dividend starts a fresh subtree.
    def subtree(starts, remaining, payments):
        length = payments[0][0] if payments else remaining
        prices = starts[:, None] * up**(2*np.arange(length+1) - length)
        values = np.maximum(sign * (prices-strike), 0)
        if payments:
            date, cash = payments[0]
            later = [(t-date, amt) for t, amt in payments[1:]]
            for i, row in enumerate(prices):
                after = subtree(row-cash, remaining-date, later)
                values[i] = np.maximum(values[i], after)
        for _ in range(length):
            prices = prices[:, :-1] * up
            hold = disc * ((1-prob)*values[:, :-1] + prob*values[:, 1:])
            values = np.maximum(sign*(prices-strike), hold)
        return values[:, 0]
    return subtree(np.array([spot]), steps, dividends)[0]


def american_greeks(sign, spot, strike, time, rate, carry, vol):
    args = np.array([spot, strike, time, rate, carry, vol])
    def bumped(index, step):
        params = args.copy()
        params[index] += step
        return american(sign, *params)
    def slope(index):
        step = 1e-5 * max(1, abs(args[index]))
        return (bumped(index, step) - bumped(index, -step))/(2*step)
    value = american(sign, *args)
    gamma = (bumped(0, 1.5) + bumped(0, -1.5) - 2*value) / 1.5**2
    # The course tree differentiates remaining time and holds carry fixed for rho.
    return [value, slope(0), gamma, slope(5), slope(3), slope(2)]


def kendall_matrix(sample):
    corr = np.sin(np.pi/2 * sample.corr(method='kendall').to_numpy())
    eig, vec = np.linalg.eigh(corr)
    if eig.min() < 1e-8:
        corr = (vec * np.maximum(eig, 1e-8)) @ vec.T
        corr /= np.outer(np.sqrt(corr.diagonal()), np.sqrt(corr.diagonal()))
    return corr


def fit_df(loglike):
    result = optimize.minimize_scalar(lambda inv: -loglike(1/inv),
                                      bounds=(.01, .49), method='bounded',
                                      options={'xatol': 1e-10})
    return 1 / result.x, -result.fun
