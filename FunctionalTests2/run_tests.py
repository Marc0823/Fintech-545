from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from functions import (american, american_greeks, attribution, empirical_risk, european,
                       fit_df, fit_t, kendall_matrix, optimize_weights, portfolio_risk)

ROOT = Path(__file__).resolve().parent


def read(name):
    return pd.read_csv(ROOT / 'data' / f'test{name}.csv')


def risk_tests():
    normal_data = read('7_1').iloc[:, 0]
    normal = stats.norm(normal_data.mean(), normal_data.std())
    student = fit_t(read('7_2').iloc[:, 0])
    rng = np.random.default_rng(8)
    for idx, model in enumerate([normal, student, student], start=1):
        for offset, label in [(0, 'VaR'), (3, 'ES')]:
            if idx == 3:
                sim = model.rvs(10000, random_state=rng)
                risk = empirical_risk(sim)[int(offset > 0)]
                mean = sim.mean()
            else:
                cutoff = model.ppf(.05)
                risk = -cutoff if offset == 0 else -model.expect(ub=cutoff, conditional=True)
                mean = model.mean()
            yield f'8_{idx+offset}', pd.DataFrame({
                f'{label} Absolute': [risk], f'{label} Diff from Mean': [risk+mean]})
    data = read('9_1_returns')
    models = [stats.norm(data.A.mean(), data.A.std()), fit_t(data.B)]
    corr = data.corr(method='spearman')
    draws = np.random.default_rng(9).multivariate_normal([0, 0], corr, 100000)
    sim = np.column_stack([m.ppf(stats.norm.cdf(z)) for m, z in zip(models, draws.T)])
    holdings = read('9_1_portfolio').set_index('Stock')
    values = holdings['Holding'] * holdings['Starting Price']
    yield '9_1', portfolio_risk(sim, values.loc[data.columns])


def allocation_tests():
    for idx, last in enumerate([1, .5], start=1):
        budget = np.array([1, 1, 1, 1, last])
        weights = optimize_weights(read('5_2').to_numpy(), budget=budget)
        yield f'10_{idx}', pd.DataFrame({'W': weights})
    for idx, bounds in enumerate([(0, 1), (.1, .5)], start=3):
        weights = optimize_weights(read('5_3').to_numpy(), excess=read('10_3_means').Mean-.04,
                                   bounds=bounds)
        yield f'10_{idx}', pd.DataFrame({'W': weights})
    yield '11_1', attribution(read('11_1_returns'), read('11_1_weights').W.to_numpy())
    yield '11_2', attribution(read('11_2_stock_returns'), read('11_2_weights').W.to_numpy(),
                              read('11_2_factor_returns'), read('11_2_beta').iloc[:, 1:].to_numpy())


def option_tests():
    for idx, price in enumerate([european, american_greeks, american], start=1):
        rows = []
        for _, row in read('12_3' if idx == 3 else '12_1').dropna(subset=['ID']).iterrows():
            sign = 1 if row['Option Type'] == 'Call' else -1
            args = [sign, row.Underlying, row.Strike, row.DaysToMaturity/row.DayPerYear,
                    row.RiskFreeRate]
            if idx == 3:
                dates = [2*int(d) for d in row.DividendDates.split(',')]
                cash = [float(v) for v in row.DividendAmts.split(',')]
                output = [price(*args, row.RiskFreeRate, row.ImpliedVol, int(2*row.DaysToMaturity),
                                list(zip(dates, cash)))]
            else:
                output = price(*args, row.RiskFreeRate-row.DividendRate, row.ImpliedVol)
            rows.append([int(row.ID), *output])
        fields = ['ID', 'Value'] if idx == 3 else ['ID', 'Value', 'Delta', 'Gamma', 'Vega', 'Rho', 'Theta']
        yield f'12_{idx}', pd.DataFrame(rows, columns=fields)


def copula_tests():
    data = read('13_returns')
    corr = kendall_matrix(data)
    columns = [f'x{i+1}' for i in range(len(corr))]
    yield '13_1', pd.DataFrame(corr, columns=columns)
    mean, sd = data.mean(), data.std()
    yield '13_2', pd.DataFrame({'mu': mean.to_numpy()})
    cov = np.outer(sd, sd)*corr
    def scale(df):
        return cov*(df-2)/df
    def joint_ll(df):
        return stats.multivariate_t.logpdf(data, loc=mean, shape=scale(df), df=df).sum()
    df, ll = fit_df(joint_ll)
    yield '13_3', pd.DataFrame(scale(df), columns=columns)
    yield '13_4', pd.DataFrame({'nu': [df], 'll': [ll]})
    margins = [fit_t(data[c]) for c in data]
    uniform = np.column_stack([m.cdf(data[c]) for c, m in zip(data, margins)])
    normal = stats.norm.ppf(uniform)
    gauss_ll = stats.multivariate_normal.logpdf(normal, cov=corr).sum() - stats.norm.logpdf(normal).sum()
    yield '13_5', pd.DataFrame({'ll': [gauss_ll]})
    def copula_ll(df):
        z = stats.t.ppf(uniform, df)
        return stats.multivariate_t.logpdf(z, shape=corr, df=df).sum() - stats.t.logpdf(z, df).sum()
    df, ll = fit_df(copula_ll)
    yield '13_6', pd.DataFrame({'nu': [df], 'll': [ll]})
    rows = []
    for name, score, k in [('Gaussian', gauss_ll, 0), ('T', ll, 1)]:
        aicc = -2*score + 2*k + 2*k*(k+1)/(len(data)-k-1)
        rows.append([name, score, k, aicc, -2*score + k*np.log(len(data))])
    yield '13_7', pd.DataFrame(rows, columns=['Copula', 'LL', 'K', 'AICC', 'BIC'])
    i, j = np.triu_indices(len(corr), k=1)
    rho = corr[i, j]
    tail = 2*stats.t.cdf(-np.sqrt((df+1)*(1-rho)/(1+rho)), df+1)
    yield '13_8', pd.DataFrame({'I': i+1, 'J': j+1, 'Rho': rho, 'Lambda': tail})
    rng = np.random.default_rng(13)
    z = rng.multivariate_normal(np.zeros(len(corr)), corr, 100000)
    u = stats.t.cdf(z / np.sqrt(rng.chisquare(df, (len(z), 1))/df), df)
    sim = np.column_stack([m.ppf(col) for m, col in zip(margins, u.T)])
    values = read('13_portfolio').set_index('Stock').currentValue.loc[data.columns]
    yield '13_9', portfolio_risk(sim, values)


def main():
    results = ROOT / 'results'
    results.mkdir(exist_ok=True)
    report = []
    for group in [risk_tests, allocation_tests, option_tests, copula_tests]:
        for key, actual in group():
            expected = read('out' + key)
            actual.to_csv(results / f'calculated_{key}.csv', index=False)
            rtol, atol = 0.001, 1e-7
            if key in ('8_3', '8_6'):
                rtol, atol = 0, .005  # independent 10,000-draw tail estimates
            if key in ('9_1', '13_9'):
                rtol = .035  # independent 100,000-draw portfolio tails
            ok = actual.columns.equals(expected.columns) and actual.shape == expected.shape
            if ok:
                for c in expected:
                    if pd.api.types.is_numeric_dtype(expected[c]):
                        ok &= np.allclose(actual[c], expected[c], rtol=rtol, atol=atol)
                    else:
                        ok &= np.array_equal(actual[c], expected[c])
            report.append([key.replace('_', '.'), 'PASS' if ok else 'FAIL', rtol, atol])
    report = pd.DataFrame(report, columns=['Test', 'Status', 'rtol', 'atol'])
    report.to_csv(results / 'comparison_report.csv', index=False)
    print(report.to_string(index=False))
    return int((report.Status != 'PASS').any())


if __name__ == '__main__':
    raise SystemExit(main())
