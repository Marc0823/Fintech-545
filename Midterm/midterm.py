
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.stats.sampling import NumericalInversePolynomial

ROOT = Path(__file__).resolve().parent
DATA = ROOT if (ROOT / 'problem1.csv').is_file() else ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)
STOCKS = [f'S{i}' for i in range(1, 11)]
SEED = 5452026
N_SIM = 100000
MARGINALS = {
    'S1': 'normal', 'S2': 't', 'S3': 'nig', 'S4': 'normal', 'S5': 't',
    'S6': 'nig', 'S7': 'normal', 'S8': 't', 'S9': 'nig', 'S10': 't'
}
COPULA = 'gaussian'


def save_json(name, value):
    def encode(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    (RESULTS / name).write_text(json.dumps(value, indent=2, default=encode), encoding='utf-8')


def higham(correlation, tol=1e-12, max_iter=10000):
    y = np.array(correlation, dtype=float)
    delta = np.zeros_like(y)
    for iteration in range(1, max_iter + 1):
        previous = y.copy()
        residual = y - delta
        eigenvalues, vectors = np.linalg.eigh((residual + residual.T) / 2)
        positive = (vectors * np.maximum(eigenvalues, 0)) @ vectors.T
        delta = positive - residual
        y = positive.copy()
        np.fill_diagonal(y, 1.0)
        if (np.linalg.norm(y - previous, 'fro') < tol
                and np.linalg.eigvalsh(y).min() >= -tol):
            return y, iteration
    raise RuntimeError('Higham did not converge.')


def load_prices(filename):
    frame = pd.read_csv(DATA / filename)
    dates = pd.to_datetime(frame.pop('Date'))
    assert dates.is_monotonic_increasing and not dates.duplicated().any()
    assert np.isfinite(frame.to_numpy()).all() and (frame > 0).all().all()
    returns = frame.pct_change(fill_method=None).iloc[1:]
    return frame, returns


# Problem 1(a): pairwise correlations, with no missing-value imputation.
def problem1():
    x = pd.read_csv(DATA / 'problem1.csv')
    r = x.corr(method='pearson').to_numpy()
    names = list(x.columns)
    counts = x.notna().astype(int).T @ x.notna().astype(int)
    pd.DataFrame(r, index=names, columns=names).to_csv(RESULTS / 'q1a_correlation.csv')
    counts.to_csv(RESULTS / 'q1a_pairwise_counts.csv')

    # Problem 1(b): each sample variance uses all observations in its own series.
    v = x.var(ddof=1).to_numpy()
    pd.DataFrame({'Stock': names, 'n': x.count().to_numpy(), 'variance': v,
                  'standard_deviation': np.sqrt(v)}).to_csv(RESULTS / 'q1b_variances.csv', index=False)

    # Problem 1(c): eigenvalues of the pairwise correlation matrix.
    original_eigenvalues = np.linalg.eigvalsh(r)
    pd.DataFrame({'eigenvalue': original_eigenvalues}).to_csv(RESULTS / 'q1c_eigenvalues.csv', index=False)

    # Problem 1(d): Higham projection onto PSD matrices with unit diagonal.
    repaired, iterations = higham(r) if original_eigenvalues.min() < -1e-12 else (r.copy(), 0)
    repaired_eigenvalues = np.linalg.eigvalsh(repaired)
    pd.DataFrame(repaired, index=names, columns=names).to_csv(RESULTS / 'q1d_higham_correlation.csv')
    pd.DataFrame({'eigenvalue': repaired_eigenvalues}).to_csv(RESULTS / 'q1d_higham_eigenvalues.csv', index=False)
    changes = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            changes.append([names[i], names[j], r[i,j], repaired[i,j], repaired[i,j]-r[i,j]])
    change_table = pd.DataFrame(changes, columns=['series1','series2','original','repaired','change'])
    change_table['absolute_change'] = change_table.change.abs()
    change_table.sort_values('absolute_change', ascending=False).to_csv(RESULTS / 'q1d_changes.csv', index=False)

    # Problem 1(e): covariance from repaired correlations and full-column variances.
    covariance = repaired * np.sqrt(np.outer(v, v))
    pd.DataFrame(covariance, index=names, columns=names).to_csv(RESULTS / 'q1e_covariance.csv')
    assert np.allclose(np.diag(covariance), v, atol=1e-15)
    assert repaired_eigenvalues.min() > -1.1e-12
    assert np.allclose(repaired, repaired.T, atol=1e-14)
    save_json('q1_diagnostics.json', {'iterations': iterations,
              'original_min_eigenvalue': original_eigenvalues.min(),
              'repaired_min_eigenvalue': repaired_eigenvalues.min(),
              'frobenius_change': np.linalg.norm(repaired-r, 'fro')})
    print('Problem 1 tables saved.', flush=True)


# Problem 2(a): arithmetic returns, OLS, and unconditional normal VaR.
def problem2():
    prices, ret = load_prices('problem2.csv')
    x, y = ret.SPY.to_numpy(), ret.XYZ.to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    intercept, beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - intercept - beta*x
    n = len(x)
    market_mean = x.mean()
    market_variance = x.var(ddof=1)
    residual_variance = residuals @ residuals / (n - 2)
    mu = intercept + beta*market_mean
    sigma = np.sqrt(beta**2*market_variance + residual_variance)
    # Model assumption in code: SPY return and the regression error are independent.
    assert abs(residuals.mean()) < 1e-12
    assert abs(x @ residuals) < 1e-10
    assert abs(mu - y.mean()) < 1e-12
    params = {'n': n, 'intercept': intercept, 'beta': beta,
              'market_mean': market_mean, 'market_variance_ddof1': market_variance,
              'residual_variance_SSE_over_n_minus_2': residual_variance,
              'XYZ_model_mean': mu, 'XYZ_model_sd': sigma,
              'last_XYZ_price': prices.XYZ.iloc[-1], 'shares': 100,
              'position_value': 100*prices.XYZ.iloc[-1]}
    save_json('q2_parameters.json', params)
    rows = []
    for confidence in [.95, .99]:
        tail = 1-confidence
        z = stats.norm.ppf(tail)
        var = -mu - sigma*z
        # Problem 2(b): normal expected shortfall, measured from zero.
        es = -mu + sigma*stats.norm.pdf(z)/tail
        rows.append([confidence, tail, var*100, es*100])
    pd.DataFrame(rows, columns=['confidence','tail_probability','VaR_percent','ES_percent']).to_csv(
        RESULTS / 'q2_risk_percent.csv', index=False)
    print('Problem 2 tables saved.', flush=True)


def information_criteria(loglike, k, n):
    aic = -2*loglike + 2*k
    return aic + 2*k*(k+1)/(n-k-1), -2*loglike + k*np.log(n)


def fit_nig(y):
    # Standardized observations improve numerical scaling; transform back afterward.
    # Coordinates are log(gamma), atanh(beta/alpha), location, log(delta).
    def unpack(theta):
        gamma = np.exp(theta[0])
        rho = np.tanh(theta[1])
        a = gamma / np.sqrt(1-rho*rho)
        return a, a*rho, theta[2], np.exp(theta[3])
    def objective(theta):
        a,b,loc,scale = unpack(theta)
        logpdf = stats.norminvgauss.logpdf(y, a, b, loc=loc, scale=scale)
        return -logpdf.sum() if np.isfinite(logpdf).all() else 1e100
    starts = [[0,0,0,0], [np.log(3),0,0,np.log(np.sqrt(3))],
              [np.log(.5),0,0,np.log(np.sqrt(.5))]]
    skew, kurt = stats.skew(y), stats.kurtosis(y)
    if kurt > 0 and skew*skew/kurt < .6:
        ratio2 = skew*skew/(3*kurt-4*skew*skew)
        rho = np.sign(skew)*np.sqrt(ratio2)
        product = 3*(1+4*ratio2)/kurt
        alpha = np.sqrt(product/(np.var(y)*(1-ratio2)**2))
        beta = rho*alpha
        gamma = np.sqrt(alpha*alpha-beta*beta)
        delta = product/gamma
        loc = y.mean()-delta*beta/gamma
        starts.append([np.log(delta*gamma),np.arctanh(rho),loc,np.log(delta)])
    bounds = [(-7, 12),(-4,4),(-30,30),(-10,8)]
    candidates = []
    for start in starts:
        start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        result = optimize.minimize(objective, start, method='L-BFGS-B', bounds=bounds,
                                   options={'maxiter':2500, 'ftol':1e-12, 'gtol':1e-6})
        if np.isfinite(result.fun):
            candidates.append(result)
    converged = [result for result in candidates if result.success]
    if not converged:
        raise RuntimeError('No NIG optimization start converged.')
    best = min(converged, key=lambda result: result.fun)
    # One derivative-free refinement checks the best local optimum.
    refined = optimize.minimize(objective, best.x, method='Nelder-Mead', bounds=bounds,
                                 options={'maxiter':5000,'xatol':1e-8,'fatol':1e-8})
    if refined.success and refined.fun < best.fun:
        best = refined
    assert best.success, best.message
    near_boundary = any(abs(value-lo)<1e-3 or abs(value-hi)<1e-3
                        for value,(lo,hi) in zip(best.x,bounds))
    return unpack(best.x), {'success':bool(best.success),'objective':float(best.fun),
                           'near_search_boundary':near_boundary,
                           'refinement_success':bool(refined.success),
                           'refinement_objective':float(refined.fun)}


# Problem 3(a): fit all three families; output scores without selecting a model.
def fit_margins(ret):
    fits, records, score_rows = {}, [], []
    for stock in STOCKS:
        x = ret[stock].to_numpy()
        center, spread = x.mean(), x.std(ddof=0)
        y = (x-center)/spread
        normal = {'mu':center, 'sigma':spread}
        nu, tloc, tscale = stats.t.fit(y)
        if nu <= 2.0001:
            def t_objective(theta):
                return -stats.t.logpdf(y, 2.0001+np.exp(theta[0]),
                                      loc=theta[1],scale=np.exp(theta[2])).sum()
            fitted = optimize.minimize(t_objective,[np.log(3),0,0], method='BFGS')
            nu,tloc,tscale = 2.0001+np.exp(fitted.x[0]),fitted.x[1],np.exp(fitted.x[2])
        student = {'nu':nu,'mu':center+spread*tloc,'scale':spread*tscale}
        (a,b,nloc,nscale), diagnostic = fit_nig(y)
        delta = spread*nscale
        nig = {'alpha':a/delta,'beta':b/delta,'delta':delta,'mu':center+spread*nloc}
        models = {'normal':normal, 't':student, 'nig':nig}
        fits[stock] = models
        scores = {'Stock':stock}
        for model, parameters in models.items():
            distribution = make_distribution(model,parameters)
            ll = float(distribution.logpdf(x).sum())
            k = {'normal':2,'t':3,'nig':4}[model]
            aicc,bic = information_criteria(ll,k,len(x))
            record = {'Stock':stock,'model':model,**parameters,'log_likelihood':ll,
                      'parameters':k,'AICc':aicc,'BIC':bic}
            records.append(record)
            scores[model+'_AICc'], scores[model+'_BIC'] = aicc,bic
        fits[stock]['nig_diagnostics'] = diagnostic
        score_rows.append(scores)
        print(stock+' parameter fits saved.',flush=True)
    save_json('q3a_fits.json',fits)
    pd.DataFrame(records).to_csv(RESULTS / 'q3a_all_parameters.csv',index=False)
    pd.DataFrame(score_rows).to_csv(RESULTS / 'q3a_information_criteria.csv',index=False)
    return fits


def make_distribution(name,p):
    if name == 'normal':
        return stats.norm(loc=p['mu'],scale=p['sigma'])
    if name == 't':
        return stats.t(p['nu'],loc=p['mu'],scale=p['scale'])
    if name == 'nig':
        return stats.norminvgauss(p['alpha']*p['delta'],p['beta']*p['delta'],
                                 loc=p['mu'],scale=p['delta'])
    raise ValueError('Model must be normal, t, or nig.')


# Problem 3(b): fit both copulas using the selected marginal distributions.
def fit_copulas(ret, fits, choices):
    u = np.column_stack([make_distribution(choices[s],fits[s][choices[s]]).cdf(ret[s])
                         for s in STOCKS])
    assert np.isfinite(u).all()
    u = np.clip(u,1e-10,1-1e-10)
    tau = np.eye(len(STOCKS))
    for i in range(len(STOCKS)):
        for j in range(i):
            tau[i,j] = tau[j,i] = stats.kendalltau(ret.iloc[:,i],ret.iloc[:,j]).statistic
    correlation = np.sin(np.pi*tau/2)
    original_min_eigenvalue = np.linalg.eigvalsh(correlation).min()
    if original_min_eigenvalue < -1e-12:
        correlation,_ = higham(correlation)
    # Likelihood evaluation requires a nonsingular correlation matrix.
    jitter = max(0.,1e-10-np.linalg.eigvalsh(correlation).min())
    if jitter:
        correlation = (correlation+jitter*np.eye(len(STOCKS)))/(1+jitter)
    pd.DataFrame(correlation,index=STOCKS,columns=STOCKS).to_csv(RESULTS/'q3b_copula_correlation.csv')
    z = stats.norm.ppf(u)
    gaussian_ll = float(np.sum(stats.multivariate_normal.logpdf(z,cov=correlation)
                               -stats.norm.logpdf(z).sum(axis=1)))
    def t_ll(nu):
        transformed = stats.t.ppf(u,nu)
        return float(np.sum(stats.multivariate_t.logpdf(transformed,shape=correlation,df=nu)
                             -stats.t.logpdf(transformed,nu).sum(axis=1)))
    theta = np.linspace(.01,.49,200)
    likelihood = np.array([t_ll(1/t) for t in theta])
    index = int(np.argmax(likelihood))
    lo,hi = theta[max(0,index-1)],theta[min(len(theta)-1,index+1)]
    optimum = optimize.minimize_scalar(lambda t:-t_ll(1/t),bounds=(lo,hi),method='bounded',
                                       options={'xatol':1e-8})
    candidates = [(float(likelihood[index]),float(theta[index])),(-float(optimum.fun),float(optimum.x))]
    t_loglike,theta_best = max(candidates)
    nu = 1/theta_best
    records = []
    for model,ll,k in [('gaussian',gaussian_ll,0),('t',t_loglike,1)]:
        aicc,bic = information_criteria(ll,k,len(ret))
        records.append({'copula':model,'nu':nu if model=='t' else np.nan,
                        'log_likelihood':ll,'conditional_parameters':k,'AICc':aicc,'BIC':bic})
    pd.DataFrame(records).to_csv(RESULTS/'q3b_copula_scores.csv',index=False)
    pd.DataFrame({'nu':1/theta,'log_likelihood':likelihood}).to_csv(RESULTS/'q3b_t_profile.csv',index=False)
    save_json('q3b_parameters.json', {'correlation':correlation,'t_nu':nu,
              'marginal_choices':choices,'original_min_eigenvalue':original_min_eigenvalue,
              'diagonal_jitter':jitter,'nu_search_bounds':[1/.49,100.]})
    print('Problem 3(b) copula score table saved.',flush=True)
    return correlation,nu


def empirical_risk(pnl,tail=.05):
    ordered = np.sort(np.asarray(pnl))
    lower,upper = int(np.floor(len(pnl)*tail)),int(np.ceil(len(pnl)*tail))
    if lower<1:
        raise ValueError('Insufficient simulated observations.')
    quantile = (ordered[lower-1]+ordered[upper-1])/2
    return -quantile,-ordered[ordered<=quantile].mean()


# Problem 3(c): simulate the selected copula.
def simulate_portfolio(prices, fits, choices, copula, correlation, nu):
    if copula not in ['gaussian','t']:
        raise ValueError('Copula must be gaussian or t.')
    assert N_SIM >= 5000
    rng = np.random.default_rng(SEED)
    z = rng.standard_normal((N_SIM,len(STOCKS))) @ np.linalg.cholesky(correlation).T
    if copula == 't':
        # One common chi-square variable per scenario, not one per stock.
        z /= np.sqrt(rng.chisquare(nu,size=N_SIM)/nu)[:,None]
        u = stats.t.cdf(z,nu)
    else:
        u = stats.norm.cdf(z)
    u = np.clip(u,1e-10,1-1e-10)
    simulated = np.empty_like(u)
    inversion_checks = {}
    for j,stock in enumerate(STOCKS):
        name,p = choices[stock],fits[stock][choices[stock]]
        distribution = make_distribution(name,p)
        if name == 'nig':
            standard = stats.norminvgauss(p['alpha']*p['delta'],p['beta']*p['delta'])
            # The integrator probes infinite endpoints; verify the finite CDF below.
            with np.errstate(invalid='ignore',over='ignore'):
                inverse = NumericalInversePolynomial(standard,u_resolution=1e-10)
            simulated[:,j] = p['mu']+p['delta']*inverse.ppf(u[:,j])
            check_u = np.r_[np.geomspace(1e-7,.01,20),np.linspace(.02,.98,41),
                             1-np.geomspace(1e-7,.01,20)]
            error = np.max(np.abs(standard.cdf(inverse.ppf(check_u))-check_u))
            assert error<1e-7, (stock,error)
            inversion_checks[stock] = float(error)
        else:
            simulated[:,j] = distribution.ppf(u[:,j])
    assert np.isfinite(simulated).all()
    holdings = pd.read_csv(DATA/'problem3_holdings.csv').set_index('Stock').loc[STOCKS]
    position_value = holdings.Holding.to_numpy()*prices.iloc[-1][STOCKS].to_numpy()
    pnl = simulated*position_value
    positions = holdings.copy()
    positions['last_price'] = prices.iloc[-1][STOCKS]
    positions['current_value'] = position_value
    positions.to_csv(RESULTS/'q3c_positions.csv')
    rows = []
    def add_row(label,book,value,series):
        var,es = empirical_risk(series)
        batches = np.array([empirical_risk(part) for part in np.array_split(series,10)])
        se = batches.std(axis=0,ddof=1)/np.sqrt(10)
        rows.append([label,book,value,var,es,se[0],se[1]])
    for j,stock in enumerate(STOCKS):
        add_row(stock,holdings.loc[stock,'Book'],position_value[j],pnl[:,j])
    for book in sorted(holdings.Book.unique()):
        mask = holdings.Book.to_numpy()==book
        add_row('Book '+book,book,position_value[mask].sum(),pnl[:,mask].sum(axis=1))
    add_row('Total','All',position_value.sum(),pnl.sum(axis=1))
    table = pd.DataFrame(rows,columns=['Position','Book','Current_value_USD','VaR95_USD','ES95_USD',
                                      'VaR_MC_SE_USD','ES_MC_SE_USD'])
    table.to_csv(RESULTS/'q3c_risk_dollars.csv',index=False)
    save_json('q3c_simulation.json',{'draws':N_SIM,'seed':SEED,'copula':copula,
              'marginal_choices':choices,'copula_nu':nu if copula=='t' else None,
              'NIG_inverse_max_cdf_errors':inversion_checks})
    print('Problem 3(c) risk table saved.',flush=True)


def main():
    # Local input hashes make cached parameter estimates safe to reuse.
    hashes = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*.csv')}
    problem1()
    problem2()
    prices,ret = load_prices('problem3.csv')
    ret = ret[STOCKS]  # Retain observed means; this question does not request demeaning.
    cache = RESULTS/'q3a_fits.json'
    cache_hash = RESULTS/'input_hashes.json'
    if cache.exists() and cache_hash.exists() and json.loads(cache_hash.read_text())==hashes:
        fits = json.loads(cache.read_text())
        print('Reusing fits for unchanged input files.',flush=True)
    else:
        fits = fit_margins(ret)
        save_json('input_hashes.json',hashes)
    margins = MARGINALS
    correlation,nu = fit_copulas(ret,fits,margins)
    simulate_portfolio(prices,fits,margins,COPULA,correlation,nu)


if __name__ == '__main__':
    main()
