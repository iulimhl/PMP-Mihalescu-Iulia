import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import os
os.environ["PYTENSOR_FLAGS"] = "cxx="


def fit_mixture_poly(csv_path, K, seed=42):
    df = pd.read_csv(csv_path)

    t = df.iloc[:, 0].to_numpy().astype(float)
    y = df.iloc[:, 1].to_numpy().astype(float)

    t_mean, t_std = t.mean(), t.std()
    t_s = (t - t_mean) / t_std

    y_mean, y_std = y.mean(), y.std()
    y_s = (y - y_mean) / y_std

    with pm.Model() as model:
        w = pm.Dirichlet("w", a=np.ones(K))
        alpha = pm.Normal("alpha", 0.0, 2.0, shape=K)
        beta = pm.Normal("beta", 0.0, 2.0, shape=K)
        gamma = pm.Normal("gamma", 0.0, 2.0, shape=K)
        sigma = pm.HalfNormal("sigma", 1.0, shape=K)

        mu = alpha + beta * t_s[:, None] + gamma * (t_s[:, None] ** 2)

        pm.Mixture(
            "y_obs",
            w=w,
            comp_dists=pm.Normal.dist(mu=mu, sigma=sigma),
            observed=y_s,
        )

        idata = pm.sample(
            draws=2000,
            tune=2000,
            chains=4,
            target_accept=0.9,
            random_seed=seed,
        )
        idata = pm.compute_log_likelihood(idata)

    return model, idata

def summarize_params(idata):
    return az.summary(
        idata,
        var_names=["w", "alpha", "beta", "gamma", "sigma"],
        hdi_prob=0.95,
    )

def compare_models(idatas_dict, ic="loo"):
    return az.compare(idatas_dict, ic=ic)

csv_path = "date_colesterol.csv"

_, idata3 = fit_mixture_poly(csv_path, 3)
_, idata4 = fit_mixture_poly(csv_path, 4)
_, idata5 = fit_mixture_poly(csv_path, 5)

print("K=3")
print(summarize_params(idata3))
print("\nK=4")
print(summarize_params(idata4))
print("\nK=5")
print(summarize_params(idata5))

print("\nLOO")
print(compare_models({"K=3": idata3, "K=4": idata4, "K=5": idata5}, ic="loo"))

print("\nWAIC")
print(compare_models({"K=3": idata3, "K=4": idata4, "K=5": idata5}, ic="waic"))
 # 2.Pentru a determina numărul optim de subpopulații, am comparat modelele corespunzătoare valorilor 3, 4, 5
# folosind criterii Bayesiene de evaluare a performanței predictive, respectiv WAIC și LOO.
# Conform tabelului de comparație obținut, modelul cu K=5 subpopulații prezintă cea mai bună performanță,
# având valoarea WAIC minimă și, respectiv, elpd_loo maxim comparativ cu celelalte modele.
# Acest rezultat indică faptul că modelul cu 5 componente oferă cea mai bună aproximare a distribuției datelor și generalizează cel mai bine.
# Prin urmare, concluzia e că datele observate sunt cel mai bine reprezentate de 5 subpopulații.