import numpy as np
from scipy.stats import invgamma
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import multiprocessing as mp

def main():
    #a
    rng = np.random.default_rng(0)

    mu0, s0 = 10.0, 5.0          # prior pt mu
    s_sigma0 = 5.0
    #a0, b0 = 3.0, 9.0            # prior pt sigma^2

    #sigma2_true = invgamma.rvs(a=a0, scale=b0)
    #sigma_true = np.sqrt(sigma2_true)
    sigma_true = abs(rng.normal(0, s_sigma0))
    mu_true = rng.normal(mu0, s0)

    x = rng.normal(mu_true, sigma_true, size=100)

    #b
    mu0_prior = 10.0   # media a priori
    s0_prior = 5.0     # cat de larg e priorul pe mu
    s_sigma_prior = 5.0  # prior slab pentru sigma

    with pm.Model() as model:
        # Priori
        mu = pm.Normal("mu", mu=mu0_prior, sigma=s0_prior)
        sigma = pm.HalfNormal("sigma", sigma=s_sigma_prior)

        # Likelihood
        x_obs = pm.Normal("x_obs", mu=mu, sigma=sigma, observed=x)

        # Posterior sampling
        trace = pm.sample(2000, tune=1000, chains=2, target_accept=0.9)

    # Rezumat numeric pentru mu
    print(az.summary(trace, var_names=["mu"]))

    #c
    az.plot_posterior(trace, var_names=["mu"], hdi_prob=0.94)
    plt.title("Posterior pentru μ")
    plt.show()


    # Prior vs posterior (optional nice check)
    with model:
        prior_pred = pm.sample_prior_predictive(5000, random_seed=1)

    az.plot_dist(prior_pred.prior["mu"], label="prior μ", color="orange")
    az.plot_dist(trace.posterior["mu"].values.flatten(), label="posterior μ")
    plt.legend()
    plt.title("Prior vs posterior pentru μ")
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()   # safe pe Windows
    main()