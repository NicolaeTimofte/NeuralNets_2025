import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import multiprocessing as mp

def main():
    np.random.seed(42)

    # Distribuții a priori
    mu_prior_mean = 5
    mu_prior_sd = 2
    sigma_prior_sd = 1

    # Generăm 100 de valori ale timpului mediu de așteptare
    mu_samples = np.random.normal(mu_prior_mean, mu_prior_sd, 100)
    sigma_samples = np.abs(np.random.normal(0, sigma_prior_sd, 100))  # HalfNormal

    # Generăm timpii de așteptare conform distribuției normale a priori
    waiting_times = np.random.normal(mu_samples, sigma_samples)

    plt.hist(waiting_times, bins=20, color='skyblue', edgecolor='k')
    plt.xlabel("Timp de așteptare (minute)")
    plt.ylabel("Frecvență")
    plt.title("Simulare timpi de așteptare - distribuție a priori")
    plt.show()

    # Să presupunem că avem niște date reale de timp de așteptare
    data = waiting_times[:50]  # folosim 50 dintre timpii generați pentru exemplu

    with pm.Model() as model:
        # Priori
        mu = pm.Normal("mu", mu=5, sigma=2)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Likelihood
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=data)

        # Inferență MCMC
        trace = pm.sample(2000, tune=2000, target_accept=0.9, random_seed=42)

        # Sumara posteriorului
        summary = az.summary(trace, var_names=["mu", "sigma"], hdi_prob=0.95)
        print(summary)
    az.plot_posterior(trace, var_names=["mu"], hdi_prob=0.95)
    plt.title("Distribuția a posteriori a lui μ (timp mediu de așteptare)")
    plt.xlabel("μ (minute)")
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()   # safe pe Windows
    main()