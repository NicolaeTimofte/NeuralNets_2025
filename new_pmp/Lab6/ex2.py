import numpy as np, matplotlib.pyplot as plt
from scipy.stats import gamma
import arviz as az

# observed data
T = 10          # hours observed
K = 180         # total calls
rate_obs = K / T

# weak Gamma prior for lambda (shape-rate)
alpha0 = 1.0
beta0  = 1.0

# posterior parameters
alpha_post = alpha0 + K
beta_post  = beta0 + T

# scipy Gamma uses scale = 1/rate
post = gamma(a=alpha_post, scale=1/beta_post)

# sample for HDI
samples = post.rvs(size=200_000, random_state=0)

hdi94 = az.hdi(samples, hdi_prob=0.94)
mode_post = (alpha_post - 1) / beta_post

print("Posterior: Gamma(alpha,beta) =", alpha_post, beta_post)
print("94% HDI =", hdi94)
print("Mode =", mode_post)

# plot posterior
xs = np.linspace(post.ppf(0.0005), post.ppf(0.9995), 600)
plt.figure()
plt.plot(xs, post.pdf(xs))
plt.title("Posterior for λ (calls per hour)")
plt.xlabel("λ")
plt.ylabel("density")
plt.tight_layout()
plt.show()