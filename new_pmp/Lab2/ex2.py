import numpy as np
import matplotlib.pyplot as plt

n = 1000
lambdas = [1, 2, 5, 10]
rng = np.random.default_rng(42)

#for the 4 fixed mean samples
samples_fixed = []
for lam in lambdas:
    draws = rng.poisson(lam, n)
    samples_fixed.append(draws)

#for the randomized mean sample
choices = rng.choice(lambdas, n) #array that stores different mean values
samples_rand = np.empty(n, dtype=int)
for i in range(n):
    lam = choices[i]
    samples_rand[i] = rng.poisson(lam) #draw one poissn number with that specific mean

def plot_hist(x, title):
    bins = np.arange(0, max(20, x.max() + 2))
    plt.figure()
    plt.hist(x, bins=bins, density=1, edgecolor='black')
    plt.title(title)
    plt.xlabel('counts per hour')
    plt.ylabel('empirical density')
    plt.tight_layout()

#plot the four fixed mean datasets
for i, lam in enumerate(lambdas):
    plot_hist(samples_fixed[i], f'diagrama poisson({lam})')

#plot the randomized mean dataset
plot_hist(samples_rand, 'diagrama rand - lambda in {1,2,5,10}')

plt.show()

#λ fix: distribuția este concentrată în jurul lui λ, cu probabilități care scad spre extremități

#λ randomizat: ai un amestec (mixture) de 4 distribuții Poisson; histograma devine mai lată,
#cu „umflături” influențate de valorile 1, 2, 5, 10 → varianta totală crește

#În modelare reală, un λ fix poate fi prea simplu; un λ aleatoriu capturează mai bine variabilitatea/ incertitudinea 
#între zile diferite ale call-center-ului