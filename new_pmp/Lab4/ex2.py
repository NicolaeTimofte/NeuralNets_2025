import numpy as np, math
from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import BeliefPropagation

# 1) Imagine curata 5x5 (exemplu determinist)
clean = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0],
], dtype=int)

H, W = clean.shape

# 2) Zgomot: flip la ~10% pixeli
rng = np.random.default_rng(0)
noisy = clean.copy()
num_flip = max(1, int(0.10 * H * W))
flip_idx = rng.choice(H*W, num_flip, replace=False)
for k in flip_idx:
    i, j = divmod(k, W)
    noisy[i, j] = 1 - noisy[i, j]

lam = 2.0  # factorul lambda

def var_name(i, j):
    return f"x_{i}_{j}"

# 3) Definim Markov Network-ul (grid 4-neighborhood)
model = MarkovNetwork()

for i in range(H):
    for j in range(W):
        model.add_node(var_name(i, j))

edges = []
for i in range(H):
    for j in range(W):
        v = var_name(i, j)
        for di, dj in [(1,0), (0,1)]:  # doar jos si dreapta ca sa nu dublezi muchiile
            ni, nj = i+di, j+dj
            if 0 <= ni < H and 0 <= nj < W:
                u = var_name(ni, nj)
                model.add_edge(v, u)
                edges.append((v, u))

# 4) Factorii unari din termenul lambda*(xi-yi)^2
unary_factors = []
for i in range(H):
    for j in range(W):
        y = noisy[i, j]
        v = var_name(i, j)
        vals = [
            math.exp(-lam*(0-y)**2),
            math.exp(-lam*(1-y)**2)
        ]
        unary_factors.append(DiscreteFactor([v], [2], vals))

# 5) Factorii pereche din termenul (xi-xj)^2
same = math.exp(0)     # 1
diff = math.exp(-1)    # penalizare daca sunt diferiti
pair_vals = [same, diff, diff, same]  # (0,0),(0,1),(1,0),(1,1)

pair_factors = []
for v, u in edges:
    pair_factors.append(DiscreteFactor([v, u], [2,2], pair_vals))

model.add_factors(*unary_factors, *pair_factors)

# 6) MAP cu Belief Propagation
bp = BeliefPropagation(model)
map_assign = bp.map_query(list(model.nodes()))

denoised = np.zeros_like(clean)
for i in range(H):
    for j in range(W):
        denoised[i, j] = map_assign[var_name(i, j)]

print("Clean:\n", clean)
print("Noisy:\n", noisy)
print("Denoised (MAP):\n", denoised)