import networkx as nx
import matplotlib.pyplot as plt
import itertools, math
import pandas as pd

#a
# definim graful MRF
G = nx.Graph()
G.add_edges_from([
    ("A1","A2"), ("A1","A3"),
    ("A2","A4"), ("A2","A5"),
    ("A3","A4"),
    ("A4","A5")
])

# desen
plt.figure(figsize=(5,4))
pos = nx.spring_layout(G, seed=0)
nx.draw(G, pos, with_labels=True, node_size=1200)
plt.show()

# clici maximale
max_cliques = list(nx.find_cliques(G))
print("Maximal cliques:", max_cliques)


#b
variables = ["A1","A2","A3","A4","A5"]
idx = {f"A{i}": i for i in range(1,6)} #cream un dictionary pentru indecsi: A1 -> 1, A2 -> 2 ...

# clici maximale
cliques = [
    ("A1","A2"),
    ("A1","A3"),
    ("A3","A4"),
    ("A2","A4","A5")
]

#assign e dictionary care retine pentru fiecare atribut valoarea sa
def phi(clique, assign):
    return math.exp(sum(idx[v] * assign[v] for v in clique))

#probabilitatea nenormalizata(fara pe Z) pentru o configuratie completa
def unnorm_prob(assign):
    p = 1.0
    for c in cliques:
        p *= phi(c, assign)
    return p

rows = []
for vals in itertools.product([-1,1], repeat=5):    #generezi toate configurațiile posibile ale celor 5 variabile
    assign = dict(zip(variables, vals))             #assign e dictionary care retine pentru fiecare atribut valoarea sa (A1 -> 1, A2 -> -1, etc)
    up = unnorm_prob(assign)
    rows.append([*vals, up])

df = pd.DataFrame(rows, columns=variables + ["unnorm"])
Z = df["unnorm"].sum()  #suma valorilor nenormalizate ale tuturor configuratiilor
df["prob"] = df["unnorm"] / Z #adaugam coloana prob in tabel

df_sorted = df.sort_values("prob", ascending=False).reset_index(drop=True)  #sortam tabelul descrescator
print(df_sorted.head(8)) #afisam cele mai bune 8 configuratii
print("\nMAP configuration:")
print(df_sorted.loc[0, variables].to_dict(), "with prob", df_sorted.loc[0,"prob"]) #o afisam pe cea mai buna(cu probabilitatea cea mai mare)