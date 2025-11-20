import random
from math import comb
import itertools
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#a
def onePlay(rng = random):
    starter = rng.choice([0, 1]) #P0 if 0, P1 if 1

    n = rng.randint(1, 6) #generate random value for the first player

    if starter == 0:    #if P0 started and is in the first round, then rigged coin in the second
        headsProbability = (4/7)
    else:
        headsProbability = (1/2)    #if P1 started and is in the first round, then fair coin for P0 in the second round

    m = 0
    for _ in range(2 * n):
         if rng.random() < headsProbability:
             m = m + 1
    
    if m > n:  #the winner is not the one that started(the one in the first round)
        winner = 1 - starter
    else:
        winner = starter

    return winner

rng = random.Random(17)
playerZero = 0
playerOne = 0
for _ in range(10000):
    if onePlay(rng) == 0:
        playerZero = playerZero + 1
    else:
        playerOne = playerOne + 1

if playerZero > playerOne:
    print(f"After the trial P0 had the most wins: {playerZero} out of 10000")
else:
    print(f"After the trial P1 had the most wins: {playerOne} out of 10000")


#b
luckers_game = DiscreteBayesianNetwork([('S', 'M'), ('N', 'M')])

cpd_S = TabularCPD(
    variable='S',
    variable_card = 2,
    values=[[0.5], [0.5]], #even chance of picking the starter player
    state_names={'S': [0, 1]}  #P0 if 0, P1 if 1
)

cpd_N = TabularCPD(
    variable='N',
    variable_card = 6,
    values=[[1/6], [1/6], [1/6], [1/6], [1/6], [1/6]], #chance of getting a specific number after throwing the dice
    state_names={'N': [1, 2, 3, 4, 5, 6]}
)

M_states = list(range(13)) #[0, 1, 2 ... 12] - cate heads pot pica dupa 2 * aruncari
S_states = [0, 1]
N_states = [1, 2, 3, 4, 5, 6]

def probK(k, n, p):
    if k < 0 or k > n: 
        return 0.0
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) #calc probabilitate de a obtine k heads din n incerccari

values_M = []
for _ in M_states:
    values_M.append([])
for s in S_states:
    if s == 0:
        p = 4/7 
    else:
        p = 1/2
    
    for n in N_states:
        col = []
        for m in M_states:
            if m <= 2*n:
                col.append(probK(m, 2*n, p)) #probabilitatea de a obtine m heads in 2 * n aruncari
            else:
                col.append(0.0)  #normal ca e 0 probabilitatea de a extrage un numar de heads mai mare decat numarul de aruncari totale
        r = 0
        while r < len(M_states):
            values_M[r].append(col[r]) #values_M[0] o sa avem probabilitatile sa obtinem 0 heads pentru n = 1...6 (x2 ca avem si 2 valori p)
            r = r + 1

cpd_M = TabularCPD(
    variable='M',
    variable_card=len(M_states),
    values=values_M,
    evidence=['S', 'N'],
    evidence_card=[len(S_states), len(N_states)],
    state_names={'M': M_states, 'S': S_states, 'N': N_states}
)

luckers_game.add_cpds(cpd_S, cpd_N, cpd_M)
assert luckers_game.check_model(), "Model not valid"
print("[BsyanN] Model built. Nodes:", luckers_game.nodes(), "Edges:", luckers_game.edges())


#c
infer = VariableElimination(luckers_game)
posterior = infer.query(variables=['S'], evidence={'M': 1})
s0, s1 = posterior.values[0], posterior.values[1]

print(s0, s1)

likelyStarter = "P1" if s1 > s0 else "P0"
print(f"=> Most likely starter given M=1: {likelyStarter}")