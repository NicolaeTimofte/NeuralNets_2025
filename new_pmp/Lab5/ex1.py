import numpy as np
import math
from hmmlearn.hmm import CategoricalHMM

model = CategoricalHMM(n_components=3, init_params="", n_features=4)
model.startprob_ = np.array([1/3, 1/3, 1/3])
model.transmat_  = np.array([
    [0.0 , 0.5 , 0.5 ],
    [0.5 , 0.25, 0.25],
    [0.5 , 0.25, 0.25],
])
model.emissionprob_ = np.array([
    [0.10, 0.20, 0.40, 0.30],
    [0.15, 0.25, 0.50, 0.10],
    [0.20, 0.30, 0.40, 0.10],
])

grades = ["FB","FB","S","B","B","S","B","B","NS","B","B"]
obs_map = {"FB":0,"B":1,"S":2,"NS":3}
obs = np.array([[obs_map[x]] for x in grades], dtype = int)

#b
log_lik = model.score(obs)

#c
v_logprob, states = model.decode(obs, algorithm = "viterbi")
print(log_lik)
print(v_logprob, states)