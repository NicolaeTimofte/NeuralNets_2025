from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import itertools, pandas as pd

# ex: S inluenteaza pe O, S influenteaza pe L(avem P(L=1 | S=1) in enunt), etcc
class_emails = DiscreteBayesianNetwork([('S','O'), ('S','L'), ('S','M'), ('L','M')])

cpd_S = TabularCPD('S', 2, [[0.6],[0.4]], state_names={'S':[0,1]})
#print(cpd_S)

# O - valoarea pentru care facem tabelul, 2 numarul de stari ale lui O(0, 1)
# probabilitatile efective (pe randuri valorile diferite pe care le poate lua O)
# evidence=["S"] - inseamna ca O este conditionat de S
# evidence_card[2] - cate valori poate lua S
cpd_O = TabularCPD('O', 2,
                   [[0.9, 0.3], 
                    [0.1, 0.7]],
                   evidence=['S'], evidence_card=[2],
                   state_names={'O':[0,1], 'S':[0,1]})
#print(cpd_O)

cpd_L = TabularCPD('L', 2,
                   [[0.7, 0.2],
                    [0.3, 0.8]],
                   evidence=['S'], evidence_card=[2],
                   state_names={'L':[0,1], 'S':[0,1]})
#print(cpd_L)

cpd_M = TabularCPD('M', 2,
                   [[0.8, 0.4, 0.5, 0.1],
                    [0.2, 0.6, 0.5, 0.9]],
                   evidence=['S','L'], evidence_card=[2,2],
                   state_names={'M':[0,1], 'S':[0,1], 'L':[0,1]})
#print(cpd_M)

class_emails.add_cpds(cpd_S, cpd_O, cpd_L, cpd_M)
class_emails.check_model() #verif daca e viabil

print(class_emails.get_independencies())

infer = VariableElimination(class_emails) # calc niave bayes ignorand variabilele ne-necesare
rows = []
for o,l,m in itertools.product([0,1],[0,1],[0,1]):  # se vor genera toate combinatiile O, L, M
    q = infer.query(['S'], evidence={'O':o,'L':l,'M':m})
    p_spam = q.values[1]
    rows.append((o,l,m,p_spam, int(p_spam>=0.5)))
df = pd.DataFrame(rows, columns=['O','L','M','P(Spam|O,L,M)','Pred'])
print(df.to_string(index=False))

#df.to_csv('results.csv', index=False)