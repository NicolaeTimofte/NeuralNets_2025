import random

#punctul a
def oneSim():
    urn = ["R"]*3 + ["B"]*4 + ["K"]*2

    #roll the dice
    d = random.randint(1, 6)

    #insert a new ball depending on the number chosen
    if d in {2, 3, 5}:
        urn.append("K")
    elif d == 6:
        urn.append("R")
    else:
        urn.append("B")

    #draw one ball
    return random.choice(urn) == "R"

#punctul b
def simulate(n: int) -> float:
    sum = 0
    for i in range(n) :
        if oneSim() == True :
            sum = sum + 1
    return sum / float(n)

aux = simulate(1000000)
print("Probabilitatea estimata:", end = " ")
print(aux)

#punctul c

#calc probabilitatea reala

#probabilitatea de a introduce o bila rosie - doar cand dice e 6
probInsertRosie = 1 / 6
#probabilitatea de a introduce o bila albastra - cand dice e 1,4
probInsertAlbastra = 1 / 3
#probabilitatea de a introduce o bila neagra - cand dice e 2, 3, 5
probInsertNeagra = 1 / 2

#calc probabilitatea generala de a extrage o bila rosie
probRosieCondRosie = (2 / 5) * probInsertRosie
probRosieCondAlbastra = (3 / 10) * probInsertAlbastra
probRosieCondNeagra = (3 / 10) * probInsertNeagra

totalProb = probRosieCondRosie + probRosieCondAlbastra + probRosieCondNeagra
print("Probabilitatea teoretica: ", totalProb)

print("Diferenta dintre estimata si teoretica: ", aux - totalProb)