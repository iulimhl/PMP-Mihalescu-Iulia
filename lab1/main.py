# Exercitiul 1- bile
import numpy as np

n=1000
r_extrase=0

for i in range(n):
    r=3
    a=4
    k=2
    zar=np.random.choice([1,2,3,4,5,6])

    if zar in[2,3,5]:
        k+=1
    elif zar==6:
        r+=1
    else:
        a+=1

    total=r+a+k
    x=np.random.uniform(0,total)
    if x<r:
        r_extrase+=1
#punctul b
p_rosie=r_extrase/n

#punctul c
p_teoretica=(3/6)*(3/10)+(1/6)*(4/10)+(2/6)*(3/10)
print("Ex. 1")
print("Probabilitatea alegerii unei bile rosii= ",round(p_rosie,4))
print("Probabilitate teoretica= ",round(p_teoretica,4))
print()

#Exercitiul 2
#2.1
import numpy as np
import matplotlib.pyplot as plt

poisson1=np.random.poisson(lam=1,size=1000)
poisson2=np.random.poisson(lam=2,size=1000)
poisson5=np.random.poisson(lam=5,size=1000)
poisson10=np.random.poisson(lam=10,size=1000)

#2.2
l=[1,2,5,10]
l_ales=np.random.choice(l, size=1000)
poisson_random=np.array([np.random.poisson(lam) for lam in l_ales])

#2.2.a
plt.hist(poisson1, bins=10, color='green', edgecolor='black', alpha=0.7)
plt.title('Distributia Poisson (l=1)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()

plt.hist(poisson2, bins=10, color='green', edgecolor='black', alpha=0.7)
plt.title('Distributia Poisson (l=2)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()

plt.hist(poisson5, bins=10, color='green', edgecolor='black', alpha=0.7)
plt.title('Distributia Poisson (l=5)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()

plt.hist(poisson10, bins=10, color='green', edgecolor='black', alpha=0.7)
plt.title('Distributia Poisson (l=10)')
plt.xlabel('Valori')
plt.ylabel('Frecvență')
plt.show()

#2.2.b
#a) Cand lambda este mic, valorile sunt mai aproape de 0 iar cand lambda este mai mare, graficul se intinde si devine mai simetric
# Distributia randomizata este mai larga si mai neregulata pentru ca am combinat mai multe valori ale lui lambda
#b) Asta arata ca atunci cand un parametru nu este constant, rezultatele vor fi mai raspandite si mai greu de prezis