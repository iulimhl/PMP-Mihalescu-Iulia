#1

from pgmpy.models import DiscreteBayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
'''
model=DiscreteBayesianNetwork([('S','O'),('S','L'),('S','M'),('L','M')])

cpd_S=TabularCPD(variable='S',variable_card=2,values=[[0.6],[0.4]])

cpd_O=TabularCPD(variable='O',variable_card=2,values=[[0.9,0.3],[0.1,0.7]],evidence=['S'],evidence_card=[2])

cpd_L=TabularCPD(variable='L',variable_card=2,values=[[0.7,0.2],[0.3,0.8]],evidence=['S'],evidence_card=[2])

cpd_M=TabularCPD(variable='M',variable_card=2,values=[[0.8,0.4,0.5,0.1],[0.2,0.6,0.5,0.9]],evidence=['S','L'],evidence_card=[2,2])

model.add_cpds(cpd_S,cpd_O,cpd_L,cpd_M)
assert model.check_model()

#a
print("Independente:")
print(model.local_independencies(['S','O','L','M']))


#b
infer=VariableElimination(model)

def classify(o,l,m):
    q=infer.query(['S'],evidence={'O':o,'L':l,'M': m},show_progress=False)
    p_spam=float(q.values[1])  # S=1
    label='spam' if p_spam>0.5 else 'normal'
    return p_spam,label

print("\nO L M  P(S=1|O,L,M)  Clasa")
for o in [0,1]:
    for l in [0,1]:
        for m in [0,1]:
            p,lab=classify(o,l,m)
            print(o,l,m,f"{p:.4f}".rjust(12),lab.rjust(6))

#desen graf
pos=nx.circular_layout(model)
nx.draw(model,pos,with_labels=True,node_size=4000,font_weight='bold',node_color='skyblue')
plt.show()

#2
model=DiscreteBayesianNetwork([('D','A'),('A','C')])

cpd_D=TabularCPD(variable='D',variable_card=6,values=[[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]])

cpd_A=TabularCPD(variable='A',variable_card=3,values=[[0,0,0,0,0,1],[1,0,0,1,0,0],[0,1,1,0,1,0]],evidence=['D'],evidence_card=[6])

cpd_C=TabularCPD(variable='C',variable_card=3,values=[[4/10,3/10,3/10],[4/10,5/10,4/10],[2/10,2/10,3/10]],evidence=['A'],evidence_card=[3])

model.add_cpds(cpd_D,cpd_A,cpd_C)
assert model.check_model()

infer=VariableElimination(model)
q=infer.query(variables=['C'])
p_red=float(q.values[0])
print("probab de a extrage o bila rosie:",round(p_red,4))

pos=nx.circular_layout(model)
nx.draw(model,pos,with_labels=True,node_size=4000,node_color='skyblue',font_weight='bold')
plt.show()

'''''
#3
import random
import math

#3.1
random.seed(0)

def game():
    start=random.choice(['P0','P1'])
    n=random.randint(1,6)
    p=4/7 if start=='P1' else 0.5
    m=sum(1 for _ in range(2*n) if random.random()<p)
    return 'P0' if (start=='P0' and n>=m) or (start=='P1' and n<m) else 'P1'

N=10000
w0=w1=0
for _ in range(N):
    w0+=game()=='P0'
    w1+=game()=='P1'
print('simulare(',N,'jocuri)->P0 win: ',round(w0/N,3),',P1 win: ',round(w1/N,3),sep='')

#3.2
model=DiscreteBayesianNetwork([('Start','M'),('N','M')])

cpd_start=TabularCPD('Start',2,[[0.5],[0.5]])
cpd_N=TabularCPD('N',6,[[1/6],[1/6],[1/6],[1/6],[1/6],[1/6]])

cols=[]
for s in ['P0','P1']:
    for n in [1,2,3,4,5,6]:
        p=4/7 if s=='P0' else 0.5
        t=2*n
        col=[math.comb(t,k)*(p**k)*((1-p)**(t-k)) for k in range(t+1)]
        if t<12:col+= [0]*(12-t)
        cols.append(col)

values=[[cols[j][i] for j in range(len(cols))] for i in range(13)]

cpd_M=TabularCPD('M',13,values,evidence=['Start','N'],evidence_card=[2,6])

model.add_cpds(cpd_start,cpd_N,cpd_M)
assert model.check_model()

#3.3
infer=VariableElimination(model)
post=infer.query(['Start'],evidence={'M':1},show_progress=False)
p0=float(post.values[0]);p1=float(post.values[1])
who='P1' if p1>p0 else 'P0'
print('post M=1 -> P(Start=P0|M=1)=',round(p0,3),', P(Start=P1|M=1)=',round(p1,3),' -> start probabil: ',who,sep='')

pos=nx.circular_layout(model)
nx.draw(model,pos,with_labels=True,node_size=3800,node_color='skyblue',font_weight='bold')
plt.show()
