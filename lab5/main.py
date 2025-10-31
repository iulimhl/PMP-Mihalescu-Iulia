from hmmlearn import hmm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
#A
model=hmm.CategoricalHMM(n_components=3,init_params="",params="")

model.startprob_=np.array([1/3,1/3,1/3])
model.transmat_=np.array([[0.0,0.5,0.5],[0.5,0.25,0.25],[0.5,0.25,0.25]])
model.emissionmat_=np.array([[0.1,0.2,0.4,0.3],[0.15,0.25,0.5,0.1],[0.2,0.3,0.4,0.1]])

states=["Difficult","Medium","Easy"]
observations=["FB","B","S","NS"]

G=nx.DiGraph()
for i,s1 in enumerate(states):
    for j,s2 in enumerate(states):
        if model.transmat_[i,j]>0:
            G.add_edge(s1,s2,weight=model.transmat_[i,j])
pos=nx.circular_layout(G)
nx.draw(G,pos,with_labels=True,node_size=2500)
nx.draw_networkx_edge_labels(G,pos,edge_labels={(i,j):d['weight'] for i,j,d in G.edges(data=True)})
plt.title("HMM")
plt.show()

model.n_features=4
X=np.array([[0]]).reshape(-1,1)
model.fit(X,lengths=[1])

#B
obs_seq=np.array([[0,0,2,1,1,2,1,1,3,1,1]]).T
logprob=model.score(obs_seq)
print("Probabilitatea log a observatiei ",logprob)
print("Probabilitatea reala ",np.exp(logprob))

#C
logprob,hidden_states=model.decode(obs_seq,algorithm="viterbi")
print("Secv cea mai probabila a dificultatilor ")
print([states[i] for i in hidden_states])
print("Probabilitatea logaritmica ",logprob)
print("Probabilitatea reala ",np.exp(logprob))

