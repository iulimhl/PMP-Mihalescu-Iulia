import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from math import comb

ys=[0,5,10]
thetas=[0.2,0.5]
max_n=60
ns=np.arange(max_n+1)

poisson=np.zeros(max_n+1)
poisson[0]=np.exp(-10.0)
for n in range(max_n):
    poisson[n+1]=poisson[n]*10.0/(n+1)

posteriors={}
predictive_samples={}

for theta in thetas:
    for y in ys:
        print(f"\nScenario: Y={y}, theta={theta}")
        lik=np.zeros(max_n+1)
        for n in range(y,max_n+1):
            lik[n]=comb(n,y)*(theta**y)*((1-theta)**(n-y))
        post=lik*poisson
        post=post/post.sum()
        n_samples=np.random.choice(ns,size=5000,p=post)
        idata=az.from_dict(posterior={"n":n_samples})
        posteriors[(theta,y)]=idata
        mean_n=(ns*post).sum()
        print(f"media n ~ {mean_n:.2f}")
        max_y_star=max_n
        y_star_vals=np.arange(max_y_star+1)
        pred_p=np.zeros(max_y_star+1)
        for n in range(max_n+1):
            if post[n]==0:continue
            for y_star in range(0,n+1):
                pred_p[y_star]+=post[n]*comb(n,y_star)*(theta**y_star)*((1-theta)**(n-y_star))
        pred_p=pred_p/pred_p.sum()
        y_star_samples=np.random.choice(y_star_vals,size=5000,p=pred_p)
        predictive_samples[(theta,y)]=y_star_samples

fig,axes=plt.subplots(len(ys),len(thetas),figsize=(8,8),sharex=True,sharey=True)
for i,y in enumerate(ys):
    for j,theta in enumerate(thetas):
        ax=axes[i,j]
        az.plot_posterior(posteriors[(theta,y)],var_names=["n"],ax=ax,hdi_prob=0.94)
        ax.set_title(f"n | Y={y}, θ={theta}")
plt.tight_layout()
plt.show()

fig,axes=plt.subplots(len(ys),len(thetas),figsize=(8,8),sharex=True,sharey=True)
for i,y in enumerate(ys):
    for j,theta in enumerate(thetas):
        ax=axes[i,j]
        az.plot_dist(predictive_samples[(theta,y)],ax=ax)
        ax.set_title(f"Y* | Y={y}, θ={theta}")
plt.tight_layout()
plt.show()
