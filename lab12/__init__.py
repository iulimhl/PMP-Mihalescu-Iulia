#a
import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("date_promovare_examen.csv")
print(df["Promovare"].value_counts())
print(df["Promovare"].value_counts(normalize=True))

X=df[["Ore_Studiu","Ore_Somn"]].values
y=df["Promovare"].values.astype(int)

scaler=StandardScaler()
Xsc=scaler.fit_transform(X)

with pm.Model() as m:
    alpha=pm.Normal("alpha",0,1)
    beta=pm.Normal("beta",0,1,shape=2)
    logits=alpha+pm.math.dot(Xsc,beta)
    p=pm.Deterministic("p",pm.math.sigmoid(logits))
    yobs=pm.Bernoulli("yobs",p=p,observed=y)
    trace=pm.sample(2000,tune=1000,target_accept=0.9,chains=4)

#b
alpha_mean=float(trace.posterior["alpha"].mean().values)
beta_mean=trace.posterior["beta"].mean(dim=("chain","draw")).values
b1=float(beta_mean[0])
b2=float(beta_mean[1])

mu1,mu2=scaler.mean_[0],scaler.mean_[1]
s1,s2=scaler.scale_[0],scaler.scale_[1]

x1=np.linspace(df["Ore_Studiu"].min(),df["Ore_Studiu"].max(),200)
x1s=(x1-mu1)/s1
x2s=-(alpha_mean+b1*x1s)/b2
x2=x2s*s2+mu2

plt.scatter(df["Ore_Studiu"],df["Ore_Somn"],c=df["Promovare"])
plt.plot(x1,x2)
plt.show()

az.plot_trace(trace,var_names=["alpha","beta"])
plt.show()

#c
abs_b1=abs(b1)
abs_b2=abs(b2)
print(abs_b1,abs_b2)
