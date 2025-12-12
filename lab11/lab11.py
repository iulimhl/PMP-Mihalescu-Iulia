import pymc as pm
import arviz as az
import numpy as np
import pandas as pd


df=pd.read_csv("Prices.csv")
y=df.Price.values
x1=df.Speed.values
x2=np.log(df.HardDrive.values)

# EX1.a
with pm.Model() as model:
    alpha=pm.Normal("alpha",mu=0,sigma=1000)
    beta1=pm.Normal("beta1",mu=0,sigma=100)
    beta2=pm.Normal("beta2",mu=0,sigma=100)
    sigma=pm.HalfNormal("sigma",sigma=1000)
    mu=alpha+beta1*x1+beta2*x2
    y_obs=pm.Normal("y_obs",mu=mu,sigma=sigma,observed=y)
    trace=pm.sample(4000,tune=2000,target_accept=0.9,return_inferencedata=True)

# EX1.b
hdi95=az.hdi(trace,hdi_prob=0.95)[["beta1","beta2"]]
print("HDI95 beta1,beta2")
print(hdi95)

# EX1.c
print("Interpretare: daca HDI nu include 0, predictorul este util.")

# EX1.d
x1_new=33
x2_new=np.log(540)
a=trace.posterior["alpha"].values.flatten()
b1=trace.posterior["beta1"].values.flatten()
b2=trace.posterior["beta2"].values.flatten()
mu_new=a+b1*x1_new+b2*x2_new
hdi_mu90=az.hdi(mu_new,hdi_prob=0.90)
print("HDI90 mu:",hdi_mu90)

# EX1.e
with model:
    pm.set_data({"Speed":np.array([x1_new]),"HardDrive":np.array([540])})
with pm.Model() as pred_model:
    alpha_p=pm.Normal("alpha",mu=trace.posterior["alpha"].mean(),sigma=trace.posterior["alpha"].std())
    beta1_p=pm.Normal("beta1",mu=trace.posterior["beta1"].mean(),sigma=trace.posterior["beta1"].std())
    beta2_p=pm.Normal("beta2",mu=trace.posterior["beta2"].mean(),sigma=trace.posterior["beta2"].std())
    sigma_p=pm.HalfNormal("sigma",sigma=trace.posterior["sigma"].mean())
    mu_p=alpha_p+beta1_p*x1_new+beta2_p*x2_new
    y_new=pm.Normal("y_new",mu=mu_p,sigma=sigma_p)
    pred=pm.sample(4000)

y_samples=pred.posterior["y_new"].values.flatten()
hdi_y90=az.hdi(y_samples,hdi_prob=0.90)
print("HDI90 predictie:",hdi_y90)

# BONUS
df["PremiumDummy"]=(df.Premium=="yes").astype(int)
y=df.Price.values
x1=df.Speed.values
x2=np.log(df.HardDrive.values)
x3=df.PremiumDummy.values

with pm.Model() as model_bonus:
    alpha_b=pm.Normal("alpha",mu=0,sigma=1000)
    beta1_b=pm.Normal("beta1",mu=0,sigma=100)
    beta2_b=pm.Normal("beta2",mu=0,sigma=100)
    beta3_b=pm.Normal("beta3",mu=0,sigma=100)
    sigma_b=pm.HalfNormal("sigma",sigma=1000)
    mu_b=alpha_b+beta1_b*x1+beta2_b*x2+beta3_b*x3
    y_obs_b=pm.Normal("y_obs",mu=mu_b,sigma=sigma_b,observed=y)
    trace_bonus=pm.sample(4000,tune=2000,target_accept=0.9,return_inferencedata=True)

hdi_bonus=az.hdi(trace_bonus,hdi_prob=0.95)[["beta3"]]
print("HDI95 beta3 premium:",hdi_bonus)
print("Interpretare: daca HDI beta3 e >0 atunci producatorii premium cresc pretul.")
