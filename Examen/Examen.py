import pytensor
pytensor.config.cxx=""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pymc as pm
import arviz as az
import numpy as np



#1
df=pd.read_csv("bike_daily.csv")

print(df.head())
print(df.describe())

plt.scatter(df["temp_c"],df["rentals"])
plt.xlabel("temp_c")
plt.ylabel("rentals")
plt.title("rentalsvs temp_c")
plt.show()

plt.scatter(df["humidity"],df["rentals"])
plt.xlabel("humidity")
plt.ylabel("rentals")
plt.title("rentalsvs humidity")
plt.show()

plt.scatter(df["wind_kph"],df["rentals"])
plt.xlabel("wind_kph")
plt.ylabel("rentals")
plt.title("rentalsvs wind_kph")
plt.show()

#2a
y=df["rentals"].values
Xcont=df[["temp_c","humidity","wind_kph"]].values
scaler=StandardScaler()
Xcont=scaler.fit_transform(Xcont)

temp=Xcont[:,0]
humidity=Xcont[:,1]
wind=Xcont[:,2]

holiday=df["is_holiday"].values

season_dum=pd.get_dummies(df["season"],drop_first=True)
season_mat=season_dum.values
season_cols=list(season_dum.columns)

print("season_cols=",season_cols)

#2b
with pm.Model() as linear_model:
    alpha = pm.Normal("alpha", mu=df["rentals"].mean(), sigma=100)

    beta_temp=pm.Normal("beta_temp",0,2)
    beta_hum=pm.Normal("beta_hum",0,2)
    beta_wind=pm.Normal("beta_wind",0,2)
    beta_holiday=pm.Normal("beta_holiday",0,2)

    beta_season=pm.Normal("beta_season",0,5,shape=season_mat.shape[1])

    sigma=pm.HalfNormal("sigma",10)

    mu=alpha+beta_temp*temp+beta_hum*humidity+beta_wind*wind+beta_holiday*holiday+pm.math.dot(season_mat,beta_season)

    rentals_obs=pm.Normal("rentals_obs",mu,sigma,observed=y)

    trace_linear=pm.sample(2000,tune=1000,target_accept=0.9,chains=1,cores=1)

print(az.summary(trace_linear,round_to=3))

#2c
temp2=temp**2

with pm.Model() as poly_model:
    alpha=pm.Normal("alpha",mu=df["rentals"].mean(),sigma=100)
    beta_temp=pm.Normal("beta_temp",0,2)
    beta_temp2=pm.Normal("beta_temp2",0,2)
    beta_hum=pm.Normal("beta_hum",0,2)
    beta_wind=pm.Normal("beta_wind",0,2)
    sigma=pm.HalfNormal("sigma",50)

    mu=alpha+beta_temp*temp+beta_temp2*temp2+beta_hum*humidity+beta_wind*wind

    rentals_obs=pm.Normal("rentals_obs",mu,sigma,observed=y)

    trace_poly=pm.sample(2000,tune=1000,target_accept=0.9,chains=1,cores=1)

az.summary(trace_poly)

#3
sum_lin=az.summary(trace_linear,round_to=3)
sum_poly=az.summary(trace_poly,round_to=3)

print("LINEARMODEL")
print(sum_lin)

print("\nPOLYNOMIALMODEL")
print(sum_poly)

#4a

waic_linear=az.waic(trace_linear)
waic_poly=az.waic(trace_poly)

print("WAIC_LINEAR")
print(waic_linear)

print("\nWAIC_POLYNOMIAL")
print(waic_poly)

#4b

with linear_model:
    ppc_linear=pm.sample_posterior_predictive(trace_linear)

with poly_model:
    ppc_poly=pm.sample_posterior_predictive(trace_poly)

az.plot_ppc(az.from_pymc(posterior_predictive=ppc_linear,model=linear_model))
plt.title("PPC_linear")
plt.show()

az.plot_ppc(az.from_pymc(posterior_predictive=ppc_poly,model=poly_model))
plt.title("PPC_polynomial")
plt.show()


temp_grid=np.linspace(df["temp_c"].min(),df["temp_c"].max(),80)
temp_std=(temp_grid-scaler.mean_[0])/scaler.scale_[0]
temp2_std=temp_std**2

post=trace_poly.posterior
a=post["alpha"].values.flatten()
b1=post["beta_temp"].values.flatten()
b2=post["beta_temp2"].values.flatten()
bh=post["beta_holiday"].values.flatten()
bw=post["beta_wind"].values.flatten()
bhum=post["beta_hum"].values.flatten()
bs=post["beta_season"].values.reshape(-1,season_mat.shape[1])

holiday0=0.0
hum0=0.0
wind0=0.0
season0=np.zeros(season_mat.shape[1])

mu=[]
for i in range(len(temp_std)):
    mu_i=a+b1*temp_std[i]+b2*temp2_std[i]+bhum*hum0+bw*wind0+bh*holiday0+np.dot(bs,season0)
    mu.append(mu_i)
mu=np.array(mu)

plt.plot(temp_grid,mu.mean(axis=1))
plt.fill_between(temp_grid,np.percentile(mu,2.5,axis=1),np.percentile(mu,97.5,axis=1),alpha=0.3)
plt.xlabel("temp_c")
plt.ylabel("predictedrentals")
plt.title("predictedmeananduncertaintyvstemp_c")
plt.show()


#5
Q=np.percentile(df["rentals"],75)
df["is_high_demand"]=(df["rentals"]>=Q).astype(int)
y_bin=df["is_high_demand"].values


print("Q=",Q)
print(df["is_high_demand"].value_counts())

#6
with pm.Model() as logistic_model:
    alpha=pm.Normal("alpha",0,10)
    beta_temp=pm.Normal("beta_temp",0,2)
    beta_temp2=pm.Normal("beta_temp2",0,2)
    beta_hum=pm.Normal("beta_hum",0,2)
    beta_wind=pm.Normal("beta_wind",0, 2)

    logit_p=alpha+beta_temp*temp+beta_temp2*temp2+beta_hum*humidity+beta_wind*wind

    y_obs=pm.Bernoulli("y_obs",logit_p=logit_p,observed=y_bin)

    trace_logistic=pm.sample(2000,tune=1000,target_accept=0.9,chains=1,cores=1)

#7
az.plot_posterior(trace_logistic, var_names=["beta_temp", "beta_hum", "beta_wind"], hdi_prob=0.95)
print(az.summary(trace_logistic, hdi_prob=0.95))


