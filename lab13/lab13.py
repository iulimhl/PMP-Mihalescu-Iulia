import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')
data=np.loadtxt('./date.csv')
x_1=data[:,0]
y_1=data[:,1]
y_1s=(y_1-y_1.mean())/y_1.std()

#1.a
order=5
x_1p=np.vstack([x_1**i for i in range(1,order+1)])
x_1s=(x_1p-x_1p.mean(axis=1,keepdims=True))/x_1p.std(axis=1,keepdims=True)
with pm.Model() as model_p:
    alpha=pm.Normal('alpha',mu=0,sigma=1)
    beta=pm.Normal('beta',mu=0,sigma=10,shape=order)
    eps=pm.HalfNormal('eps',5)
    mu=alpha+pm.math.dot(beta,x_1s)
    y_pred=pm.Normal('y_pred',mu=mu,sigma=eps,observed=y_1s)
    idata_p=pm.sample(2000,return_inferencedata=True)
alpha_post=idata_p.posterior['alpha'].mean(("chain","draw")).values
beta_post=idata_p.posterior['beta'].mean(("chain","draw")).values
idx=np.argsort(x_1s[0])
y_post=alpha_post+np.dot(beta_post,x_1s)
plt.plot(x_1s[0][idx],y_post[idx],label=f'model order {order},sd=10')
plt.scatter(x_1s[0],y_1s,marker='.')
plt.legend()
plt.show()

#1.b
with pm.Model() as model_p_100:
    alpha=pm.Normal('alpha',mu=0,sigma=1)
    beta=pm.Normal('beta',mu=0,sigma=100,shape=order)
    eps=pm.HalfNormal('eps',5)
    mu=alpha+pm.math.dot(beta,x_1s)
    y_pred=pm.Normal('y_pred',mu=mu,sigma=eps,observed=y_1s)
    idata_p_100=pm.sample(2000,return_inferencedata=True)
alpha_post=idata_p_100.posterior['alpha'].mean(("chain","draw")).values
beta_post=idata_p_100.posterior['beta'].mean(("chain","draw")).values
idx=np.argsort(x_1s[0])
y_post=alpha_post+np.dot(beta_post,x_1s)
plt.plot(x_1s[0][idx],y_post[idx],label='order5,sd=100')
plt.scatter(x_1s[0],y_1s,marker='.')
plt.legend()
plt.show()

sd_vec=np.array([10,0.1,0.1,0.1,0.1])
with pm.Model() as model_p_vec:
    alpha=pm.Normal('alpha',mu=0,sigma=1)
    beta=pm.Normal('beta',mu=0,sigma=sd_vec,shape=order)
    eps=pm.HalfNormal('eps',5)
    mu=alpha+pm.math.dot(beta,x_1s)
    y_pred=pm.Normal('y_pred',mu=mu,sigma=eps,observed=y_1s)
    idata_p_vec=pm.sample(2000,return_inferencedata=True)
alpha_post=idata_p_vec.posterior['alpha'].mean(("chain","draw")).values
beta_post=idata_p_vec.posterior['beta'].mean(("chain","draw")).values
idx=np.argsort(x_1s[0])
y_post=alpha_post+np.dot(beta_post,x_1s)
plt.plot(x_1s[0][idx],y_post[idx],label='order5,sd=[10,0.1,0.1,0.1,0.1]')
plt.scatter(x_1s[0],y_1s,marker='.')
plt.legend()
plt.show()

#2
rng=np.random.default_rng(0)
idx500=rng.integers(0,len(x_1),size=500)
x_2=x_1[idx500]
y_2=y_1[idx500]
y_2s=(y_2-y_2.mean())/y_2.std()
order=5
x_2p=np.vstack([x_2**i for i in range(1,order+1)])
x_2s=(x_2p-x_2p.mean(axis=1,keepdims=True))/x_2p.std(axis=1,keepdims=True)
with pm.Model() as model_p_500:
    alpha=pm.Normal('alpha',mu=0,sigma=1)
    beta=pm.Normal('beta',mu=0,sigma=10,shape=order)
    eps=pm.HalfNormal('eps',5)
    mu=alpha+pm.math.dot(beta,x_2s)
    y_pred=pm.Normal('y_pred',mu=mu,sigma=eps,observed=y_2s)
    idata_p_500=pm.sample(2000,return_inferencedata=True)
alpha_post=idata_p_500.posterior['alpha'].mean(("chain","draw")).values
beta_post=idata_p_500.posterior['beta'].mean(("chain","draw")).values
idx=np.argsort(x_2s[0])
y_post=alpha_post+np.dot(beta_post,x_2s)
plt.plot(x_2s[0][idx],y_post[idx],label='order5,n=500,sd=10')
plt.scatter(x_2s[0],y_2s,marker='.',alpha=0.4)
plt.legend()
plt.show()

#3
def fit_poly(x,y_s,order,beta_sd=10):
    x_p=np.vstack([x**i for i in range(1,order+1)])
    x_s=(x_p-x_p.mean(axis=1,keepdims=True))/x_p.std(axis=1,keepdims=True)
    with pm.Model() as m:
        alpha=pm.Normal('alpha',mu=0,sigma=1)
        if order==1:
            beta=pm.Normal('beta',mu=0,sigma=beta_sd)
            mu=alpha+beta*x_s[0]
        else:
            beta=pm.Normal('beta',mu=0,sigma=beta_sd,shape=order)
            mu=alpha+pm.math.dot(beta,x_s)
        eps=pm.HalfNormal('eps',5)
        y_pred=pm.Normal('y_pred',mu=mu,sigma=eps,observed=y_s)
        idata=pm.sample(2000,return_inferencedata=True)
        pm.compute_log_likelihood(idata,model=m)
    return x_s,idata

x1s,idata_l=fit_poly(x_1,y_1s,1,10)
x2s,idata_q=fit_poly(x_1,y_1s,2,10)
x3s,idata_c=fit_poly(x_1,y_1s,3,10)

def plot_mean(order,x_s,idata,label):
    x_new=np.linspace(x_s[0].min(),x_s[0].max(),100)
    alpha_post=idata.posterior['alpha'].mean(("chain","draw")).values
    if order==1:
        beta_post=idata.posterior['beta'].mean(("chain","draw")).values
        y_post=alpha_post+beta_post*x_new
        plt.plot(x_new,y_post,label=label)
    else:
        beta_post=idata.posterior['beta'].mean(("chain","draw")).values
        idx=np.argsort(x_s[0])
        y_post=alpha_post+np.dot(beta_post,x_s)
        plt.plot(x_s[0][idx],y_post[idx],label=label)

plt.figure(figsize=(7,4))
plot_mean(1,x1s,idata_l,'linear')
plot_mean(2,x2s,idata_q,'quadratic')
plot_mean(3,x3s,idata_c,'cubic')
plt.scatter(x1s[0],y_1s,marker='.')
plt.legend()
plt.show()

waic_l=az.waic(idata_l,scale="deviance")
waic_q=az.waic(idata_q,scale="deviance")
waic_c=az.waic(idata_c,scale="deviance")
loo_l=az.loo(idata_l,scale="deviance")
loo_q=az.loo(idata_q,scale="deviance")
loo_c=az.loo(idata_c,scale="deviance")
print("WAIC linear:",waic_l)
print("WAIC quadratic:",waic_q)
print("WAIC cubic:",waic_c)
print("LOO linear:",loo_l)
print("LOO quadratic:",loo_q)
print("LOO cubic:",loo_c)

cmp_waic=az.compare({'linear':idata_l,'quadratic':idata_q,'cubic':idata_c},ic="waic",scale="deviance",method="BB-pseudo-BMA")
cmp_loo=az.compare({'linear':idata_l,'quadratic':idata_q,'cubic':idata_c},ic="loo",scale="deviance",method="BB-pseudo-BMA")
print("\nCOMPARE WAIC:\n",cmp_waic)
print("\nCOMPARE LOO:\n",cmp_loo)
