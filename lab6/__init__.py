import numpy as np
import arviz as az
import scipy.stats as st

a=1; b=0
n=10; k=180
ap=a+k
bp=b+n

mode=(ap-1)/bp

samples=st.gamma(a=ap,scale=1/bp).rvs(size=200000)
hdi=az.hdi(samples,hdi_prob=0.94)

print("Posterior: Gamma(",ap,",",bp,")")
print("Mode:",mode)
print("94% HDI:",hdi)
