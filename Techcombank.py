# %%
from vnstock.api.quote import Quote
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.ar_model import ar_select_order as criteria
from statsmodels.stats.diagnostic import het_arch as arch_effect
import math
from scipy.optimize import minimize
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
from scipy.stats import norm

# Download data
tcb = Quote(symbol="TCB", source="vci")
dataframe = tcb.history(start="2019-01-01", end="2026-08-20", interval='day')

# %%
# Calculate returns
#print(dataframe.head())
dataframe['returns'] = np.log(dataframe['close']/dataframe['close'].shift(1))
#print(dataframe['returns'].head())
y = dataframe['returns']*100


'''
# plot closed price and returns
fig, ax1 = plt.subplots()
ax1.plot(dataframe['time'] ,y, color= 'blue')
ax1.set_ylabel("Return (%)", color= 'blue')
ax1.set_ylim(y.min()*1.10, y.max()*1.10)

ax2 = ax1.twinx()
ax2.plot(dataframe['time'], dataframe['close'], color= 'red')
ax2.set_ylabel("Price", color = 'red')
ax2.set_ylim(0,dataframe['close'].max()*1.1)
plt.show()'''

# %%
# choose lag order using BIC
y = y.dropna()
lags = criteria(y, maxlag=20, ic="bic").ar_lags
print(lags)

# regress AR(1) 
AR1 = AR(y, lags=lags)
results = AR1.fit()
print(results.summary())
mu = results.params['const']


# %%
# check if residual exist arch effect
u = results.resid
lm_stat, lm_value, F_stat, F_value = arch_effect(u)
print(" F-value = ", f"{F_value:.2e}" , "\n", "LM-value = ", f"{lm_value:.2e}")
u = u.values

# %%
params = [0,0.01,0.02]
arch_lags = 1
garch_lags = 1

# %%
# create function calculating variance equation
def variance_equation(params, arch_lags, garch_lags, residuals):
    constant, arch_coeficients, garch_coeficients = params[0:1], params[1:arch_lags+1], params[arch_lags+1:arch_lags+1+garch_lags]
    p = len(arch_coeficients)
    q = len(garch_coeficients)
    n = len(residuals)
    arch_term = 0
    garch_term = 0
    sigma_sq = np.zeros(n)
   
    for i in range (max(p,q)):
        sigma_sq[i] = np.var(residuals)

    for t in range (max(p,q), n):
        arch_term = sum(arch_coeficients[i]*residuals[t-1-i]**2 for i in range (p))
        garch_term = sum(garch_coeficients[j]*sigma_sq[t-1-j] for j in range (q))
        sigma_sq[t] = constant + arch_term + garch_term
    return sigma_sq

sigma_sq = variance_equation(params=params,arch_lags=arch_lags, garch_lags=garch_lags, residuals=u)
print(sigma_sq[:10])


# %% calculate log likelihood 
def log_likelihood (params, arch_lags, garch_lags, residuals):
    sigma_sq = variance_equation(params, arch_lags, garch_lags, residuals)
    log_likelihood = sum(-0.5*np.log(2*math.pi) - 0.5*np.log(sigma_sq[i]) - residuals[i]**2/(2*sigma_sq[i]) for i in range (len(residuals)))
    return -log_likelihood

print("log_likelihood = ", log_likelihood(params, arch_lags=arch_lags, garch_lags=garch_lags, residuals=u))

MLE = minimize (log_likelihood,x0=params, args=(arch_lags, garch_lags, u), bounds=[(0, None)]*len(params))
print(MLE.x)

test = arch_model(u,mean="Zero", vol="GARCH",p=1,q=1,dist="normal").fit()
print(test.summary())
print(test.params)

# %%
# diagnosting if residuals of variance equation exist arch effect
final_sigma_sq = variance_equation(MLE.x, arch_lags, garch_lags, u)
standardized_res = u/np.sqrt(final_sigma_sq)
print(acorr_ljungbox(standardized_res**2,lags=[10],return_df=True))
MA20 = pd.Series(dataframe["close"]).rolling(window=20).mean()

# %%
# calculate VaR
sigma_t = np.sqrt(final_sigma_sq)
z_alpha = norm.ppf(0.95)  # +1.64485
VaR_t = z_alpha * sigma_t - mu
print("var= ", np.average(VaR_t))

# plotting
fig, ax1 = plt.subplots()
ax1.plot(dataframe['time'][2:], np.sqrt(final_sigma_sq), color='steelblue', linewidth=0.8, alpha=0.8, label="Volatility")
ax1.plot(dataframe["time"][2:], VaR_t, color='steelblue')
ax1.set_ylabel("Volatility (σ)", color='steelblue', fontsize=11)
ax1.set_ylim(0, final_sigma_sq.max()*1.1)
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(dataframe['time'][2:],dataframe["close"][2:], color='firebrick', linewidth=0.8, alpha=0.8, label="Closed price")
#ax2.plot(dataframe["time"][2:], MA20[2:], color='navy', linewidth=0.8, alpha=0.8, label="MA20")
ax2.set_ylabel("Techcombank closed price (VND)", color='firebrick', fontsize=11)
ax2.set_ylim(0, dataframe["close"].max()*1.1)
ax2.tick_params(axis='y', labelcolor='firebrick')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

ax1.set_title("TCB — Estimated GARCH Volatility vs Closing Price", fontsize=13)
ax1.grid(alpha=0.2)
fig.tight_layout()
plt.show()

print(len(MA20))
print(len(dataframe["close"]))
print(MA20.isna().sum())
