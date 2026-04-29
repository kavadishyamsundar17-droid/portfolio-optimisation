# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 16:27:21 2026

@author: kavad
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("Generating Mathematical Market Simulation...")
np.random.seed(42) 

# ==========================================
# GRAPH 1: THE FAT TAIL HISTOGRAM
# ==========================================
print("Drawing Graph 1: Return Distribution...")

# Simulating 1,250 days of stock returns with deliberate "Fat Tails" 
asset_returns = np.random.standard_t(df=3, size=1250) / 100 

plt.figure(figsize=(10, 6))

# Plot the histogram of our simulated market
plt.hist(asset_returns, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Simulated Asset Returns')

# Pure Numpy calculation for the Normal Distribution curve (No SciPy needed)
mu = np.mean(asset_returns)
std = np.std(asset_returns)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
# Manual Probability Density Function (PDF) formula
p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2)

plt.plot(x, p, 'k', linewidth=2.5, label='Perfect Normal Distribution')

plt.title('Return Distribution: Evidence of Extreme Market Tails', fontweight='bold', fontsize=14)
plt.xlabel('Daily Logarithmic Returns', fontsize=12)
plt.ylabel('Frequency Density', fontsize=12)
plt.axvline(0, color='red', linestyle='dashed', linewidth=1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# GRAPH 2: THE CVaR EFFICIENT FRONTIER
# ==========================================
print("Drawing Graph 2: CVaR Efficient Frontier...")

num_portfolios = 3000
num_assets = 4

# Simulating expected returns and a covariance matrix for 4 assets
mean_returns = np.array([0.06, 0.08, 0.10, 0.12])
cov_matrix = np.array([
    [0.040, 0.005, 0.010, 0.015],
    [0.005, 0.060, 0.020, 0.025],
    [0.010, 0.020, 0.080, 0.040],
    [0.015, 0.025, 0.040, 0.120]
])

# Generate 1000 days of historical scenarios based on this matrix
scenarios = np.random.multivariate_normal(mean_returns / 252, cov_matrix / 252, 1000)

port_returns = []
port_cvars = []

for _ in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    p_ret = np.sum(weights * mean_returns)
    port_sim_returns = scenarios.dot(weights)
    
    var_95 = np.percentile(port_sim_returns, 5)
    tail_losses = port_sim_returns[port_sim_returns <= var_95]
    
    p_cvar = -np.mean(tail_losses) * np.sqrt(252) 
    
    port_returns.append(p_ret)
    port_cvars.append(p_cvar)

port_returns = np.array(port_returns)
port_cvars = np.array(port_cvars)
ratios = port_returns / port_cvars

plt.figure(figsize=(10, 6))
scatter = plt.scatter(port_cvars, port_returns, c=ratios, cmap='viridis', marker='o', s=10, alpha=0.8)

plt.colorbar(scatter, label='Return-to-CVaR Ratio')
plt.title('Portfolio Feasible Universe: Expected Return vs. 95% CVaR', fontweight='bold', fontsize=14)
plt.xlabel('Annualized Conditional Value at Risk (CVaR) - Tail Risk', fontsize=12)
plt.ylabel('Expected Annualized Return', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Success! Right-click the graphs in your Plots pane to save them as PNGs for your PDF.")
