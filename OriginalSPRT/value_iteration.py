import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import norm
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---------- Parameters ----------
mu0, mu1, sigma = 0.0, 0.3, 1.0   # Observation parameters under H0/H1
c = 0.001                           # Fixed sampling cost
alpha = 0.001                      # Urgency coefficient: cost increases linearly with t
max_iter = 20                      # Maximum number of iterations

# ---------- Grid discretization ----------
N = 1001
pi = np.linspace(1e-6, 1-1e-6, N)

# Precompute observation PDFs
dx = 0.01
x = np.arange(mu0 - 5*sigma, mu1 + 5*sigma + dx, dx)
f0x = norm.pdf(x, mu0, sigma)
f1x = norm.pdf(x, mu1, sigma)

# ---------- Initialize containers ----------
V = np.maximum(1 - pi, pi)               # initial value function
policy = np.where(pi >= 0.5, 1, 0)       # initial policy: choose H1 if pi>=0.5
value_history  = [V.copy()]
policy_history = [policy.copy()]
thresholds     = []

fixed_costs = []   # will store constant sampling cost c at each t
time_costs  = []   # will store urgency cost alpha * t

tol = 1e-8
for t in range(max_iter):
    V_old = V.copy()

    # --- separate cost components ---
    c_base = c
    c_time = alpha * t
    fixed_costs.append(c_base)
    time_costs.append(c_time)
    c_tot = c_base + c_time

    # Compute expected value of sampling
    mix    = pi[:, None] * f1x + (1 - pi)[:, None] * f0x
    p_post = (pi[:, None] * f1x) / mix
    V_post = np.interp(p_post, pi, V_old)
    R_samp = -c_base - c_time + np.sum(V_post * mix, axis=1) * dx

    # Stopping rewards
    R0 = 1 - pi
    R1 = pi

    # Bellman update
    V = np.maximum(np.maximum(R0, R1), R_samp)
    value_history.append(V.copy())

    # Extract policy: 0=choose H0, 1=choose H1, 2=continue sampling
    all_rew = np.vstack([R0, R1, R_samp])
    policy = np.argmax(all_rew, axis=0)
    policy_history.append(policy.copy())

    # Record sampling interval thresholds (lower and upper π)
    idx = np.where(policy == 2)[0]
    if idx.size:
        thresholds.append((pi[idx[0]], pi[idx[-1]]))
    else:
        thresholds.append((np.nan, np.nan))

    # Convergence check
    if np.max(np.abs(V - V_old)) < tol:
        print(f"Converged at iteration {t}")
        break

# Convert histories to arrays
Vmat = np.array(value_history)
Pmat = np.array(policy_history)
ths_low, ths_high = zip(*thresholds)
iters = np.arange(len(thresholds))

# ---------- Plot Value Function Heatmap ----------
plt.figure(figsize=(6, 4))
plt.imshow(Vmat, origin='lower', aspect='auto',
           extent=[0, 1, 0, Vmat.shape[0]])
plt.colorbar(label='Value V(π)')
plt.xlabel('Posterior π')
plt.ylabel('Iteration')
plt.title('Value Function Heatmap')
plt.tight_layout()

# ---------- Plot Policy Function Heatmap ----------
cmap = ListedColormap(['#f08f92', '#9cbedb', '#a9d5a5'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(6, 4))
im = plt.imshow(
    Pmat,
    origin='lower',
    aspect='auto',
    extent=[0, 1, 0, Pmat.shape[0]],
    cmap=cmap,
    norm=norm
)
cbar = plt.colorbar(im, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['H₀', 'H₁', 'Sample'])  # or ['0','1','2'] if you prefer
cbar.set_label('Policy action', rotation=270, labelpad=15)

plt.xlabel('Posterior π')
plt.ylabel('Iteration')
plt.title('Policy Function (0=H₀, 1=H₁, 2=Continue Sampling)')
plt.tight_layout()
plt.show()

# ---------- Plot Thresholds Over Time ----------
plt.figure(figsize=(6, 4))
plt.plot(iters, ths_low,  'o-', label='π lower threshold')
plt.plot(iters, ths_high, 's-', label='π upper threshold')
plt.ylim(0, 1)
plt.xlabel('Iteration (time step)')
plt.ylabel('Threshold π')
plt.title('Sampling Bounds Over Time (Urgency)')
plt.legend()
plt.tight_layout()

# ---------- Plot Cost Components Over Time ----------
plt.figure(figsize=(6, 4))
plt.plot(iters, fixed_costs,  'o-', label='Fixed sampling cost $c$')
plt.plot(iters, time_costs,   's-', label='Urgency cost $\\alpha t$')
plt.plot(iters, np.array(fixed_costs)+np.array(time_costs),
         '^-', label='Total cost $c + \\alpha t$')
plt.xlabel('Iteration (time step)')
plt.ylabel('Cost')
plt.title('Sampling vs. Time Cost')
plt.legend()
plt.tight_layout()

plt.show()