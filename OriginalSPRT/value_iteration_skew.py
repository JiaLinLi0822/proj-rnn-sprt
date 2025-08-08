import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# === Define skewed observation distributions ===
ell = np.linspace(-0.7, 0.7, 16)
lam = 3.0              # Increased from 3.0 to make observations more informative
w   = 10**(ell/2) * np.exp(-lam * np.abs(ell))
pA  = w / w.sum()
pB  = pA[::-1]
dell = ell[1] - ell[0]   # integration step size

# ---------- Parameters ----------
c = 0.01             # Fixed sampling cost (reduced from 0.001)
alpha = 0.001         # Urgency coefficient: cost increases linearly with t (reduced)
max_iter = 20          # Maximum number of iterations

# ---------- Prior grid ----------
N = 1001
pi = np.linspace(1e-6, 1 - 1e-6, N)

# ---------- Initialize containers ----------
V = np.maximum(1 - pi, pi)          # initial value function
policy = np.where(pi >= 0.5, 1, 0)  # initial policy: choose H1 if pi>=0.5
value_history  = [V.copy()]
policy_history = [policy.copy()]
thresholds     = []
fixed_costs    = []
time_costs     = []

tol = 1e-8
for t in range(max_iter):
    V_old = V.copy()

    # --- separate cost components ---
    c_base = c
    c_time = alpha * t
    fixed_costs.append(c_base)
    time_costs.append(c_time)

    # === compute expected value of sampling ===
    # mix = p(x|A)*P(A) + p(x|B)*(1-P(A))
    mix    = pi[:, None] * pA[None, :] + (1 - pi)[:, None] * pB[None, :]
    # posterior P(A|x)
    p_post = (pi[:, None] * pA[None, :]) / mix
    # interpolate previous V onto those posteriors
    V_post = np.interp(p_post.ravel(), pi, V_old).reshape(N, -1)
    # Properly compute expected value as weighted sum (mix already sums to 1 for each pi)
    R_samp = -c_base - c_time + np.sum(V_post * mix, axis=1)

    # stopping rewards
    R0 = 1 - pi
    R1 = pi

    # Bellman update
    V = np.maximum(np.maximum(R0, R1), R_samp)
    value_history.append(V.copy())

    # extract policy: 0=H0, 1=H1, 2=sample
    all_rew = np.vstack([R0, R1, R_samp])
    policy = np.argmax(all_rew, axis=0)
    policy_history.append(policy.copy())

    # record sampling‐interval thresholds
    idx = np.where(policy == 2)[0]
    if idx.size:
        thresholds.append((pi[idx[0]], pi[idx[-1]]))
    else:
        thresholds.append((np.nan, np.nan))

    # convergence check
    change = np.max(np.abs(V - V_old))
    if change < tol:
        print(f"Converged at iteration {t}")
        break

# === convert histories to arrays ===
Vmat = np.array(value_history)
Pmat = np.array(policy_history)
ths_low, ths_high = zip(*thresholds)
iters = np.arange(len(thresholds))

# ---------- Plot the pA and pB ----------
plt.figure(figsize=(6, 6))
markerline1, stemlines1, baseline1 = plt.stem(ell, pA, linefmt='b-', markerfmt='bo', basefmt='k-')
markerline2, stemlines2, baseline2 = plt.stem(ell, pB, linefmt='r-', markerfmt='ro', basefmt='k-')

# Set labels for the legend
markerline1.set_label('pA')
markerline2.set_label('pB')

plt.legend()
plt.grid(True)
plt.xlabel('logLR')
plt.ylabel('Probability')
plt.title('Probability vs logLR')
plt.legend()
plt.show()

# ---------- Plot Value Function Heatmap ----------
plt.figure(figsize=(6, 4))
plt.imshow(
    Vmat,
    origin='lower',
    aspect='auto',
    extent=[0, 1, 0, Vmat.shape[0]]
)
plt.colorbar(label='Value V(π)')
plt.xlabel('Posterior π')
plt.ylabel('Iteration')
plt.title('Value Function Heatmap')
plt.tight_layout()

# ---------- Plot Policy Function Heatmap ----------
cmap = ListedColormap(['#f08f92', '#9cbedb', '#a9d5a5'])
bounds = [-0.5, 0.5, 1.5, 2.5]
norm  = BoundaryNorm(bounds, cmap.N)

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
cbar.ax.set_yticklabels(['H₀', 'H₁', 'Sample'])
cbar.set_label('Policy action', rotation=270, labelpad=15)

plt.xlabel('Posterior π')
plt.ylabel('Iteration')
plt.title('Policy Function (0=H₀, 1=H₁, 2=Continue Sampling)')
plt.tight_layout()

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
plt.plot(iters, fixed_costs, 'o-', label='Fixed sampling cost $c$')
plt.plot(iters, time_costs,  's-', label='Urgency cost $\\alpha t$')
plt.plot(
    iters,
    np.array(fixed_costs) + np.array(time_costs),
    '^-',
    label='Total cost $c + \\alpha t$'
)
plt.xlabel('Iteration (time step)')
plt.ylabel('Cost')
plt.title('Sampling vs. Time Cost')
plt.legend()
plt.tight_layout()

plt.show()