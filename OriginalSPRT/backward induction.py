import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

# ---------- Parameters ----------
mu0, mu1, sigma = 0.0, 1.0, 1.0   # Observation distributions under H0/H1
c = 0.05                           # Sampling cost per draw
Tmax = 20                         # Maximum allowed samples (finite horizon)
N = 1001                          # Number of belief grid points

# ---------- Grid discretization ----------
pi = np.linspace(1e-6, 1-1e-6, N)

# Precompute for Bayesian update integration
dx = 0.01
x = np.arange(mu0 - 5*sigma, mu1 + 5*sigma + dx, dx)
f0x = norm.pdf(x, mu0, sigma)
f1x = norm.pdf(x, mu1, sigma)

# ---------- Backward Induction ----------
# V[t,i]: value with t samples already used (remaining = Tmax-t)
# policy[t,i]: optimal action at step t (0=H0,1=H1,2=sample)
V = np.zeros((Tmax+1, N))
policy = np.zeros((Tmax+1, N), dtype=int)
thresholds = []
sampling_hist_for_anim = [None] * (Tmax + 1)

# Immediate-stop rewards
R0 = 1 - pi   # choose H0
R1 = pi       # choose H1

# Terminal boundary: at t=Tmax, must stop
V[Tmax] = np.maximum(R0, R1) - c * Tmax
policy[Tmax] = np.where(R1 > R0, 1, 0)
sampling_hist_for_anim[Tmax] = np.full(N, -np.inf) # Can't sample at Tmax

# Backward induction loop
for t in range(Tmax-1, -1, -1):
    # compute sampling value at step t
    mix = pi[:, None] * f1x + (1 - pi)[:, None] * f0x
    p_post = (pi[:, None] * f1x) / mix
    V_next = np.interp(p_post, pi, V[t+1])
    R_samp = -c + np.sum(V_next * mix, axis=1) * dx
    sampling_hist_for_anim[t] = R_samp



    # Bellman backup
    all_R = np.vstack([R0, R1, R_samp])
    policy[t] = np.argmax(all_R, axis=0)
    V[t] = np.max(all_R, axis=0)

    # Extract thresholds
    idx = np.where(policy[t] == 2)[0]
    if idx.size:
        thresholds.append((pi[idx[0]], pi[idx[-1]]))
    else:
        thresholds.append((np.nan, np.nan))

# Append final step threshold (for t=Tmax, where no sampling is possible)
thresholds.append((np.nan, np.nan))
thresholds.reverse()

# ---------- Plot thresholds over time ----------
lows, highs = zip(*thresholds)
iters = np.arange(Tmax+1)
plt.figure(figsize=(8,4))
plt.plot(iters, lows, 'b-o', markersize=4, label='Lower threshold')
plt.plot(iters, highs, 'r-o', markersize=4, label='Upper threshold')
plt.xlabel('Time step t (samples taken)')
plt.ylabel('Belief Threshold π')
# plt.title('Sampling Thresholds vs. Time (Backward Induction)')
plt.legend()
# plt.grid(True)
# plt.ylim(0.28, 0.72)

plt.text(len(iters)/2, 0.68, 'choose H1', horizontalalignment='center')
plt.text(len(iters)/2, 0.5, 'sample', horizontalalignment='center')
plt.text(len(iters)/2, 0.32, 'choose H0', horizontalalignment='center')

plt.tight_layout()
plt.savefig('bi_thresholds.png', dpi=150)
plt.show()


# ---------- Animation: sample vs stop values ----------
# Precompute stop value
R_stop = np.maximum(R0, R1)

fig, ax = plt.subplots(figsize=(8,4))
line_samp, = ax.plot([], [], lw=2, color='C0', label='Sample Value')
line_stop, = ax.plot([], [], lw=2, color='C3', label='Stop Value')
th_low = ax.axvline(0, ls='--', color='k')
th_high = ax.axvline(0, ls='--', color='k')
ax.set_xlim(0,1)
ax.set_ylim(0,1.05)
ax.set_xlabel('Posterior π')
ax.set_ylabel('Value')
ax.legend(loc='upper right')
plt.subplots_adjust(top=0.88)
# plt.tight_layout()

def init_anim():
    line_samp.set_data([], [])
    line_stop.set_data([], [])
    th_low.set_xdata([0,0])
    th_high.set_xdata([0,0])
    ax.set_title('Value Curves Animation', pad=20)
    return line_samp, line_stop, th_low, th_high

def update_anim(t):
    line_samp.set_data(pi, sampling_hist_for_anim[t])
    line_stop.set_data(pi, R_stop)
    
    # This now works correctly because `thresholds` has been reversed
    low, high = thresholds[t]
    th_low.set_xdata([low, low] if not np.isnan(low) else [0,0])
    th_high.set_xdata([high, high] if not np.isnan(high) else [0,0])

    ax.set_title(f'Value Curves at Time Step t={t}')
    return line_samp, line_stop, th_low, th_high

ani = animation.FuncAnimation(
    fig, update_anim, frames=Tmax+1,
    init_func=init_anim, blit=True, interval=200, repeat=True # blit=True should work now
)

# Save to GIF
ani.save('bi_value_policy.gif', writer='pillow', fps=5)
print('BI animation saved: bi_value_policy.gif')
plt.show()