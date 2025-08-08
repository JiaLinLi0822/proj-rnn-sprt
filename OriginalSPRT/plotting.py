import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import ast
from collections import Counter
from sklearn.linear_model import LogisticRegression
import os

def set_style(fig, ax):
    """
    Adjusts matplotlib figure and axes style.

    Parameters:
        fig : matplotlib.figure.Figure
        ax  : matplotlib.axes.Axes or np.ndarray of Axes
    """

    # Update global font and style
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7
    })

    # Handle array of axes
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    for a in axes:
        # Hide top and right spines
        for spine in ['top', 'right']:
            a.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            a.spines[spine].set_linewidth(0.75)

        a.tick_params(direction='in', length=2, width=0.5, top=False, right=False)

        # legend样式调整
        leg = a.get_legend()
        if leg is not None:
            leg.get_frame().set_linewidth(0.5)
            for lh in leg.get_lines():
                lh.set_linewidth(1.0)
                lh.set_markersize(3.0)

    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig, ax

def plot_task_distribution(logLRs):

    lam = 3.0
    w   = 10**(logLRs/2) * np.exp(-lam*np.abs(logLRs))
    pB  = w / w.sum()
    pA  = pB[::-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.set_size_inches(2.5, 2.5)  # standard single-column width
    markerline1, stemlines1, baseline1 = plt.stem(logLRs, pA, linefmt='b-', markerfmt='bo', basefmt='k-')
    markerline2, stemlines2, baseline2 = plt.stem(logLRs, pB, linefmt='r-', markerfmt='ro', basefmt='k-')

    markerline1.set_label('pA')
    markerline2.set_label('pB')

    plt.xlabel('logLR')
    plt.ylabel('Probability')
    plt.title('Probability vs logLR')
    plt.legend()
    set_style(fig, ax)
    
    return fig, ax




def plot_decision_time_distribution(data):
    