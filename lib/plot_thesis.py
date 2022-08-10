import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp, conelp, coneqp
from scipy.stats import norm, chi2

import matplotlib.pyplot as plt
import matplotlib.lines as line
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import numdifftools as nd
from time import time

from typing import Callable
from dataclasses import dataclass
from matplotlib import patches


def PlotTrajectoryCutting(x, x0, x_opt, h, problem_name):
    x1grid = np.linspace(0.1, 0.2, 1000)
    x2grid = np.linspace(0.08, 0.16, 1000)

    xx1, xx2 = np.meshgrid(x1grid, x2grid)
    xx = np.concatenate([xx1.reshape(-1, 1), xx2.reshape(-1, 1)], axis=1)
#     print(xx[-1])

    xs = []

    for point in xx:
        if np.abs(h(point)[4]) <= 1e-3:
            xs.append(point)
#             print(x)
    xs = np.array(xs)
    xs = np.sort(xs, axis = 0)
#     print(xs[-1])
    plt.figure(figsize=(5, 5))

    plt.plot(x.T[0], x.T[1], 'bo-', label= "Log Barrier Trajectory")
    
    plt.scatter(x.T[0][:1], x.T[1][:1], c='r', s=150)
    plt.scatter(np.array(xs)[:, 0], np.array(xs)[:, 1], c='g', marker = '.', s=1)

    plt.plot([0.2, 0.2, 0.1, 0.1], [0.135, 0.08, 0.08, 0.087], c='g', linewidth=3)
    plt.scatter([x_opt[0]], [x_opt[1]], c = 'r', marker = 'X', s=150)
    plt.fill_between(np.array(xs)[:, 0], np.array(xs)[:, 1],  0.08 * np.ones_like(np.array(xs)[:, 0]), color='g', alpha=0.2)

    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)

    plt.legend(loc='best', fontsize=15)
    
    plt.savefig('trajectory_' + problem_name + '.pdf',  bbox_inches='tight', pad_inches=0.07)
    plt.show()
    
    return 0

def plot_dyn_trajectory(x, args, xlim=(-4, 5), ylim=(-4, 5), 
                        constraint = 'non-convex', R = 1, 
                        b1 = 3, b2 = 2, b3 = 3, b4 = 3, dims = (0,1), problem_name = 'LQR'):
    N = np.shape(x)[0]
    plt.figure(figsize=(5, 5))
    
    x_0, x_B, x_C, A, B_m, N, dt = args
#     if constraint == 'non-convex':
#         circle2 = plt.Circle((x_C[0], x_C[1]), 1, color='b', fill=False)
#     elif constraint == 'convex':
    circle2 = plt.Circle((x_C[0], x_C[1]), 1, color='r', fill=True, alpha = 0.2)
    arc1 = patches.Arc((x_C[0], x_C[1]), 6, 6,
                       angle=0, theta1 = 90, theta2 = 180, 
                       linewidth=2, color='g', fill = False)#, zorder=2)
    arc2 = patches.Arc((x_C[0], x_C[1]), 6, 6,
                       angle=0, theta1 = 270, theta2 = 360, 
                       linewidth=2, color='g', fill = False)#, zorder=2)
    
    ax = plt.gca()
    ax.cla() # clear things for fresh plot

    # change default range so that new circles will work
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    x1_top = 0
    x2_top = 3
    x1_low = -3
    x2_low = 0
    y1_top = b1 - x1_top
    y2_top = b1 - x2_top
    y1_low = -b2 - x1_low
    y2_low = -b2 - x2_low
    
    x1_left = -3
    x2_left = 0
    x1_right = 0
    x2_right = 3
    y1_left = b4 + x1_left
    y2_left = b4 + x2_left
    y1_right = -b3 + x1_right
    y2_right = -b3 + x2_right
#     print ('b1,b2',b1,b2, 'x1,y1b1, y1b2', x1, y1_b1, y1_b2, 'x1,y1b1, y1b2', x2, y2_b1, y2_b2)
    if constraint == 'convex':
        ax.plot([x1_top, x2_top], [y1_top, y2_top], color='g')
        ax.plot([x1_low, x2_low], [y1_low, y2_low], color='g')
        ax.add_artist(arc1)
        ax.add_artist(arc2)
        
    if constraint == 'convex_linear':
        ax.plot([x1_top, x2_top], [y1_top, y2_top], color='g')
        ax.plot([x1_low, x2_low], [y1_low, y2_low], color='g')
        ax.plot([x1_left, x2_left], [y1_left, y2_left], color='g')
        ax.plot([x1_right, x2_right], [y1_right, y2_right], color='g')

    elif constraint == 'non-convex':
        ax.add_artist(circle2)
    ax.plot(x.T[dims[0]], x.T[dims[1]], 'bo-', label= 'trajectory ')

    ax.scatter(x_B[dims[0]], x_B[dims[1]], c='r', marker='X', s=100)
#     plt.fill_between(np.array(xs)[:, 0], np.array(xs)[:, 1],  0.08 * np.ones_like(np.array(xs)[:, 0]), color='g', alpha=0.2)

    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    ax.legend(loc='best', fontsize=15)
    plt.savefig(problem_name + '_trajectory.pdf')
    plt.show()

