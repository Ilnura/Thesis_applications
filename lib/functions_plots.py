import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp, conelp, coneqp
from scipy.stats import norm, chi2

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"

# %matplotlib inline

def PlotTrajectory(x, x0, x_opt, g):
    x1 = np.linspace(-1, 1, 1000)
    x2 = np.linspace(-1, 1, 1000)
    
#     xx, yy = np.meshgrid(x1, x2)
#     xxx = np.concatenate([xx.reshape(-1, 1), yy.reshape(-1, 1)],axis=1)
#     ys = []
#     xs = []
#     m = np.size(g(xxx[0]))
#     for xxxx in xxx:
#         if np.abs(g(xxxx)[m-1]) <= 1e-3:
#             ys.append(g(xxxx)[m-1])
#             xs.append(xxxx)

    plt.figure(figsize=(3, 5))
    plt.plot(x.T[0], x.T[1], 'bo-', label= "Log Barrier Trajectory")
    
    plt.scatter(x.T[0,:1], x.T[1,:1], c = 'r', s=150)
    plt.scatter([x_opt[0]], [x_opt[1]], c = 'g', marker = '*', s=350)

    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend(loc='best', fontsize=15)
    plt.show()
#     plt.savefig('trajectory_convergence.eps',  bbox_inches='tight', pad_inches=0.07)
    return 0



def PlotConvergence(experiments_num, el, m):

    plt.figure(figsize=(5, 2.5))

#     plt.legend(loc='best', fontsize=15)
    plt.xlabel(r"$t$", fontsize=20)
    
    if (m == "accuracy"):
        for element in el:
            plt.plot(range(np.size(element)), element, color='blue')
        plt.ylabel(r"$f(x_t) - f(x^*)$", fontsize=20)
    elif (m == "constraints"):
        for element in el:
            plt.plot(range(np.size(element)), element, color='green')
        plt.ylabel(r"$\max_i f^i(x_t)$", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.show()
    
#     plt.savefig(m + 'convergence.eps',  bbox_inches='tight', pad_inches=0.07)
#     print(el_avg[-1])
    return 0



def PlotConvergenceShaded(els, 
        experiments_num, 
        m, fname, 
        colors=['blue', 'orange'], 
        legends=['LogBarrier'], 
        fontsize=20, 
        labelsize=20,
        figsize=(5, 5)):
    #Averaged plots     
    plt.figure(figsize=figsize)
    i = 0
    patches = []
    for el in els:
        if experiments_num > 1:
            el_avg = np.mean(el, axis=0)
            el_max = np.max(el, axis=0)
            el_min = np.min(el, axis=0)
            el_std = np.std(el, axis=0)
        else:
            el_avg = el
            el_max = el
            el_min = el
            el_std = np.zeros(np.size(el))
            
        plt.fill_between(range(np.size(el_avg)), 
                             el_min, el_max, label='extremal values',
                             color=colors[i], alpha=0.1)
#         plt.fill_between(range(np.size(el_avg)), 
#                              el_avg - el_std , el_avg + el_std, 
#                              label='extremal values',
#                               color=colors[i], alpha=0.3)

        if m == "accuracy":
            plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average accuracy', 
                     color=colors[i])
            plt.ylabel(r"$f^0(x_t) - f^0(x^*)$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        elif m == "values":
            plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average objective', 
                     color=colors[i])
            plt.ylabel(r"$f^0(x_t)$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        elif m == "constraints":
            if legends[i] == "Threshold":
                plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average constraint', 
                     color=colors[i], linestyle='dashed')
            else: 
                plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average constraint', 
                     color=colors[i])
            plt.ylabel(r"$\max_i f^i(x_t)$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        elif m == "grad_norms":
            plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average grad norm', 
                     color=colors[i])
            plt.ylabel(r"$\|\nabla f(x_t)\|$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        
        patches.append( mpatches.Patch(color=colors[i], label=legends[i]) )  
        i += 1
        
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(handles=patches, fontsize=labelsize)
    plt.savefig(fname)
    plt.show()
    return 0


####################################################################################
###     Yardens version  #####################################################
#########################################################################

def plot_convergence_shaded(ax, els, 
        experiments_num, 
        m, fname, 
        colors=['royalblue', 'orange'], 
        legends=['LogBarrier', 'SafeOpt', r'Threshold'], 
        fontsize=11, 
        labelsize=11):
    for i, el in enumerate(els):
        if experiments_num > 1:
            el_avg = np.mean(el, axis=0)
            el_max = np.max(el, axis=0)
            el_min = np.min(el, axis=0)
            el_std = np.std(el, axis=0)
        else:
            el_avg = el
            el_max = el
            el_min = el
            el_std = np.zeros(np.size(el))
        ax.fill_between(range(np.size(el_avg)), 
                             el_min, el_max,
                              alpha=0.2, edgecolor=None)
        if legends[i] == r'Threshold':
            attributes = dict(linestyle='dashed', color = 'orangered', linewidth= 2.)
#             2.5)
        else:
            linestyle = attributes = dict(linewidth = 1.)
        ax.plot(range(np.size(el_avg)), 
        el_avg, label=legends[i], **attributes)
        label = dict(accuracy=r"$|f^0(x_t) - f^0(x^*)|$""\n(Accuracy)", 
                     constraints=r"$\max_i f^i(x_t)$""\n(Constraints)",
                     values = r"$f^0(x_t)$""\n(Values)",
                     grad_norms=r"$\|\nabla f(x_t)\|$""\n(Gradients Norm)")[m]
        if ax.is_last_row():
            ax.set_xlabel(r"$t$", fontsize=fontsize)
        if ax.is_first_col():
            ax.set_ylabel(label, fontsize=fontsize)
