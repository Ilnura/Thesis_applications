import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp, conelp, coneqp
from scipy.stats import norm, chi2
import h5py
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
        if (m != "runtimes"):    
            plt.fill_between(range(np.size(el_avg)), 
                             el_min, el_max, label='extremal values',
#                              color=colors[i], 
                             alpha=0.1)
#         plt.fill_between(range(np.size(el_avg)), 
#                              el_avg - el_std , el_avg + el_std, 
#                              label='extremal values',
#                               color=colors[i], alpha=0.3)

        if m == "accuracy":
            plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average accuracy' 
#                      ,color=colors[i]
                    )
            plt.ylabel(r"$f^0(x_t) - f^0(x^*)$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        elif m == "values":
            plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average objective' 
#                      ,color=colors[i]
                    )
            plt.ylabel(r"$f^0(x_t)$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        elif m == "constraints":
            if legends[i] == "Threshold":
                plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average constraint', 
                     color='red', 
                      linestyle='dashed')
            else: 
                plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average constraint' 
#                      ,color=colors[i]
                        )
            plt.ylabel(r"$\max_i f^i(x_t)$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        elif m == "grad_norms":
            plt.plot(range(np.size(el_avg)), 
                     el_avg, label='average grad norm'
#                      , color=colors[i]
                    )
            plt.ylabel(r"$\|\nabla f(x_t)\|$", fontsize=fontsize)
            plt.xlabel(r"$t$", fontsize=fontsize)
        elif m == "runtimes":
            ds = range(2,5)
            plt.fill_between(ds, 
                             el_min, el_max, label='extremal values'
#                              ,color=colors[i]
                             , alpha=0.1)
            plt.plot(ds, 
                     el_avg, label='average grad norm', 
                     color=colors[i], marker = 's')
            plt.ylabel("Runtime", fontsize=fontsize)
            plt.xlabel(r"$d$", fontsize=fontsize)
        
        patches.append( mpatches.Patch(color=colors[i], label=legends[i]) )  
        i += 1
        
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.legend(handles=patches, fontsize=labelsize)
    plt.savefig(fname)
    plt.show()
    return 0

def plot_convergence_shaded(ax, 
        els, 
        experiments_num, 
        m, fname, 
        colors=['royalblue', 'orange'], 
        legends=['LogBarrier', 'SafeOpt', r'Threshold'], 
        problem_name = 'None',
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
        if m == "runtimes":
            ax.fill_between(range(2,5), 
                             el_min, el_max,
                              alpha=0.2, edgecolor=None)
        else:
            ax.fill_between(range(np.size(el_avg)), 
                             el_min, el_max,
                              alpha=0.2, edgecolor=None)
        if legends[i] == r'Threshold':
            attributes = dict(linestyle='dashed', color = 'orangered', linewidth= 2.)
#             2.5)
        else:
            linestyle = attributes = dict(linewidth = 1.)
         
        if m == "runtimes":
            ax.plot(range(2,5), 
                    el_avg, label=legends[i], marker = 'x', **attributes)
        else:
            ax.plot(range(np.size(el_avg)), 
                    el_avg, label=legends[i], **attributes)
        label = dict(accuracy=r"$|f^0(x_t) - f^0(x^*)|$""\n(Accuracy)", 
                     constraints = r"$\max_i f^i(x_t)$""\n(Constraints)",
                     values = r"$f^0(x_t)$""\n(Values)",
                     grad_norms=r"$\|\nabla f(x_t)\|$""\n(Gradients Norm)",
                     runtimes = "Runtime")[m]

        if ax.is_last_row():
            ax.set_xlabel(r"$t$", fontsize=fontsize)
            if m == "runtimes":
                ax.set_xlabel(r"$d$", fontsize=fontsize)
                ax.text(2., 200., problem_name)
                ax.set_xticks([ 2, 3, 4])
#                 ax.annotate(problem_name, xy = (2,40))
        if ax.is_first_col():
            ax.set_ylabel(label, fontsize=fontsize)
            
            
def read_LineBO(file, methods_num, f):
    ysss = []
    xsss = []
    ysss_min = []
    zsss_max = []
    ysss_exact = []
    zsss_exact = []
    for j in range(methods_num):
        xss = []
        yss = []
        yss_min = []
        zss_max = []
        yss_exact = []
        zss_exact = []
        for i in range(10):
            xs = []
            ys = []
            ys_min = []
            zs_max = []
            ys_exact = []
            zs_exact = []
            y_min = 100.
            z_max = -100.
            for g in file[str(j)][str(i)]:
                xs.append(g['x'])
    #             print(g['x'], g['s'])
                ys.append(-g['y'])
                ys_exact.append(-g['y_exact'])
                zs_exact.append(np.max(g['s']))
    #             for i in range(np.size(g['s'])):
#                 y_min = min(-g['y_exact'], y_min)
                y_min = min(f(g['x']), y_min)
                z_max = max(np.max(g['s']), z_max)
                ys_min.append(y_min)
                zs_max.append(z_max)

            xs = np.array(xs)
            ys = np.array(ys)
            ys_exact = np.array(ys_exact)
            ys_min = np.array(ys_min)
            zs_max = np.array(zs_max)
            
            xss.append(xs)
            yss.append(ys)
            yss_exact.append(ys_exact)
            yss_min.append(ys_min)
            zss_max.append(zs_max)
    #         zss_exact.append(zs_exact)
        yss = np.array(yss)
        xss = np.array(xss)
        yss_exact = np.array(yss_exact)
        yss_min = np.array(yss_min)
        zss_max = np.array(zss_max)
    #     zss_exact = np.array(zss_exact)

        ysss_exact.append(yss_exact)
        ysss_min.append(yss_min)
        zsss_max.append(zss_max)
        xsss.append(xss)
    #     zsss_exact.append(zss_exact)
#     print(g)
    return (ysss_exact, ysss_min, zsss_max, xsss)

def plot_experiments_linebo(d, axes_col, problem_name, experiments_num, f, 
                            cons = True, 
                            SafeOpt = False, 
                            opt_val = 0.):
    if (d <= 4 and SafeOpt == True): 
        with open('../runs/SafeOpt_' + problem_name + '_d' + str(d) + '.npy', 'rb') as file:
            SO_errors = np.load(file)
            SO_cons = np.load(file) 
            shapeSO = np.shape(SO_errors)
            legends=['LB-SGD',   
                   'RandomLineBO', 
                   'CoordinateLineBO', 
                   'AscentLineBO',
                   'SafeOpt' 
                  ]
    else:
        legends=['LB-SGD',   
                   'RandomLineBO', 
                   'CoordinateLineBO', 
                   'AscentLineBO' 
                  ]
    with open('../runs/LB_SGD_' + problem_name + '_d' + str(d) + '.npy', 'rb') as file:
        LB_errors = np.load(file)
        LB_cons = np.load(file) 
    
    file = h5py.File('/Users/ilnura/libs/LineBO/runs/'+ problem_name + '/' + problem_name + '_'+ str(d) + '/data/evaluations.hdf5', 'r')
    (ys_exact, LineBO_errors, LineBO_cons, xs) = read_LineBO(file, 3, f)
    shapeLB = np.shape(LB_errors)

    if SafeOpt == True:
        shapeSO = np.shape(SO_errors)
        errors = [LB_errors + opt_val * np.ones(shapeLB)
                  ,LineBO_errors[0] #+ opt_val * np.ones(shape)
                  ,LineBO_errors[1] #+ opt_val * np.ones(shape)
                  ,LineBO_errors[2] #+ opt_val * np.ones(shape)
                  ,SO_errors + opt_val * np.ones(shapeSO)
                 ]
        constraints = [LB_cons, 
                      LineBO_cons[0], 
                      LineBO_cons[1], 
                      LineBO_cons[2],
                      SO_cons, 
                      np.zeros(shapeLB)]
    else:
        errors = [LB_errors + opt_val * np.ones(shapeLB)
                  ,LineBO_errors[0] #+ opt_val * np.ones(shape)
                  ,LineBO_errors[1] #+ opt_val * np.ones(shape)
                  ,LineBO_errors[2] #+ opt_val * np.ones(shape)
                 ]
        constraints = [LB_cons, 
                      LineBO_cons[0], 
                      LineBO_cons[1], 
                      LineBO_cons[2],
                      np.zeros(shapeLB)]
#     if (d <= 4):
    if cons:
        plot_convergence_shaded(axes_col[0], errors, 
                          experiments_num, 
                          colors=['royalblue', 'orange'], 
                          legends=legends,
                          fname = "../runs/objective_" + problem_name +'_d' + str(d), m="values")
        axes_col[0].annotate(r'$d = {}$'.format(d), (0.7, 0.85), fontsize=11, xycoords='axes fraction')

        plot_convergence_shaded(axes_col[1], constraints,
                          experiments_num, 
                          colors=['royalblue', 'orange', 'orangered'], 
                          legends=legends + [r'Threshold'],
                          fname="../runs/constraints_" + problem_name +'_d' + str(d), m="constraints")
    else: 
        plot_convergence_shaded(axes_col, errors, 
                          experiments_num, 
                          colors=['royalblue', 'orange'], 
                          legends=legends,
                          fname = "../runs/objective_" + problem_name +'_d' + str(d), m="values")
        axes_col.annotate(r'$d = {}$'.format(d), (0.7, 0.85), fontsize=11, xycoords='axes fraction')

def plot_experiments(d, axes_col, problem_name, experiments_num):
    with open('../runs/SafeOpt_' + problem_name + '_d' + str(d) + '.npy', 'rb') as file:
        SO_errors = np.load(file)
        SO_cons = np.load(file)
        SO_runtimes = np.load(file) 
    with open('../runs/LB_SGD_' + problem_name + '_d' + str(d) + '.npy', 'rb') as file:
        LB_errors = np.load(file)
        LB_cons = np.load(file) 
        LB_runtimes = np.load(file)
#     experiments_num = 5
    plot_convergence_shaded(axes_col[0], [LB_errors, SO_errors], 
                          experiments_num, 
                          colors=['royalblue', 'orange'], 
                          legends=['LB-SGD', 'SafeOpt' ],
                          fname = "../runs/objective_" + problem_name +'_d' + str(d), m="accuracy")
    axes_col[0].annotate(r'$d = {}$'.format(d), (0.7, 0.85), fontsize=11, xycoords='axes fraction')

    shape = np.shape(SO_errors)
    plot_convergence_shaded(axes_col[1], [LB_cons, SO_cons, np.zeros(shape)], 
                          experiments_num, 
                          colors=['royalblue', 'orange', 'orangered'], 
                          legends=['LB-SGD', 'SafeOpt', r'Threshold'],
                          fname="../runs/constraints_" + problem_name +'_d' + str(d), m="constraints")
    
def plot_runtimes(axes_col, problem_name, experiments_num, ds, figsize=(5, 5)):
    SO_runtimes = []
    LB_runtimes = []
    for d in ds:
        with open('../runs/SafeOpt_' + problem_name + '_d' + str(d) + '.npy', 'rb') as file:
            SO_errors = np.load(file)
            SO_cons = np.load(file)
            SO_times = np.load(file)
        with open('../runs/LB_SGD_' + problem_name + '_d' + str(d) + '.npy', 'rb') as file:
            LB_errors = np.load(file)
            LB_cons = np.load(file) 
            LB_times = np.load(file)

        SO_runtimes.append(SO_times)
        LB_runtimes.append(LB_times)

    SO_runtimes = np.array(SO_runtimes).T
    LB_runtimes = np.array(LB_runtimes).T
    if problem_name == 'QP':
        LineBO_runtimes = np.array([ np.repeat(8.180, 10), 
                                     np.repeat(17.837, 10),
                                     np.repeat(40.8, 10)
                                   ]).T
    elif problem_name == 'Rosenbrock':
        LineBO_runtimes = np.array([np.repeat(7.584, 10), 
                                     np.repeat(10.593, 10),
                                     np.repeat(13.293, 10)
                                   ]).T
        
    plot_convergence_shaded(axes_col, 
                            [LB_runtimes, SO_runtimes, LineBO_runtimes], 
                            experiments_num, 
                            legends=['LB-SGD', 'SafeOpt', 'LineBO'],
                            m = "runtimes",  
                            fname = "../runs/runtimes_" + problem_name +'_d' + str(d),
                            problem_name = problem_name
                          )
    SO_t_avg = np.mean(SO_runtimes, axis=0)
    LB_t_avg = np.mean(LB_runtimes, axis=0)
    LineBO_t_avg = np.mean(LineBO_runtimes, axis=0)
    return (SO_t_avg, LB_t_avg, LineBO_t_avg)
