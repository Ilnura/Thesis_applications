# not updated, should update according to the jupiter notebook

%load_ext autoreload
%autoreload 2

from skopt import benchmarks as bench

import os
import sys
sys.path.append('..')

import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import norm, chi2
from scipy.linalg import expm
from scipy.integrate import quad

import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
%matplotlib inline

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import lib.LB_convex_optimizer as LB
from lib.functions_plots import PlotTrajectory, PlotConvergence, PlotConvergenceShaded


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import GPy
from safeopt import SafeOpt, SafeOptSwarm
from safeopt import linearly_spaced_combinations, plot_3d_gp

from safeopt import SafeOpt

def run_SafeOpt(n_iters, x00, x_opt, d):
    x = np.array([x00])
    
    y0 = np.array([[-f(x00)]])
    ys = -h(x00)
    y0m = [y0] + [np.array([[ys[i]]]) for i in range(0, m)]
    gp = [GPy.models.GPRegression(x, 
                                  y, 
                                  noise_var=0.01**2) for y in y0m]

    bounds = []
    for i in range(d):
        bounds.append([-1, 1])

    parameter_set = linearly_spaced_combinations(bounds,
                                                 num_samples=100)
    
    fmin_list = [-np.inf]
    lipschitz_list = [0.25]
    for i in range(m):
        fmin_list.append(0.)
        lipschitz_list.append(1.)
        
    if d <= 2:
        opt = SafeOpt(gp, 
                      parameter_set, 
                      fmin=fmin_list, 
                      lipschitz=lipschitz_list)
    elif d > 2:
        opt = SafeOptSwarm(gp, 
                           bounds=bounds, 
                           fmin=fmin_list)
    else: 
        print("wrong dimension")
        
    x_traj = []
    values = []
    constraints = []
    errors_so = []
    cons_so = []
    for i in range(n_iters):
        x = opt.optimize()
        y0 = np.array([[-f(x) ]]) + np.random.normal(0, 0.001)
        ys = - h(x)  + np.random.normal(0, 0.001, m)
        y0m = [y0] + [np.array([[ys[i]]]) for i in range(0, m)]
        f_list = []
        for y in y0m:
            f_list.append(float(y))
            
        opt.add_new_data_point(x, f_list)

        x_traj.append(x)
        values.append(f(x) - f(x_opt))
        constraints.append(np.max(h(x)))
        best_value = f(x) - f(x_opt)
        best_value = np.min(np.array(values))
        errors_so.append(best_value)
        worst_constraint = np.max(constraints)
        cons_so.append(worst_constraint)
        
#     opt.plot(n_iters - 1)#, axis=0, figure=None, plot_3d=False)
    x_traj = np.array(x_traj)
    errors_so = np.array(errors_so)
    return x_traj, errors_so, cons_so, gp


def run_exp_SafeOpt_LB_SGD(d = 2, experiments_num = 5, 
                           n_iters = 100, 
                           n = 1, 
                           M0 = 0.5 / d, Ms = 0. * np.ones(m), 
                           x00 = np.zeros(d), 
                           x_opt =  np.ones(d) / d**0.5, 
                           f = f, h = h, 
                           sigma = 0.001, nu = 0.01, 
                           eta0 = 0.05, 
                           T = 3):

    errors_toy2 = []
    cons_toy2 = []
    for i in range(experiments_num):
        x_traj, errors_so, cons_so, gp = run_SafeOpt(n_iters, x00, x_opt, d)
        plt.show()
        plt.plot(x_traj[:,0], x_traj[:,1], "o")
        plt.plot(x_traj[-1,0], x_traj[-1,1], "ro")
        plt.plot(x_opt[0], x_opt[1], "g*")
        x_size = x_traj.shape[0]
        plt.show()
        errors_toy2.append(errors_so)
        cons_toy2.append(cons_so)
    
    my_oracle3 = LB.Oracle(
        f = f,
        h = h, 
        sigma = sigma,
        hat_sigma = 0.01,
        delta = 0.01,
        m = m,
        d = d,
        nu = nu,
        zeroth_order = True,
        n = n)

    opt3 = LB.SafeLogBarrierOptimizer(
        x00 = x00,
        x0 = x00,
        M0 = M0,
        Ms = Ms,
        sigma = my_oracle3.sigma,
        hat_sigma = my_oracle3.hat_sigma,
        init_std = 0.2,
        eta0 = eta0,
        oracle = my_oracle3,
        f = f,
        h = h,
        d = d,
        m = m,
        reg = 0.0001,
        x_opt = x_opt,
        factor = 0.85,
        T = T,
        K = int(n_iters / T / 2. / n),
#         factor = 0.5,
#         T = 100,
#         K = 4,
        experiments_num = experiments_num,
        mu = 0.,
        convex = True,
        random_init = True,
        no_break = True)

    opt3.run_average_experiment()
    
    return (errors_toy2, cons_toy2, gp, opt3)  

##################################################### QP ###########################################################

####################### d = 2
d = 2
m = 2 * d 
x0 = np.ones(d) * 0.2
x_opt =  np.ones(d) / d**0.5
experiments_num = 10
# n = int(d / 2)
n = 1

# constraints
def h(x):
    d = np.size(x)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return A.dot(x) - b 

# objective
def f(x):    
    d = np.size(x)
    xx = 2. * np.ones(d)
#     xx[0] = 5.
    return np.linalg.norm(x - xx, 2)**2 / 4. / d


(errors_SO_d2, cons_SO_d2, gp, opt_d2) = run_exp_SafeOpt_LB_SGD(d=2, 
                                                                experiments_num=experiments_num, 
                                                                n_iters=100,
                                                                n=n)
for i in range(experiments_num):
    opt_d2.errors_total[i] = np.repeat(opt_d2.errors_total[i], 2 * n)
    opt_d2.constraints_total[i] = np.repeat(opt_d2.constraints_total[i], 2 * n)
    
PlotConvergenceShaded([opt_d2.errors_total, errors_SO_d2], 
                      opt_d2.experiments_num, 
                      colors=['blue', 'orange'], 
                      legends=['LB-SGD', 'SafeOpt'],
                      figsize=(10, 5),
                      fname = "objective_QP_d2",  m = "accuracy")

shape = np.shape(opt_d2.constraints_total)

PlotConvergenceShaded([opt_d2.constraints_total, cons_SO_d2, np.zeros(shape)], 
                      opt_d2.experiments_num, 
                      colors=['blue', 'orange', 'red'], 
                      legends=['LB-SGD', 'SafeOpt', '0'],
                      figsize=(10, 5),
                      fname = "constraints_QP_d2",  m = "constraints")

############################ d = 3
d = 3
m = 2 * d 
x0 = np.ones(d) * 0.2
x_opt =  np.ones(d) / d**0.5
# n = int(d / 2)
n = 1
experiments_num = 10

# constraints
def h(x):
    d = np.size(x)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return A.dot(x) - b 

# objective
def f(x):    
    d = np.size(x)
    xx = 2. * np.ones(d)
#     xx[0] = 5.
    return np.linalg.norm(x - xx, 2)**2 / 4. / d 

(errors_SO_d3, cons_SO_d3, gp, opt_d3) = run_exp_SafeOpt_LB_SGD(d=3, experiments_num=experiments_num, 
                                                                n_iters=200, n=n)
for i in range(experiments_num):
    opt_d3.errors_total[i] = np.repeat(opt_d3.errors_total[i], 2 * n)
    opt_d3.constraints_total[i] = np.repeat(opt_d3.constraints_total[i], 2 * n)
    
PlotConvergenceShaded([opt_d3.errors_total, errors_SO_d3], 
                      opt_d3.experiments_num, 
                      colors=['blue', 'orange'], 
                      legends=['LB-SGD', 'SafeOpt'],
                      figsize=(10, 5),
                      fname = "objective_QP_d3",  m = "accuracy")

shape = np.shape(opt_d3.constraints_total)

PlotConvergenceShaded([opt_d3.constraints_total, cons_SO_d3, np.zeros(shape)], 
                      opt_d3.experiments_num, 
                      colors=['blue', 'orange', 'red'], 
                      legends=['LB-SGD', 'SafeOpt', '0'],
                      figsize=(10, 5),
                      fname = "constraints_QP_d3",  m = "constraints")

############################ d = 4
d = 4
m = 2 * d 
x_opt =  np.ones(d) / d**0.5
experiments_num = 10
n = int(d / 2)

def h(x):
    d = np.size(x)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return A.dot(x) - b 

def f(x):    
    d = np.size(x)
    xx = 2. * np.ones(d)
#     xx[0] = 5.
    return np.linalg.norm(x - xx, 2)**2 / 4. / d 

(errors_SO_d4, cons_SO_d4, gp, opt_d4) = run_exp_SafeOpt_LB_SGD(d=4, 
                                                                experiments_num=experiments_num, 
                                                                n_iters=d * 100, 
                                                                n=n)
for i in range(experiments_num):
    opt_d4.errors_total[i] = np.repeat(opt_d4.errors_total[i], 2 * n)
    opt_d4.constraints_total[i] = np.repeat(opt_d4.constraints_total[i], 2 * n )
    
PlotConvergenceShaded([opt_d4.errors_total, errors_SO_d4], 
                      opt_d4.experiments_num, 
                      colors=['blue', 'orange'], 
                      legends=['LB-SGD', 'SafeOpt'],
                      figsize=(10, 5),
                      fname = "objective_QP_d4_1",  m = "accuracy")

shape = np.shape(opt_d4.constraints_total)

PlotConvergenceShaded([opt_d4.constraints_total, cons_SO_d4, np.zeros(shape)], 
                      opt_d4.experiments_num, 
                      colors=['blue', 'orange', 'red'], 
                      legends=['LB-SGD', 'SafeOpt', '0'],
                      figsize=(10, 5),
                      fname = "constraints_QP_d4_1",  m = "constraints")

################################ d = 5

d = 5
m = 2 * d 
x_opt =  np.ones(d) / d**0.5
experiments_num = 10
n = int(d / 2)

def h(x):
    d = np.size(x)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return A.dot(x) - b 

def f(x):    
    d = np.size(x)
    xx = 2. * np.ones(d)
#     xx[0] = 5.
    return np.linalg.norm(x - xx, 2)**2 / 4. / d 

(errors_SO_d5, cons_SO_d5, gp, opt_d5) = run_exp_SafeOpt_LB_SGD(d=5, 
                                                                experiments_num=experiments_num, 
                                                                n_iters=d * 100, 
                                                                n=n)
for i in range(experiments_num):
    opt_d5.errors_total[i] = np.repeat(opt_d5.errors_total[i], 2 * n)
    opt_d5.constraints_total[i] = np.repeat(opt_d5.constraints_total[i], 2 * n )
    
PlotConvergenceShaded([opt_d5.errors_total, errors_SO_d5], 
                      opt_d5.experiments_num, 
                      colors=['blue', 'orange'], 
                      legends=['LB-SGD', 'SafeOpt'],
                      figsize=(10, 5),
                      fname = "objective_QP_d4_1",  m = "accuracy")

shape = np.shape(opt_d5.constraints_total)

PlotConvergenceShaded([opt_d5.constraints_total, cons_SO_d5, np.zeros(shape)], 
                      opt_d5.experiments_num, 
                      colors=['blue', 'orange', 'red'], 
                      legends=['LB-SGD', 'SafeOpt', '0'],
                      figsize=(10, 5),
                      fname = "constraints_QP_d4_1",  m = "constraints")


##################################################### Rosenbrock ###########################################################
# Rosenbrock fucntion


####################### d = 2
d = 2
m = 2
experiments_num = 10
n_iters = 100
n = 2

def f(x):
    f_rosenbrock = 0.
    for i in range(d - 1):
        f_rosenbrock += 100. * (x[i + 1] - x[i]**2)**2 + (1. - x[i])**2
    return f_rosenbrock

def h(x):
    h1 = np.linalg.norm(x,2)**2 - 1.
    h2 = np.linalg.norm(x + 0.1 * np.ones(d),2)**2 - d * 0.1
    return np.array([h1, h2]) 
#     return np.array([(x[0] + 0.1)**2 + (x[1] - 0.1)**2 - 0.2, x[0]**2 + x[1]**2 - 1.])

(errors_SO_r, cons_SO_r, gp_r, opt_r) = run_exp_SafeOpt_LB_SGD(d = 2, 
                                                               experiments_num = experiments_num, 
                                                               n_iters = n_iters, 
                                                               n = n, 
                                                               M0 = d * 10., 
                                                               Ms = d * np.ones(m),
                                                               x00 = -0.05 * np.ones(2), 
                                                               x_opt = np.ones(d), 
                                                               sigma = 0.001,
                                                               nu = 0.01,
                                                               f = f, 
                                                               h = h)
for i in range(experiments_num):
    opt_r.errors_total[i] = np.repeat(opt_r.errors_total[i], 2 * n)
    opt_r.constraints_total[i] = np.repeat(opt_r.constraints_total[i], 2 * n)
    
PlotConvergenceShaded([opt_r.errors_total, errors_SO_r], 
                      opt_r.experiments_num, 
                      colors=['blue', 'orange'], 
                      legends=['LB-SGD', 'SafeOpt'],
                      figsize=(10, 5),
                      fname = "objective_rosenbrock",  m = "accuracy")

shape = np.shape(opt_r.constraints_total)

PlotConvergenceShaded([opt_r.constraints_total, cons_SO_r, np.zeros(shape)], 
                      opt_r.experiments_num, 
                      colors=['blue', 'orange', 'red'], 
                      legends=['LB-SGD', 'SafeOpt', '0'],
                      figsize=(10, 5),
                      fname = "constraints_rosenbrock",  m = "constraints")


    