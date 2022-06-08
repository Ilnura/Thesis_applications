import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from scipy.stats import norm, chi2
from scipy.linalg import expm
from scipy.integrate import quad

import numdifftools as nd
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import lib.LB_optimizer as LB
from lib.functions_plots import plot_convergence_shaded


from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import GPy
from safeopt import SafeOpt, SafeOptSwarm
from safeopt import linearly_spaced_combinations, plot_3d_gp

def run_SafeOpt(n_iters, 
                f, h, 
                 x00, x_opt, 
                 d, m, 
                 sigma, bnd_l, bnd_u, gp_var = 0.01, L = 0.25, gp_num_samples = 50, linear_cons = False):
    """
    Runs SafeOpt method
    
    Args:
        n_iters: int, number of iterations
        f: callable, objective
        h: callable, returns the vector of constraint fucntions, m-dimensional
        x00: array, initial poind, d-dimensional
        x_opt: array, optimal point
        d: int, dimensionality
        m: int, number of constraints
        sigma: float, noise standard deviation
        bnd_l: float, lower bound on the box used for gridding
        bnd_u: float, upper bound on the box used for gridding
        gp_var: float, variance of the GP model
        L: float, Lipschitz continuity constant
        gp_num_samples: int, number of samples initial
        linear_cons: Bool, if we know that the constraints are linear, we can use linear kernel for them potentially
    """
    
    x = np.array([x00])
    y0 = np.array([[-f(x00)]])
    ys = -h(x00)
    if m > 1:
        y0m = [y0] + [np.array([[ys[i]]]) for i in range(0, m)]
    else:
        y0m = [y0] + [np.array([[ys]])]

    if linear_cons:
        gp_cons = [GPy.models.GPRegression(x, 
                                  y, 
                                  noise_var=gp_var**2) for y in y0m[1:]]
        gp_obj = [GPy.models.GPRegression(x, 
                                  y0, 
                                  noise_var=gp_var**2)]
        gp = gp_obj + gp_cons
    
    else: 
        gp = [GPy.models.GPRegression(x, 
                                  y, 
                                  noise_var=gp_var**2) for y in y0m]

    bounds = []
    for i in range(d):
        bounds.append([bnd_l, bnd_u])

    fmin_list = [-np.inf]
    lipschitz_list = [L]
    for i in range(m):
        fmin_list.append(0.)
        lipschitz_list.append(1.)
        
    if d <= 2:
        parameter_set = linearly_spaced_combinations(bounds, num_samples=gp_num_samples)
        opt = SafeOpt(gp, 
                      parameter_set, 
                      fmin=fmin_list
                     ,lipschitz=lipschitz_list)
    elif d > 2:
        opt = SafeOptSwarm(gp,
                           bounds=bounds, 
                           fmin=fmin_list
                           )
    else: 
        print("wrong dimension")
        
    x_traj = []
    values = []
    constraints = []
    errors_so = []
    cons_so = []
    for i in range(n_iters):
        x = opt.optimize()
        y0 = np.array([[-f(x) ]]) + np.random.normal(0, sigma)
        ys = -h(x)  + np.random.normal(0, sigma, m)
        if m > 1:
            y0m = [y0] + [np.array([[ys[i]]]) for i in range(0, m)]
        else:
            y0m = [y0] + [np.array([[ys]])]

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
        
    x_traj = np.array(x_traj)
    errors_so = np.array(errors_so)
    return x_traj, errors_so, cons_so, gp