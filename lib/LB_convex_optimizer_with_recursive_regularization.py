# Write the documentation
# Remove recurcive regularization

import numpy as np
from numpy import linalg
from scipy.optimize import linprog, minimize_scalar
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp, conelp, coneqp
from scipy.stats import norm, chi2
from utils.functions_plots import PlotTrajectory, PlotConvergence, PlotConvergenceShaded
import matplotlib.pyplot as plt
import matplotlib.lines as line
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import numdifftools as nd

from typing import Callable
from dataclasses import dataclass

@dataclass
class Oracle:
    f: Callable = None
    h: Callable = None
    df: np.array = None
    dh: np.array = None 
    sigma: float = None   
    hat_sigma: float = None 
    delta: float = None
    m: int = None
    d: int = None
    nu: float = None
    objective_value: float = None
    constraints_values: np.array = None
    alphas: np.array = None
    objective_grad: np.array = None
    constraints_grad: np.array = None
    zeroth_order: bool = None
        
    def sample(self, x: np.array) -> None:
        self.objective_value = self.f(x) + np.random.normal(0, self.sigma)
        self.constraints_values = self.h(x) + np.random.normal(0, self.sigma, self.m)
        self.df = nd.Gradient(self.f)(x)
#         print(self.df)
        self.dh = nd.Gradient(self.h)(x)
        if self.zeroth_order:
            s_unnormalized = np.random.normal(0, 1, self.d)
            s = s_unnormalized / np.linalg.norm(s_unnormalized)
            self.objective_grad = (self.d *
                                   (self.f(x + self.nu * s) 
                                    + np.random.normal(0, self.sigma) - self.objective_value)
                                   / self.nu) * s
            self.constraints_grad = np.outer((self.d *
                                    (self.h(x + self.nu * s) + np.random.normal(0, self.sigma, self.m) -
                                    self.constraints_values) / self.nu), s)
            self.alphas = - self.constraints_values - (np.log(1. / self.delta))**0.5 * self.sigma * np.ones(self.m) - self.nu * np.ones(self.m)
        else:
            self.objective_grad = self.df + np.random.normal(0, self.hat_sigma, self.d)
            self.constraints_grad = self.dh + np.random.normal(0, self.hat_sigma, (self.m, self.d))
            self.alphas = - self.constraints_values - (np.log(1. / self.delta))**0.5 * self.sigma / 2. * np.ones(self.m)

@dataclass
class Optimizer:
    x00: np.array = None
    x0: np.array = None
    M0: float = None
    eps: float = None
    sigma: float = None
    hat_sigma: float = None
    oracle: Oracle = None
    f: Callable = None
    d: float = None
    reg: float = None
    x_opt: float = None
    T: int = None
    K: int = None
    suffix: int = None
    S: int = None
    step: np.array = None
    experiments_num: int = None
    mu: float = None
    mu0: float = None
    mus: list = None
    s: int = None
    xs: list = None
    tk: int = None
    convex: bool = None
    random_init: bool = False
    no_break: bool = True
    x_total: list = None
    errors_total: list = None
    grad_norm_total: list = None
    x_trajectory: np.array = None
    errors_trajectory: np.array = None
    grad_norm_trajectory: np.array = None


    def compute_gamma(self, t, M) -> float:
        gamma = min(1. / M,  self.mu0 / self.mu / (t + 1))
#         gamma = 1. / M
        return gamma
    
    def SGD(self, M, T, xt):
        for t in range(T):
            self.oracle.sample(xt)
            self.step = self.oracle.objective_grad + self.regularizer_grad(xt)
#             gamma = self.compute_gamma(self.tk, M)
            gamma = self.compute_gamma(t, M)
            xt = xt - gamma * self.step
            
            self.x_trajectory = np.vstack((self.x_trajectory, xt))
            self.errors_trajectory = np.hstack((self.errors_trajectory, self.f(xt) - self.f(self.x_opt)))
            self.grad_norm_trajectory = np.hstack((self.grad_norm_trajectory, np.linalg.norm(self.oracle.df)))
            self.tk += 1
            if T - t == self.suffix:
                x_arr = xt
            if T - t < self.suffix:
                x_arr = np.vstack((x_arr, xt))
                
        if self.suffix > 1:
            xt = np.mean(x_arr, axis = 0)      
        return xt
    
    def SGDsc(self, T, x0):
        M = self.M0 + self.mu
        x1 = self.SGD( M, T, x0)        
        for k in range(self.K):    
            M = M * 2
            T = T * 2
            x2 = self.SGD( M, T, x1)
            x1 = x2       
        return x1
    
    def regularizer_grad(self, x):
        regularizer_grad = np.zeros(self.d)
        for i in range(self.s):
            regularizer_grad += self.mus[i] * (x - self.xs[i])
        return regularizer_grad
    
    def recursively_regularized_SGD(self):  
        self.xs = [self.x0]
        self.mus = [self.mu0]
        self.tk = 0
        x_last = self.x0
        for self.s in range(self.S):
            self.mu = self.mus[-1]   
            T = self.T
            x_last = self.SGDsc( T, x_last)
            self.xs.append(x_last)
            self.mus.append(self.mu * 2)
            
        plt.plot(self.x_trajectory[:,0], self.x_trajectory[:,1])
    
        return x_last

    def get_random_initial_point(self):
        x00 = np.ones(self.d) * 0.2
        x0 = x00 + np.random.uniform(low=0, high=1, size=self.d) * 0.1
        return x0
    
    def run_average_experiment(self):
        self.x0 = self.x00
        
        if self.random_init:
            self.x0 = self.get_random_initial_point()

        f_0 = self.f(self.x0)
        f_opt = self.f(self.x_opt)
        
        self.x_trajectory = np.array(self.x0)
        self.errors_trajectory = f_0 - f_opt
        self.oracle.sample(self.x0)
        self.grad_norm_trajectory = np.linalg.norm(self.oracle.df)
        
        x_last = self.recursively_regularized_SGD()

        self.x_total = [self.x_trajectory]
        self.errors_total = [self.errors_trajectory]
        self.grad_norm_total = [self.grad_norm_trajectory]
        
        for i in range(self.experiments_num - 1):
            self.x0 = self.x00
            if self.random_init:
                self.x0 = self.get_random_initial_point()
                f_0 = self.f(self.x0)
            
            self.x_trajectory = np.array(self.x0)
            self.errors_trajectory = f_0 - f_opt
            self.oracle.sample(self.x0)
            self.grad_norm_trajectory = np.linalg.norm(self.oracle.df)
            
            x_last = self.recursively_regularized_SGD()
            self.x_total.append(self.x_trajectory)
            self.errors_total.append(self.errors_trajectory)
            self.grad_norm_total.append(self.grad_norm_trajectory)
            
        print('Finished Optimizer')
        return x_last
    
#################################################################################

@dataclass
class SafeLogBarrierOptimizer:
    x0: np.array = None
    M0: float = None
    Ms: np.array = None
    sigma: float = None
    hat_sigma: float = None
    eta0: float = None
    eta: float = None
    step: np.array = None
    oracle: Oracle = None
    f: Callable = None
    h: Callable = None
    d: float = None
    m: float = None
    reg: float = None
    x_opt: float = None
    T: int = None
    S: int = None
    K: int = None
    experiments_num: int = None
    mu0: float = None
    mu: float = None
    mus: list = None
    xs: list = None
    s: int = None
    convex: bool = None
    random_init: bool = False
    no_break: bool = True
    errors_total: list = None
    constraints_total: list = None
    beta: float = None
        
    def compute_gamma(self, t) -> float:
        step_norm = np.linalg.norm(self.step)
        alphas = self.oracle.alphas
        dhs = self.oracle.constraints_grad
        
        alphas_reg = alphas
        L_dirs = np.zeros(self.m)
        for i in range(self.m):
            L_dirs[i] = np.abs((dhs[i].dot(self.step)) / step_norm) + 3 * self.hat_sigma 
            alphas_reg[i] = max(self.reg, alphas[i])

        M2 = self.M0 + 2 * self.eta * np.sum(self.Ms / alphas_reg) + 4 * self.eta * np.sum(L_dirs**2 / alphas_reg**2) 
        gamma = min(1. / step_norm * np.min(alphas / ( 2 * L_dirs +  alphas_reg**0.5 * self.Ms**0.5)), 
                        1. / (M2 + self.mu), 
                        1. /  (t + 1) / self.mu )
        return gamma

    def regularizer_grad(self, x):
        regularizer_grad = np.zeros(self.d)
        for i in range(self.s):
            regularizer_grad += self.mus[i] * (x - self.xs[i])
        return regularizer_grad
    
    def dB_estimator(self):
        
        alphas = self.oracle.alphas
        jacobian = self.oracle.constraints_grad
        df_e = self.oracle.objective_grad
        denominators = 1. / np.maximum(np.ones(self.m) * self.reg, alphas)
        dB = df_e + self.eta * jacobian.T.dot(denominators)
        return dB
    
    
    def barrier_SGD(self):
        
        x_last = self.x0
        self.xs = [self.x0]
        self.mus = [self.mu0]
        x_trajectory = np.array([x_prev])
        errors_trajectory = self.f(self.x0) - self.f(self.x_opt)
        constraints_trajectory = np.max(self.h(x_prev))
        worst_constraint = np.max(self.h(x_prev))
        Tk = 0
        
        for self.s in range(self.S):
            xt = x_last
            
            for t in range(self.T):
                self.mu = self.mus[-1]
#                 print(mus)
#                 regularizer_grad = np.zeros(self.d)
#                 for i in range(s):
#                     regularizer_grad += mus[i] * (xt - xs[i])
#                 regularizer_grad = mus[-1] * (xt - xs[-1])
#                 regularizer_grad += mu * (xt - x_prev)
                
                self.oracle.sample(xt)  
                self.step = self.dB_estimator() + self.regularizer_grad(xt)
                step_norm = np.linalg.norm(self.step)
                gamma = self.compute_gamma(t)
                
                if step_norm < self.eta and self.no_break == False:
                    break

                xt = xt - gamma * self.step
                Tk += 1
                
                x_trajectory = np.vstack((x_trajectory, xt))
                errors_trajectory = np.hstack((errors_trajectory, self.f(xt) - self.f(self.x_opt)))
                constraints_trajectory = np.hstack((constraints_trajectory, np.max(self.h(xt))))
                worst_constraint = max(worst_constraint, np.max(self.h(xt)))

            self.xs.append(xt)
            x_last = xt
            self.mus.append(self.mu * 2)
  
        plt.plot(x_trajectory[:,0], x_trajectory[:,1])
    
        return x_trajectory, errors_trajectory, constraints_trajectory, x_last, Tk
    
    
    def log_barrier_decaying_eta(self):
        f_opt = self.f(self.x_opt)
        x_long_trajectory = self.x0
        errors_long_trajectory = self.f(self.x0) - f_opt
        constraints_long_trajectory = np.max(self.h(self.x0))    
        T_total = 0
        
        eta = self.eta0
        x0 = self.x0
        x_prev = x0
        
        for k in range(self.K):
            if k != self.K - 1:
                mu0 = 0
            else:
                mu0 = self.mu0
                
            x_traj_k, errors_traj_k, constraints_traj_k, x_last_k, T_k = self.barrier_SGD()

            errors_long_trajectory = np.hstack((errors_long_trajectory, errors_traj_k))
            constraints_long_trajectory = np.hstack((constraints_long_trajectory, constraints_traj_k))
            x_long_trajectory = np.vstack((x_long_trajectory, x_traj_k))
            T_total = T_total + T_k
            self.x0 = x_last_k
            eta = eta * 0.5

        return x_long_trajectory, errors_long_trajectory, constraints_long_trajectory, T_total, x_last_k

    def get_random_initial_point(self):
        for i in range(1000 * self.d):
            x0 = np.array([0.1335990371483741, 0.2743781816448671, 
                           0.2879962344461537, 0.10242147970254536, 
                           0.3959197145814795, 0.5982863622683936]) + np.random.uniform(low=0, high=1, size=self.d) * 0.1
            if (self.h(x0) < - self.beta).all():
                break
        return x0
    
    def run_average_experiment(self):
        self.beta = self.eta0
        if self.random_init:
            self.x0 = self.get_random_initial_point()

        f_0 = self.f(self.x0)
        f_opt = self.f(self.x_opt)
        (x_long_trajectory, errors_long_trajectory, 
                            constraints_long_trajectory, 
                            T_total, 
                            x_last) = self.log_barrier_decaying_eta()
#         PlotTrajectory(x_long_trajectory, self.x0, self.x_opt, self.h)
        x_total = []
        errors_total = []
        constraints_total = []

        errors_total.append(errors_long_trajectory)
        constraints_total.append(constraints_long_trajectory)
        
        for i in range(self.experiments_num - 1):
            if self.random_init:
                self.x0 = self.get_random_initial_point()
                f_0 = self.f(self.x0)
            (x_long_trajectory, errors_long_trajectory, 
                                constraints_long_trajectory, 
                                T_total, 
                                x_last) = self.log_barrier_decaying_eta()

            x_total.append(x_long_trajectory)
            errors_total.append(errors_long_trajectory)
            constraints_total.append(constraints_long_trajectory)
            
        self.errors_total = errors_total
        self.constraints_total = constraints_total
        print('Finished')
        return x_last


##############################################################################################################