import numpy as np
from numpy import linalg
from scipy.optimize import linprog
from cvxopt import matrix, spmatrix
from cvxopt.solvers import options, qp, lp 
from scipy.stats import norm, chi2
import matplotlib.pyplot as plt

def f(x, x_0):
    return 0.5*np.linalg.norm(x - x_0)**2

def df(x, x_0):
    return x - x_0

def get_measurements(A, b, x, sigma):
    m = np.shape(A)[0]
    return A.dot(x) - b + np.random.normal(0, sigma**2, m)

#X - list of np.arrays, Y - list of np.arrays, x - np.array
def make_and_add_measurement(A, b, X, Y, x, sigma):
    X.append(x)
    Y.append(get_measurements(A, b, x, sigma))
    return X,Y

def estimate_constraints(X, Y):
    N = np.shape(X)[0]
    X_ext = np.vstack((np.array(X).T,-1*np.ones(N))).T
    Ab_est = np.linalg.inv(X_ext.T.dot(X_ext)).dot(X_ext.T).dot(Y)
    A_est = Ab_est[:-1].T
    b_est =  Ab_est[-1]
    return  A_est, b_est

def solve_DFS(A_est, b_est, grad):
    sol = linprog(grad, 
                                  A_ub = A_est, 
                                  b_ub = b_est, 
                                  bounds=((None, None)), 
                                  options={"disp": False})
    return sol.x, sol.status

def true_solution(x_0, A, b):
    d = np.size(x_0)
    P = np.eye(d)
    q = -1*x_0
    m = np.size(b)
    G = A
    h = b
    args = [matrix(P, tc = 'd'), matrix(q, tc = 'd'), matrix(G, tc = 'd'), matrix(h, tc = 'd')]
    sol = qp(*args)
    if 'optimal' not in sol['status']:
        return None
    plt.scatter(sol['x'][0],sol['x'][1], color = 'r')
    return sol['primal objective'] + 0.5*np.linalg.norm(x_0)**2, np.array(sol['x']).reshape((P.shape[1],))

def violation(x, A, b):
    m = np.shape(A)[0]
    viol = 0.
    delta = np.zeros(m)
    for i in range(m):
        delta[i] = b[i] - A[i].dot(x)
    delta_min = np.min(delta)
    if delta_min < 0:       
        return 1., -1*delta_min
    else:
        return 0., -1*delta_min
    
def SCFW(df, A, b, x_start, sigma, omega, gamma_init, n_0, T_max, eps, fopt, x_0, center, N = 0):
    X = []
    Y = []
    x = x_start
    e = []
    err = np.zeros(T_max)
    delta = np.zeros(T_max)
    x_vec_0 = np.zeros(T_max)
    x_vec_1 = np.zeros(T_max)
    d = np.size(x_start)
    phi = sigma*np.sqrt(chi2.ppf(1 - eps, d+1))
    omega_t = np.zeros(2*d)
    counter = 0
    viol = 0.
    nonconv = 0
    number_of_meas = 0.
    useless_steps = 0.
    gamma_seq = np.zeros(T_max)
    gamma_prev = np.zeros(T_max)
    
    for i in range(d):
        e.append(np.eye(d)[i])
        e.append(-np.eye(d)[i])
    
    for t in range(T_max):
        if t > 0:
            A_est_prev, b_est_prev, X_arr_prev = A_est, b_est, X_arr
        
        x_vec_0[t] = x[0]
        x_vec_1[t] = x[1]
        err[t] = f(x, x_0) - fopt
        n_t = d * int((t + 1) * np.log(t + 2)**2 * phi**2 * (1. / omega**2 + 1))#на d уже умножено в exp
        print(n_t, t + 1, d, 
              np.log(t + 2)**2,  
              phi**2, 2*(t + 2) * d * np.log(t + 3)**2 * phi**2 *(1. / omega + 1))

        for i in range(n_t):
            if center == 0:
                x_measure = x + omega*e[i%(2*d)]
            elif center == 1:
                x_measure = x_start + omega*e[i%(2*d)]
            
######    Plotting trajectories ############ 
#             if i < 2*d:
#                 plt.scatter(x_measure[0], x_measure[1], color = "green")
        
            make_and_add_measurement(A, b, X, Y, x_measure, sigma)
            number_of_meas = number_of_meas + 1.
        
        A_est, b_est = estimate_constraints(X, Y)
        sol_DFS = solve_DFS(A_est, b_est, df(x, x_0))
        s =  sol_DFS[0]

######    Plotting trajectories ############ 
        #plt.scatter(s[0], s[1], marker = "v", color = "magenta")
    
        X_arr = np.array(X)
        if t > 0:
            gamma_prev[t] = bound_gamma(A_est, b_est, X_arr, sigma, x_prev, s_prev, eps, omega_case = 0)[0]
            #print  "G_{t+1} - G_t", gamma_prev - gamma_max
        sol =  bound_gamma(A_est, b_est, X_arr, sigma, x, s, eps, omega_case = 0)
        gamma_max = sol[0]
        
        if sol_DFS[1] >= 2:
            gamma_max = 0
            
#         if gamma_max > 1:
#             print "t", t, "gamma_max", gamma_max
        gamma_seq[t] = gamma_max
        delta[t] = violation(x, A, b)[1]
        #gamma = min(1./(t+2 - useless_steps), gamma_max)
        gamma = 1./(t+2 - useless_steps)
        if gamma != 0:
            counter = counter + 1
        elif gamma == 0 and counter != 0:
            nonconv = 1
        #print counter
   
        #if (gamma_max == 0) and (t - useless_steps > 0):
        s_prev = s
        x_prev = x
        #sol_prev =  bound_gamma(A_est_prev, b_est_prev, X_arr_prev, sigma, x, s, eps)
        #gamma_max_prev = sol_prev[0]
        #gamma = min(1./(t+2 - useless_steps), gamma_max_prev)
        gamma = 1. / (t + 2) 
        if gamma_max == 0:    
            useless_steps = useless_steps + 1.
            
######    Plotting trajectories ############  
#         if d == 2:
#             for i in range(101):
#                 gamma_s[i] =  bound_gamma(A_est, b_est, X_arr, sigma, x, x_for_safety[i], eps, omega_case = 0)[0]
#                 g_safety[i] = (1 - gamma_s[i])*x + gamma_s[i]*x_for_safety[i]
#             plt.plot(g_safety.T[0], g_safety.T[1], color = "orange")
#
        x = (1 - gamma)*x + gamma*s
        viol = viol + violation(x, A, b)[0]
    
    ######    Plotting trajectories ############ 
    #plt.plot(x_vec_0, x_vec_1, color = "red")
    
    viol_ratio = viol/T_max
    print(number_of_meas, nonconv)
    return x, err, delta, viol_ratio, gamma_seq, nonconv, gamma_prev

def SCFW_reduced(df, A, b, 
                 x_start, sigma, omega, 
                 gamma_init, n_0, T_max, 
                 eps, fopt, x_0, center, N = 0):
    X = []
    Y = []
    x = x_start
    e = []
    err = np.zeros(T_max)
    delta = np.zeros(T_max)
    x_vec_0 = np.zeros(T_max)
    x_vec_1 = np.zeros(T_max)
    d = np.size(x_start)
    omega_t = np.zeros(2 * d)
    x_meas_0 = np.zeros(4 * T_max)
    x_meas_1 = np.zeros(4 * T_max)
    counter = 0
    viol = 0.
    nonconv = 0
    number_of_meas = 0.
    useless_steps = 0.
    gamma_seq = np.zeros(T_max)
    gamma_prev = np.zeros(T_max)
    x_for_safety_1 = np.ones((100,2))
    x_for_safety_1.T[1] = np.arange(-1,1,0.02)
    x_for_safety_2 = np.ones((100,2))
    x_for_safety_2.T[0] = -1 * np.arange(-1,1,0.02)
    x_for_safety_3 = -1 * np.ones((100,2))
    x_for_safety_3.T[1] = -1 * np.arange(-1,1,0.02)
    x_for_safety_4 = -1 * np.ones((100,2))
    x_for_safety_4.T[0] = np.arange(-1,1,0.02)
    x_for_safety = np.vstack((x_for_safety_1,x_for_safety_2, x_for_safety_3,x_for_safety_4,(1,-1)))
    g_safety = np.zeros((401,2))
    gamma_s = np.zeros(401)

    for i in range(d):
        e.append(np.eye(d)[i])
        e.append(-np.eye(d)[i])
    
    for t in range(T_max):
        x_vec_0[t] = x[0]
        x_vec_1[t] = x[1]
        err[t] = f(x, x_0) - fopt
        
        n_t = 2 * n_0 * (t + 1) * int(np.log(T_max) ** 2 * np.log(1. / eps))

        for i in range(n_t):
            if center == 0:
                x_measure = x + omega*e[i % (2 * d)]
            elif center == 1:
                x_measure = x_start + omega * e[i % (2 * d)]
            
            make_and_add_measurement(A, b, X, Y, x_measure, sigma)
            number_of_meas = number_of_meas + 1.
            
            if i<4:
                x_meas_0[4 * t + i] = x_measure[0]
                x_meas_1[4 * t + i] = x_measure[1]
            
            #check if more measurements needed
            if i > 2 * d and i % (2 * d)==0:
                A_est, b_est = estimate_constraints(X, Y)
                s_new = solve_DFS(A_est, b_est, df(x, x_0))[0]
                x_new = (1 - 1./(t+2))*x + 1./(t+2)*s_new
                #x_new_1 = x+ 1./(t+2)*(s_new - x)
                #print "diff", x_new - x_new_1
                if check_safety(A_est, b_est, np.array(X), sigma, x_new, eps) == 1:
                    #print i,"safe", "x_new", x_new, "gamma_max", bound_gamma(A_est, b_est, np.array(X), sigma, x, s_new, eps, omega_case = 0)[0]
                    break

        A_est, b_est = estimate_constraints(X, Y)
        sol_DFS = solve_DFS(A_est, b_est, df(x, x_0))
        s =  sol_DFS[0]

        plt.scatter(s[0], s[1], color = "magenta", marker = "v")
        X_arr = np.array(X)

         
        sol =  bound_gamma(A_est, b_est, X_arr, sigma, x, s, eps, omega_case = 0)
        gamma_max = sol[0]
        
        if sol_DFS[1] >= 2:
            gamma_max = 0
            
        gamma_seq[t] = gamma_max
        delta[t] = violation(x, A, b)[1]

        gamma = 1./ (t + 2)
        if gamma != 0:
            counter = counter + 1
        elif gamma == 0 and counter != 0:
            nonconv = 1
     
        if d == 2:
            for i in range(401):
                gamma_s[i] =  bound_gamma(A_est, b_est, X_arr, sigma, x, x_for_safety[i], eps, omega_case = 0)[0]
                g_safety[i] = (1 - gamma_s[i]) * x + gamma_s[i] * x_for_safety[i]

            plt.plot(g_safety.T[0], g_safety.T[1], color = "orange")
    
        x = (1 - gamma) * x + gamma * s
        viol = viol + violation(x, A, b)[0]
        
    plt.plot(x_vec_0, x_vec_1, color = "red")
        
    if d == 2:
        for t in range(T_max):
            for j in range(4):
                plt.scatter(x_meas_0[4 * t + j], x_meas_1[4 * t + j], color = "green")
    
    viol_ratio = viol/T_max
    print(number_of_meas, nonconv)
    return x, err, delta, viol_ratio, gamma_seq, nonconv, gamma_prev


def Exp(
    T_max,
    eps,
    omega,
    E, 
    Enum,
    d = 4,
    x_start=np.array([0.,0.,0.,0.]),
    bad_cond=0,
    int_cond=0,
    n0=20,
    sigma=0.1,
    rand_seed=42,
    method=SCFW,
    center=0,
    maxmin=0,
    storage=None,
    N = 0
    ):
    
    if storage is None:
        storage = []
    
    A = np.zeros((2*d, d))
    for i in range(d):
        A[i] = np.eye(d)[i]
        A[d + i] = -1.*np.eye(d)[i]
    b = np.ones(2*d)
    
    if bad_cond == 1:
        A[0] = np.ones(d)
        A[0,0] = 0.5
        A[d] = -np.ones(d)
        A[d,0] = -0.5
        b[0] = 0.5
        b[d] = 0.5
        
    if int_cond == 0:
        x_0 = 0.5 * np.ones(d) + 2 * np.eye(d)[0]
    elif int_cond == 1:
        x_0 = 0.5 * np.ones(d) + 0.2 * np.eye(d)[0]
    elif int_cond == 2:
        x_0 = -0.5 * np.ones(d) + -2 * np.eye(d)[0]
    elif int_cond == 3:
        x_0 = 2*np.ones(d)
        
    gamma_init = 1.
    n_0 = n0
    
    plt.figure(figsize=(15, 15))
    plt.xlim((-1.1, 1.5))
    plt.ylim((-1.2, 1.2))

    viol = np.zeros(Enum)
    nonconvergence = np.zeros(Enum)
    eps_v = np.zeros(Enum)
    avg_conv = np.zeros(Enum)
    
    sol_f = []
    sol_v = []
    sol_g = []

    for j in range(Enum):
        
        solution = []
        np.random.seed(rand_seed)
        true = true_solution(x_0, A, b)
        fopt = true[0]
        xopt = true[1]

        for i in range(E):
            solution.append(method(df, A, b, x_start, 
                                   sigma, omega, gamma_init, 
                                   n_0, T_max, eps, fopt, 
                                   x_0, center, N))
        
        plt.tick_params(axis='both', which='major', labelsize=40)
        plt.scatter(xopt[0],xopt[1], marker = "*", color = "red")
        filex = 'perda_perda_plot0'+str(j)+'.eps'
        plt.xlabel(r"$x_1$", fontsize=50)
        plt.ylabel(r"$x_2$", fontsize=50)
       #plot a box constraints
        if bad_cond == 0:
            plt.plot(np.array([-1.,1.,1.,-1.,-1.]),np.array([-1.,-1.,1.,1.,-1.]))
        plt.savefig(filex)
        plt.show()

        sol_f.append(np.zeros((E, T_max)))
        for i in range(E):
            sol_f[j][i] = solution[i][1]
        
        sol_v.append(np.zeros((E, T_max)))
        for i in range(E):
            sol_v[j][i] = np.array(solution[i][2])
            
        vio = 0.
        for i in range(E):
            if solution[i][3] > 0:
                vio = vio + 1.
        print("eps", eps, "vio", vio/E)
        viol[j] = vio/E
        
        non_conv = 0.
        conv = np.zeros(E)      
        nconv = np.zeros(E)
        for i in range(E):
            nconv[i] = solution[i][5]
            conv[i] = (f(solution[i][0], x_0) - fopt)/(f(solution[0][0], x_0) - fopt) 
            #if conv[i] >= f(x_start, x_0) - 1.:
            #    non_conv = non_conv + 1.
        avg_conv[j] = np.mean(conv)
        print("eps", eps, "non_conv", np.mean(nconv)) #non_conv/E
        nonconvergence[j] =  np.mean(nconv) #non_conv/E
        eps_v[j] = eps
        eps = eps/2.
        
        sol_g.append(np.zeros((E, T_max)))
        for i in range(E):
            sol_g[j][i] = np.array(solution[i][4])
        print("gamma")
    
    colors = ['green', 'magenta', 'blue', 'olive', 'orange', 'red' ]
        
    plt.figure(figsize=(15, 15))
    for j in range(Enum):
        mean_sol = sol_f[j].mean(axis = 0)
        std_sol = sol_f[j].std(axis = 0)
        min_sol = sol_f[j].min(axis = 0)
        max_sol = sol_f[j].max(axis = 0)
        storage.append((range(T_max), mean_sol, std_sol, 
                        min_sol, max_sol, '$\delta = $'+10*str(eps_v[j])))
        plt.plot(range(T_max), mean_sol / mean_sol.max(), color = colors[j])#, label='$\delta = $'+str(eps_v[j]))
        plt.fill_between(range(T_max), (mean_sol + std_sol) / mean_sol.max(), (mean_sol-std_sol) / mean_sol.max(), facecolor=colors[j], alpha=0.3)#, label = 'Standard deviation of convergence')
        if maxmin == 1:
            plt.fill_between(range(T_max), max_sol, min_sol, facecolor='yellow', alpha=0.5, label='Critical values')
    plt.legend(loc='best', fontsize=40)
    plt.tick_params(axis='both', which='major', labelsize=40)
    filex1 = 'perda_perda_plot1'+str(j)+'.eps'
    plt.xlabel(r"$t$", fontsize=50) 
    plt.ylabel(r"$\frac{f(x_t) - f_*}{f(x_0) - f_*}$", fontsize=50)
    plt.savefig(filex1)
    plt.show()
    storage.append([])
    return eps_v, viol, nonconvergence, avg_conv, storage

def check_safety(A_est, b_est, X_arr, sigma, x_new, eps):
    N = np.shape(X_arr)[0]
    m = np.shape(A_est)[0]
    #d = np.shape(A_est)[1]
    d = x_new.size
    phi = sigma*np.sqrt(chi2.ppf(1 - eps, d + 1))
    #phi = sigma*np.sqrt(chi2.ppf(eps, d+1))
    x_avg_N = X_arr.T.dot(np.ones(N))
    x_avg = x_avg_N * 1. / N
    R_inv = X_arr.T.dot(X_arr) - 1. / N * x_avg_N.reshape((d,1)).dot(x_avg_N.reshape((1,d)))
    def aRb(a,b):
        L = np.linalg.cholesky(R_inv)
        y = np.linalg.solve(L,a)
        z = np.linalg.solve(L,b)
        return y.dot(z)
    delta = np.zeros(m)
    for i in range(m):
        delta[i] = b_est[i] - A_est[i].dot(x_new)
    if phi * (1. / N + aRb(x_new - x_avg, x_new - x_avg))**0.5 < np.min(delta):
        safety = 1
    else:
        safety = 0
    return safety 

def bound_gamma(A_est, b_est, X_arr, sigma, x, s, eps, omega_case = 1):
    phi = sigma * np.sqrt(chi2.ppf(1 - eps, x.size + 1))
    N = np.shape(X_arr)[0]
    m = np.shape(A_est)[0]
    d = np.shape(A_est)[1]
    Delta = s - x
    x_avg_N = X_arr.T.dot(np.ones(N))
    x_avg = x_avg_N * 1. / N
    R_inv = X_arr.T.dot(X_arr) - 1. / N * x_avg_N.reshape((d,1)).dot(x_avg_N.reshape((1,d)))
    def aRb(a,b):
        L = np.linalg.cholesky(R_inv)
        y = np.linalg.solve(L,a)
        z = np.linalg.solve(L,b)
        return y.dot(z)
    delta = np.zeros(m)
    Gamma = np.zeros(m)
    case = np.zeros(m)
    
    for i in range(m):
        delta[i] = b_est[i] - A_est[i].dot(x)
        
        p = phi**2*aRb(Delta, Delta) - (A_est[i].dot(Delta))**2
        q =  2*(phi**2*aRb(Delta, x - x_avg) + delta[i]*A_est[i].dot(Delta))
        r = phi**2/N + phi**2*aRb(x - x_avg, x - x_avg) - delta[i]**2
        
        if p < 0 and q > 0 and r < 0:
            Gamma[i] = ((q**2 - 4 * p * r)**0.5 - q) / (2 * p)
            #print "case1", Gamma[i]
            
        elif p < 0 and q <= 0 and r < 0:
            Gamma[i] = 1.
            #print "case2", "1"
        elif p > 0 and r < 0:
             #max(
            Gamma[i] = ((q**2 - 4 * p * r)**0.5 - q) / (2 * p)
            case[i] = 2
            #print "case3",Gamma[i]
        else:
            Gamma[i] = 0.
            #print "case4", "0"
            
    gamma_max = np.min(Gamma)
    
    x_new  = x + gamma_max * Delta
    #print "gamma_max", gamma_max, "far",np.min(delta[i]),"safety", (phi**2/N + phi**2*aRb(x_new - x_avg, x_new - x_avg))**0.5 - np.min(b_est - A_est.dot(x_new))

    return gamma_max, np.min(delta)