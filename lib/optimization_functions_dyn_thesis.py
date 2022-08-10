import numpy as np


def df_z(u, f, args, nu, loop, dyn, n_k):
    '''
        Finite differences approximation for gradient of f
    '''
    df = np.zeros((u.shape))
    for n in range(n_k):
        e = np.random.normal(0,1,u.shape)
        e_norm = np.linalg.norm(e)
        s =  1. / e_norm * e
        df += u.size * (f(u + nu * s , args, loop, dyn) - f(u, args, loop, dyn)) * s  / nu / n_k 
        
    return df


def B(u, g, args):
    '''
        log barrier
    '''
    return -np.log(-g(u, args)) 


def dB(u, g, df_z, gmin, args, nu, loop, dyn, n_k):
    '''
        log barrier gradient estimator
    '''
    db = np.zeros_like(u)
    db =  db + df_z(u, g, args, nu, loop, dyn, n_k) / gmin
    return db


def g_up(de, val1, val2):
    '''
        Upper confidence bound on g
    '''
    return min(val1 + 0.1 * (np.log(1. / de))**0.5 * sigma / 2., min(val2, 0.) / 2.) 


def g_obstacle(U, R, args, loop, dyn):
    '''
        constraint function with ball obstacle
    '''
    g = []
    x_0, x_B, x_C, A, B_m, N, dt = args
    x = dyn(U, args, loop)
    xC_spatial = np.array([x_C[0], x_C[1]])
    for n in range(N-1):
        x_spatial = np.array([x[n+1][0], x[n+1][1]])
#         g = min(g, 
        g.append(R - np.linalg.norm(x_spatial - xC_spatial))
    return g

def g_u_constraints(U, args, loop, dyn):
    max_force = 20.
    max_torque = 1.5
    x_0, x_B, x_C, A, B_m, N, dt = args
    x = dyn(U, args, loop)
#     if  len(np.shape(B_m)) == 1:
#         p = 1
#     else:
#         p = np.shape(B_m)[1]
    p = 2
    g_u_c = []
    if loop == 'open':
        u = U
    elif loop == 'closed_linear':
        u = np.zeros((N-1,p))
        for n in range(N-1):
            u[n] = U.dot(x[n] - x_B)
            g_u_c.append([u[n][0] - max_force,
                          -u[n][0] - max_force,
                          u[n][1] - max_torque,
                          -u[n][1] - max_torque])
    return np.array(g_u_c)*0.5
    

def g_round_wall(U, R, args, loop, dyn):
    '''
        constraint function with round wall
    '''
    x_0, x_B, x_C, A, B_m, N, dt = args
    x = dyn(U, args, loop)
    xC_spatial = np.array([x_C[0], x_C[1]])
    g_round = []
    for n in range(N-1):
        x_spatial = np.array([x[n+1][0], x[n+1][1]])
        g_round.append((np.linalg.norm(x_spatial - xC_spatial)**2 - R**2) / R**2)
    return g_round

def g_straight_walls(U, Aw, bw, args, loop, dyn):
    '''
        constraint function with straight polytopic walls
    '''
    g_walls = []
    x_0, x_B, x_C, A, B_m, N, dt = args
    x = dyn(U, args, loop)
    
    for n in range(N-1):
        x_spatial = np.array([x[n+1][0], x[n+1][1]])
        g_walls.append(0.2 * (Aw.dot(x_spatial) - bw))
    g_walls = np.array(g_walls)
    return g_walls
    
    
def f(U, args, loop, dyn):
    assert loop in ['open', 'closed_linear']
    x_0, x_B, x_C, A, B_m, N, dt = args
    
    if  len(np.shape(B_m)) == 1:
        p = 1
    else:
        p = np.shape(B_m)[1]

    x = dyn(U, args, loop)
    
    if loop == 'open':
        u = U
    elif loop == 'closed_linear':
        u = np.zeros((N - 1,p))
        for n in range(N-1):
            u[n] = U[0:1].dot(x[n] - U[2])

    f = 0.
    rho = 1
    xB_spatial = np.array([x_B[0],x_B[1]])
    
    for n in range(N-1):
        x_spatial = np.array([x[n+1][0], x[n+1][1]])
        f +=  np.linalg.norm(x_spatial - xB_spatial)**2 / 4. / N + np.linalg.norm(u[n])**2 /50./N
#         f +=  rho * ft
#         rho = n + 1
#         rho = rho * 0.99
#     f += 10 * np.linalg.norm(x_spatial - xB_spatial)**2 #np.linalg.norm(x[n] - x_B)**2
#     f = 0
#     f +=  np.linalg.norm(x_spatial - xB_spatial)**2 
    return f