import numpy as np

def dyn_linear(U, args, loop):
    assert loop in ['open', 'closed_linear']
    
    x_0, x_B, x_C, A, B_m, N, dt = args
  
    f = 0.
    d = np.size(x_0)
    if  len(np.shape(B_m)) == 1:
        p = 1
    else:
        p = np.shape(B_m)[1]
    x = np.zeros((N, d))
    x[0] = x_0
    
    if loop == 'open':
        u = U
    elif loop == 'closed_linear':
        u = np.zeros((N, p))
        for n in range(N):
            u[n] = U.dot(x[n] - x_B)
    
    for n in range(N - 1):
        if u[n].size == 1:
            x[n + 1] = A.dot(x[n]) + B_m * u[n]
#             print(A, x[n], A.dot(x[n]))
        else:
            x[n + 1] = A.dot(x[n]) + B_m.dot(u[n])
    return x


def dyn_unicycle(U, args, loop):
    assert loop in ['open', 'closed_linear']
    
    x_0, x_B, x_C, _, _, N, dt = args
#     if  len(np.shape(B_m)) == 1:
#         p = 1
#     else:
#         p = np.shape(B_m)[1]
    p = 2
    f = 0.
    x = np.zeros((N,3))
    x[0] = x_0
    if loop == 'open':
        u = U
    elif loop == 'closed_linear':
        u = np.zeros((N-1,p))
        for n in range(N-1):
            u[n] = U[0:1].dot(x[n]- U[2]) 
    for n in range(N-1):
        x[n + 1,0] = x[n,0] + 2. * u[n,0] * fgamma(u[n,1], dt) * np.cos(x[n,2] + dt * u[n,1] / 2.)
        x[n + 1,1] = x[n,1] + 2. * u[n,0] * fgamma(u[n,1], dt) * np.sin(x[n,2] + dt * u[n,1] / 2.)
        x[n + 1,2] = x[n,2] +  dt * u[n,1]
    return x


def dyn_unicycle_euler(U, args, loop):
    assert loop in ['open', 'closed_linear']
    
    x_0, x_B, x_C, _, _, N, dt = args
    p = 2
    f = 0.
    x = np.zeros((N,3))
    x[0] = x_0
    if loop == 'open':
        u = U
    elif loop == 'closed_linear':
        u = np.zeros((N-1,p))
        for n in range(N-1):
            u[n] = U[0:1].dot(x[n] - U[2])
    for n in range(N-1):
        x[n + 1,0] = x[n,0] + dt * u[n,0] * np.cos(x[n,2])
        x[n + 1,1] = x[n,1] + dt * u[n,0] * np.sin(x[n,2])
        x[n + 1,2] = x[n,2] +  dt * u[n,1]
    return x


def fgamma(omega, dt):
    if omega != 0:
        return np.sin(omega * dt / 2.)
    else:
        return dt / 2.
    
    
def dyn_heli_linearized(U, args_heli = (0.1,-0.5,-0.5,0,-5,2.,2.1,11,18), loop = 'closed_linear', x_B = np.zeros(8), x_0 = np.zeros(8), N = 10):
    assert loop in ['open', 'closed_linear']
    ts, kx, ky, kz, kp, bx, by, bz, bp = args_heli
    x = np.zeros((N,8))
    u = np.zeros((N,4))
    A11 = np.eye(4)
    A12 = ts * np.eye(4)
    A21 = np.zeros((4,4))
    A22 = np.eye(4) + ts * np.diag((kx,ky,kz,kp))
    B1 = np.zeros(4)
    B2 = np.diag((bx,by,bz,bp))
    x[0] = x_0
    for n in range(N - 1):
        if loop == 'open':
            u[n] = U[n]
        elif loop == 'closed_linear':
            u[n] = U.dot(x[n] - x_B)

        x[n + 1][0:4] = A11.dot(x[n][0:4]) +  A12.dot(x[n][4:8]) + B1.dot(u[n]) 
        x[n + 1][4:8] = A21.dot(x[n][0:4]) +  A22.dot(x[n][4:8]) + B2.dot(u[n]) 
        print(x)
    return x

