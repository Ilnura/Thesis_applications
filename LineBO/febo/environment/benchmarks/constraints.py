import numpy as np

def ball(x):
    return 2*(np.linalg.norm(x)**2 - 1)

def my_ball(X):
    d = np.size(X)
    r = 0.5
    x0 = d**(-0.5) * np.ones(d)
    h = 0.2 * np.linalg.norm(X - x0)**2 - r**2 + 10. * (X[1] - x0[1])**2
    return h
    # r =  d**(-0.5)
    # x0 = d**(-0.5) * np.ones(d)
    # h = 1.2 * np.linalg.norm(X - x0)**2 - r**2 + 2. * (X[1] - x0[1])**2
    
    # r = d**(-0.5)
    # x0 = d**(-0.5) * np.ones(d)
    # h = np.linalg.norm(X - x0)**2 - r**2 + 2. * (X[1] - x0[1])**2
    # print("Perda")

def my_ball2(X):
    d = np.size(X)
    r = 10.
    x0 = d**(-0.5) * np.ones(d)
    h = 0.2 * np.linalg.norm(X - x0)**2 - r**2 + 10. * (X[1] - x0[1])**2
    # h = 1.2 * np.linalg.norm(X - x0)**2 - r**2 + 2. * (X[1] - x0[1])**2
    return h

def my_box0(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return A.dot(X)[0] - b[0]
def my_box1(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[1]
def my_box2(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[2]
def my_box3(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[3]
def my_box4(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[4]
def my_box5(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[5]
def my_box6(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[6]
    
def my_box7(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[7]

def my_quad_constraints1(X):
    d = np.size(X)
    h1 = np.linalg.norm(X,2)**2 - 1.
    return h1

def my_quad_constraints2(X):
    d = np.size(X)
    h2 = np.linalg.norm(X + 0.1 * np.ones(d),2)**2 - 2. * 0.1
    return h2

def my_quad_constraints1_big(X):
    d = np.size(X)
    h1 = np.linalg.norm(X,2)**2 - 10.
    return h1

def my_quad_constraints2_big(X):
    d = np.size(X)
    h2 = np.linalg.norm(X + 0.1 * np.ones(d),2)**2 - 2.
    return h2

def my_box0_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. * np.ones(2 * d) / d**0.5
    return A.dot(X)[0] - b[0]
def my_box1_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. *  np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[1]
def my_box2_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. *  np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[2]
def my_box3_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. * np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[3]
def my_box4_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. * np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[4]
def my_box5_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. * np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[5]
def my_box6_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. * np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[6]
def my_box7_big(X):
    d = np.size(X)
    A = np.vstack((np.eye(d),-np.eye(d)))
    b = 10. *  np.ones(2 * d) / d**0.5
    return (A.dot(X) - b)[7]