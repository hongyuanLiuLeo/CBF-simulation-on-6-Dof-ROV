import numpy as np
import math
from cvxopt import matrix
from cvxopt import solvers

from cvxopt import matrix, solvers
import numpy as np

def solve_qp(P, q, G, h):
    # 自定义 cvxopt.solvers.qp 的包装函数
    P = matrix(P, tc='d')
    q = matrix(q, tc='d')
    G = matrix(G, tc='d')
    h = matrix(h, tc='d')
    solvers.options['show_progress'] = False
    Sol = solvers.qp(P, q, G, h)
    
    return np.array(Sol['x'])

def compute_safe_control(u_norm):
    xd_norm = u_norm[0]*np.cos(u_norm[1])
    yd_norm = u_norm[0]*np.sin(u_norm[1])

    P = np.eye(2)
    q = -2*np.array([xd_norm,yd_norm])
    A = np.array([[ 0.70711,0.70711]])
    b = [ 1.41421]     

    opti_sol = solve_qp(P,q,A,b).flatten()
    
    
    opti_u = np.array([math.sqrt(opti_sol[0]**2+opti_sol[1]**2),np.arctan2(opti_sol[1],opti_sol[0])])
    return opti_u

# 示例使用
u_norm = np.array([0.1, 0.1])
result = compute_safe_control(u_norm)
print(result)
