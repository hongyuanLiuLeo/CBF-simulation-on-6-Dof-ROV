import numpy as np
import math
from cvxopt import matrix
from cvxopt import solvers
import casadi as ca
import casadi.tools as ca_tools

from models.auv import AUV_Dynamics
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class ECBF_control():
    def __init__(self,system,obs,safe_dist=0.12):
        self._safe_dist = safe_dist
        self._state = system._state._x['pos']
        self._x = self._state[0]
        self._y = self._state[1]
        self._z = self._state[2]
        
        self._K = 1

        self._A = np.empty((0,3))
        self._b = np.empty((0,))

        self._segments = []
        current_row_start = 0

        for index,ob in enumerate(obs):
            self._A = np.vstack((self._A,ob.mat_A))
            self._b = np.concatenate((self._b,ob.vec_b))

            num_rows = ob.mat_A.shape[0]
            row_start = current_row_start
            row_end = current_row_start + num_rows - 1
            self._segments.append([row_start, row_end, index])
            current_row_start = row_end + 1
                
        self._segments = np.array(self._segments)

    def compute_edge_idx(self):

        pos = np.array([self._x,self._y,self._z])
        
        constraints = np.dot(self._A,pos) - self._b
        row_norms = np.sqrt(np.sum(self._A**2, axis=1))
        row_norms = row_norms.reshape(-1, 1)
        constraints = constraints / row_norms.flatten()

        max_positive_indices = []

        for seg in self._segments:
            start, end, _ = seg
            segment_constraints = constraints[start:end+1]
            positive_indices = np.where(segment_constraints > 0)[0]

            positive_values = segment_constraints[positive_indices]
            max_positive_value = positive_values.max()
            max_positive_index = np.where(segment_constraints == max_positive_value)[0][0]

            max_positive_index = start + max_positive_index
            
            max_positive_indices.append(max_positive_index)

        return np.array(max_positive_indices)
    
    def compute_h(self):
        pos = np.array([self._x,self._y,self._z])
        return np.dot(self._A[self._idx].reshape(-1,3),pos) - self._b[self._idx] - self._safe_dist
    
    def compute_A(self):
        self._idx = self.compute_edge_idx()
        return -1*self._A[self._idx].reshape(-1,3)
    
    def compute_b(self):
        
        b_ineq = self._K*self.compute_h()
        return b_ineq
    
    def compute_safe_control(self,u_norm):
        
        xd_norm = u_norm[0]
        yd_norm = u_norm[1]
        zd_norm = u_norm[2]
        
        P = np.eye(3)
        q = -2*np.array([xd_norm,yd_norm,zd_norm])
        A = self.compute_A()
        b = self.compute_b()

        opti_sol = solve_qp(P,q,A,b).flatten()

        return opti_sol



def solve_qp(P,q,G,h):
# Custom wrapper cvxopt.solvers.qp
# Takes in numpy array Converts to matrix double
    P = matrix(P,tc='d')
    q = matrix(q,tc='d')
    G = matrix(G,tc='d')
    h = matrix(h,tc='d')
    solvers.options['show_progress'] = False
    Sol = solvers.qp(P,q,G,h)
    
    return np.array(Sol['x'])



class PID:
    def __init__(self):
        self.previous_error = np.zeros(6, dtype=np.float64)
        self.integral = np.zeros(6, dtype=np.float64)

    def generate_control_input(self,system,trajectory,obs=None,dt=0.02):
        self._ECBF = ECBF_control(system,obs,0.12)
        xe = self.pi_control(system, trajectory,dt)
        # thrust_input = self.thrust_allocation(xe)

        x_opti = self._ECBF.compute_safe_control(xe[0:3])


        return np.hstack((x_opti,np.zeros(3)))
    
    def pi_control(self,system,trajectory,dt):
        x = system._state._x['pos'] # in n frame
        xd = trajectory

        xe = xd - x

        xe_b = system._state.J_inv() @ xe # in b frame


        self.derivative = (xe_b - self.previous_error)/dt

        self.integral += xe_b * dt

        # kp, ki, kd
        kp = np.diag([2,2,1.5,2.5,2.5,0.1])
        ki = np.diag([0.002,0.002,0.0015,0.001,0.001,0.001])

        # F = np.array([surge,sway,heave,roll,pitch,yaw])

        F = kp @ xe_b + ki @ self.integral

        
        self.previous_error = xe_b

        return F
    
    # def thrust_allocation(self,xe):
    #     x_aver = xe/4.0

    #     F1 = np.array([math.sqrt(2)/2,-math.sqrt(2)/2,0])
    #     F2 = np.array([math.sqrt(2)/2,math.sqrt(2)/2,0])
    #     F3 = np.array([-math.sqrt(2)/2,-math.sqrt(2)/2,0])
    #     F4 = np.array([-math.sqrt(2)/2,math.sqrt(2)/2,0])
    #     F5 = np.array([0,0,1])

    #     F1 = np.minimum(np.dot(x_aver, F1) * F1, 50)
    #     F2 = np.minimum(np.dot(x_aver, F2) * F2, 50)
    #     F3 = np.minimum(np.dot(x_aver, F3) * F3, 50)
    #     F4 = np.minimum(np.dot(x_aver, F4) * F4, 50)
    #     F5 = np.minimum(np.dot(xe, F5) * F5, 50)

    #     l1 = np.array([0.156,0.111,0.0])
    #     l2 = np.array([0.156,-0.111,0.0])
    #     l3 = np.array([-0.156,0.111,0.0])
    #     l4 = np.array([-0.156,-0.111,0.0])
        
    #     tau1 = np.cross(l1,F1)
    #     tau2 = np.cross(l2,F2)
    #     tau3 = np.cross(l3,F3)
    #     tau4 = np.cross(l4,F4)

    #     F_all = F1+F2+F3+F4+F5
    #     tau_all = tau1+tau2+tau3+tau4

    #     return np.concatenate((F_all,tau_all),axis=0)

    







