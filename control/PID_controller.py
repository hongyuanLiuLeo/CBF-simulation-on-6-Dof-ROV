import numpy as np
import math
from cvxopt import matrix
from cvxopt import solvers
import casadi as ca
import casadi.tools as ca_tools


import os
import sys
#sys.path.insert(0,os.path.abspath(".."))

class ECBF_control():
    def __init__(self,system,obs,safe_dist=0.5):
        self._safe_dist = safe_dist
        # self._obs = obs
        # self._obs_v = obs_v
        self.state = system._state._x
        self.x = self.state[0]
        self.y = self.state[1]
        self.v = self.state[2]
        self.w = self.state[3]
        self.K = 1

        self._A = np.empty((0,2))
        self._b = np.empty((0,))

        self._segments=[]
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

        # print(self._A)
        # print(self._b)
        # print(self._segments)
        # a = 1
    
    def field_of_view(self):
        return self.K*(100 - (np.power(self.x,2)+np.power(self.y,2)) - self._safe_dist)
        
    def field_of_view_d(self):
        return np.array([[2*self.x,2*self.y]])   
    
    def compute_edge_idx(self):

        pos = np.array([self.x,self.y])
        
        constraints = np.dot(self._A,pos) - self._b
        row_norms = np.sqrt(np.sum(self._A**2, axis=1))
        row_norms = row_norms.reshape(-1, 1)
        constraints = constraints / row_norms.flatten()
        # print(row_norms.flatten())
        # print(constraints)
        # a = 1

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
        # postive_index = np.where(constraints > 0)[0]
        # postive_values = constraints[postive_index]
        # max_positive_values = postive_values.max()
        # max_positive_index = np.where(constraints == max_positive_values)[0][0]
        # print(max_positive_indices)
        # a = 1

        return np.array(max_positive_indices)

    def compute_h(self):

        # return np.power(self.x-self._obs[0],2)+np.power(self.y-self._obs[1],2) - self._safe_dist
        # b = np.empty((0,))
        # for index,ob in enumerate(obs):
        #     b = np.concatenate((b,ob.vec_b))

        pos = np.array([self.x,self.y])

        # print(self._b[self._idx])
        # print(np.dot(self._A[self._idx].reshape(-1,2),pos))
        # print(np.dot(self._A[self._idx].reshape(-1,2),pos) - self._b[self._idx] - self._safe_dist)
        # a = 1
        return np.dot(self._A[self._idx].reshape(-1,2),pos) - self._b[self._idx] - self._safe_dist
    
    def compute_A(self):
        # A = -1*np.array([[2*(self.x-self._obs[0]),2*(self.y-self._obs[1])]])
        self._idx = self.compute_edge_idx()
        # print(-1*self._A[self._idx].reshape(-1,2))
        # a = 1
        return -1*self._A[self._idx].reshape(-1,2)
    
    def compute_b(self):
        
        b_ineq = self.K*self.compute_h()
        return b_ineq
    
    def compute_safe_control(self,u_norm):
        
        xd_norm = u_norm[0]
        yd_norm = u_norm[1]
        
        P = np.eye(2)
        q = -2*np.array([xd_norm,yd_norm])
        A = self.compute_A()
        b = self.compute_b()
        fov_A = self.field_of_view_d()
        fov_b = self.field_of_view()
        A = np.vstack((A,fov_A))
        b = np.append(b,fov_b)

        # min_abs_index = np.abs(b).argmin()
        # A = A[min_abs_index].reshape(1,-1)
        # b = [b[min_abs_index]]
        # print(fov_A)
        # print(fov_b)
        # print(A)
        # print(b)
        # c = 1
        opti_sol = solve_qp(P,q,A,b).flatten()
        # print(opti_sol)
        # d = 1
        
        #opti_u = np.array([math.sqrt(opti_sol[0]**2+opti_sol[1]**2),np.arctan2(opti_sol[1],opti_sol[0])])
        
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


class PIDController:
    def __init__(self):
        pass

    def generate_control_input(self,system,local_trajectory,obs,dt=0.1):

        self._ECBF = ECBF_control(system,obs,0.5)
        des_v,des_wd = self.pi_control(system,local_trajectory)
        u_norm = np.array([des_v,des_wd])
        # print(u_norm)
        
        w_curr = system._state._x[3]
        des_w = w_curr + des_wd*dt

        # print(des_w)

        a = 1
        x_norm = np.array([des_v*np.cos(des_w),des_v*np.sin(des_w)])
        # print("x_norm")
        # print(x_norm)
        
        x_opti = self._ECBF.compute_safe_control(x_norm)
        # print("x_opti")
        # print(x_opti)
        # b = 1
        v_opti = np.linalg.norm(x_opti)
        cp = np.cross([np.cos(w_curr),np.sin(w_curr)],x_opti)
        sgn = np.sign(cp)
        value = (x_opti[0]*np.cos(w_curr)+x_opti[1]*np.sin(w_curr))/v_opti
        value = np.clip(value, -1.0, 1.0)
        #wd_opti = np.arctan2(x_opti[1],x_opti[0]) - w_curr
        #v_opti= np.clip(v_opti, -1, 1)

        wd_opti = sgn*np.arccos(value)

        u_opti = np.array([v_opti,wd_opti])

        # print("u_opti")
        # print(u_opti)
        # c = 1
        return u_opti
        
    def pi_control(self,system,local_trajectory):
        
        xd = local_trajectory[0]
        yd = local_trajectory[1]

        x = system._state._x[0]
        y = system._state._x[1]
        w = system._state._x[3]

        Pv = 0.15
        Pw = 2.2
        xe = xd - x
        ye = yd - y
        D = math.sqrt(xe**2+ye**2)
        
        cp = np.cross([np.cos(w),np.sin(w)],[xe,ye])
        sgn = np.sign(cp)
        value = (xe*np.cos(w)+ye*np.sin(w))/D
        value = np.clip(value, -1.0, 1.0)
        phe = sgn*np.arccos(value)
        
        
        des_v = Pv*D        
        des_wd = Pw*phe
        
        return des_v,des_wd

