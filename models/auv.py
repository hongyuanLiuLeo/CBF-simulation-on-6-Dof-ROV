import datetime
import numpy as np
import matplotlib.patches as patches
import math
import casadi as ca
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from geometry_utils import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.auv_simulation import *

class AUV_Dynamics:

    # Rigid-body inertia matrix (M_RB)
    @staticmethod
    def M_RB():
        m = 13.5 # Mass of the ROV (kg)
        Ix,Iy,Iz = 0.26,0.23,0.37 # Moment of inertia arount x,y,z

        M_RB = np.array([[m,0,0,0,0,0],
                      [0,m,0,0,0,0],
                      [0,0,m,0,0,0],
                      [0,0,0,Ix,0,0],
                      [0,0,0,0,Iy,0],
                      [0,0,0,0,0,Iz]])
        return M_RB
    
    # Added mass matrix (M_A)
    @staticmethod
    def M_A():
        X_udot = -6.63 # Added mass in surge direction (kg)
        Y_vdot = -7.12 # Added mass in sway direction (kg)
        Z_wdot = -18.68 # Added mass in heave direction (kg)
        K_pdot = -0.189 # Added mass in roll (kg*m^2)
        M_qdot = -0.135 # Added mass in pitch (kg*m^2)
        N_rdot = -0.222 # Added mass in yaw (kg*m^2)

        M_A = -np.array([[X_udot,0,0,0,0,0],
                         [0,Y_vdot,0,0,0,0],
                         [0,0,Z_wdot,0,0,0],
                         [0,0,0,K_pdot,0,0],
                         [0,0,0,0,M_qdot,0],
                         [0,0,0,0,0,N_rdot]])
        return M_A

    @staticmethod
    def C_v(x):
        m = 11.5
        Ix,Iy,Iz = 0.16,0.16,0.16
        u = x[0]
        v = x[1]
        w = x[2]
        p = x[3]
        q = x[4]
        r = x[5]

        # C_v = np.array([[0,-m*r,m*q,0,0,0],
        #                 [m*r,0,-m*p,0,0,0],
        #                 [-m*q,m*p,0,0,0,0],
        #                 [0,0,0,0,Iz*r,-Iy*q],
        #                 [0,0,0,-Iz*r,0,Ix*p],
        #                 [0,0,0,Iy*q,-Ix*p,0]])

        C_v = np.array([[0,0,0,0,m*w,-m*v],
                        [0,0,0,-m*w,0,m*u],
                        [0,0,0,m*v,-m*u,0],
                        [0,m*w,-m*v,0,Iz*r,-Iy*q],
                        [-m*w,0,m*u,-Iz*r,0,Ix*p],
                        [m*v,-m*u,0,Iy*q,-Ix*p,0]])
        
        return C_v
    
    @staticmethod
    def C_A(x):
        X_udot = -5.5
        Y_vdot = -12.7
        Z_wdot = -14.57
        K_pdot = -0.12
        M_qdot = -0.12
        N_rdot = -0.12

        u = x[0]
        v = x[1]
        w = x[2]
        p = x[3]
        q = x[4]
        r = x[5]

        C_A = np.array([[0,0,0,0,-Z_wdot*w,Y_vdot*v],
                        [0,0,0,Z_wdot*w,0,-X_udot*u],
                        [0,0,0,-Y_vdot*v,X_udot*u,0],
                        [0,-Z_wdot*w,Y_vdot*v,0,-N_rdot*r,M_qdot*q],
                        [Z_wdot*w,0,-X_udot*u,N_rdot*r,0,-K_pdot*p],
                        [-Y_vdot*v,X_udot*u,0,-M_qdot*q,K_pdot*p,0]])
        return C_A
    
    @staticmethod
    def D():
        X_u = -4.03
        Y_v = -6.22
        Z_w = -5.18
        Kp = -0.07
        Mq = -0.07
        Nr = -0.07

        D = -np.array([[X_u,0,0,0,0,0],
                       [0,Y_v,0,0,0,0],
                       [0,0,Z_w,0,0,0],
                       [0,0,0,Kp,0,0],
                       [0,0,0,0,Mq,0],
                       [0,0,0,0,0,Nr]])
        
        return D
    
    @staticmethod
    def D_n(x):
        X_uu = -18.18
        Y_vv = -21.66
        Z_ww = -36.99
        K_pp = -1.55
        M_qq = -1.55
        N_rr = -1.55

        u_abs = abs(x[0])
        v_abs = abs(x[1])
        w_abs = abs(x[2])
        p_abs = abs(x[3])
        q_abs = abs(x[4])
        r_abs = abs(x[5])

        D_n = -np.array([[X_uu*u_abs,0,0,0,0,0],
                         [0,Y_vv*v_abs,0,0,0,0],
                         [0,0,Z_ww*w_abs,0,0,0],
                         [0,0,0,K_pp*p_abs,0,0],
                         [0,0,0,0,M_qq*q_abs,0],
                         [0,0,0,0,0,N_rr*r_abs]])
        return D_n
    
    
    @staticmethod
    def g(eta):
        W = 112.8
        B = 114.8
        x = eta[0]
        y = eta[1]
        z = eta[2]
        phi = eta[3]
        theta = eta[4]
        psi = eta[5]
        zg = 0.02

        g = np.array([0,0,0,zg*W*np.cos(theta)*np.sin(phi),zg*W*np.sin(theta),0])

        return g



    # the systen is continuous differential system
    @staticmethod
    def forward_dynamics(t, state, tau):
        """Return updated state in a form of `np.ndnumpy`"""

        ''' 
        state:[12x1] state vector [u, v, w, p, q, r, x, y, z, phi, theta, psi] 
        position: state[-6:] : [x, y, z, phi, theta, psi] in n frame
        velocity: state[:6] : [u, v, w, p, q, r] in b frame

        M * x_dot + C * x = tau
        
        '''
        
        eta = state[-6:]
        nu = state[:6]
        u,v,w,p,q,r = nu[0],nu[1],nu[2],nu[3],nu[4],nu[5]
        u_abs,v_abs,w_abs,p_abs,q_abs,r_abs = abs(u),abs(v),abs(w),abs(p),abs(q),abs(r)
        phi,theta,psi = eta[3],eta[4],eta[5]

        g = 9.82 # Gravitational acceleration (m/s^2)
        m = 13.5 # Mass of the ROV (kg)
        Ix,Iy,Iz = 0.26,0.23,0.37 # Moment of inertia arount x,y,z

        rou = 1000   # Water density (kg/m^3)
        delta = 0.0135
        W = m*g
        B = rou * g * delta

        X_udot = -6.63 # Added mass in surge direction (kg)
        Y_vdot = -7.12 # Added mass in sway direction (kg)
        Z_wdot = -18.68 # Added mass in heave direction (kg)
        K_pdot = -0.189 # Added mass in roll (kg*m^2)
        M_qdot = -0.135 # Added mass in pitch (kg*m^2)
        N_rdot = -0.222 # Added mass in yaw (kg*m^2)

        X_u = -13.7
        X_uu = -141.0
        Y_v = -0.0
        Y_vv = -217.0
        Z_w = -33.0
        Z_ww = -190.0
        K_pp = -1.19
        K_p = -0.0
        M_qq = -0.47
        M_q = -0.8
        N_rr = -1.5
        N_r = -0.0

        x_b = 0
        y_b = 0
        z_b = -0.01

        M_RB = np.array([[m,0,0,0,0,0],
                      [0,m,0,0,0,0],
                      [0,0,m,0,0,0],
                      [0,0,0,Ix,0,0],
                      [0,0,0,0,Iy,0],
                      [0,0,0,0,0,Iz]])
        
        M_A = -np.array([[X_udot,0,0,0,0,0],
                         [0,Y_vdot,0,0,0,0],
                         [0,0,Z_wdot,0,0,0],
                         [0,0,0,K_pdot,0,0],
                         [0,0,0,0,M_qdot,0],
                         [0,0,0,0,0,N_rdot]])
        
        # Total mass matrix (M = M_RB + M_A)
        M = M_RB + M_A

        D_l = -np.array([[X_u,0,0,0,0,0],
                         [0,Y_v,0,0,0,0],
                         [0,0,Z_w,0,0,0],
                         [0,0,0,K_p,0,0],
                         [0,0,0,0,M_q,0],
                         [0,0,0,0,0,N_r]])
        
        # Quadratic damping matrix (proportional to square of velocity)
        D_q = -np.array([[X_uu*u_abs,0,0,0,0,0],
                         [0,Y_vv*v_abs,0,0,0,0],
                         [0,0,Z_ww*w_abs,0,0,0],
                         [0,0,0,K_pp*p_abs,0,0],
                         [0,0,0,0,M_qq*q_abs,0],
                         [0,0,0,0,0,N_rr*r_abs]])
        # Total damping matrix
        D = D_l + D_q

        # Compute Rigid-body Coriolis matrix (C_RB)
        C_RB = np.array([[0,0,0,0,m*w,-m*v],
                        [0,0,0,-m*w,0,m*u],
                        [0,0,0,m*v,-m*u,0],
                        [0,m*w,-m*v,0,Iz*r,-Iy*q],
                        [-m*w,0,m*u,-Iz*r,0,Ix*p],
                        [m*v,-m*u,0,Iy*q,-Ix*p,0]])
        
        # Compute Added-mass Coriolis matrix (C_A)
        C_A = np.array([[0,0,0,0,-Z_wdot*w,Y_vdot*v],
                        [0,0,0,Z_wdot*w,0,-X_udot*u],
                        [0,0,0,-Y_vdot*v,X_udot*u,0],
                        [0,-Z_wdot*w,Y_vdot*v,0,-N_rdot*r,M_qdot*q],
                        [Z_wdot*w,0,-X_udot*u,N_rdot*r,0,-K_pdot*p],
                        [-Y_vdot*v,X_udot*u,0,-M_qdot*q,K_pdot*p,0]])
        
        # Return the total Coriolis matrix
        C = C_RB + C_A

        g_eta = np.array([(W-B)*np.sin(theta),
                          -(W-B)*np.cos(theta)*np.sin(phi),
                          -(W-B)*np.cos(theta)*np.cos(phi),
                          y_b * B * np.cos(theta) * np.cos(phi) - z_b * B * np.cos(theta) * np.sin(phi),
                          -z_b * B * np.sin(theta) - x_b * B * np.cos(theta) * np.cos(phi),
                          x_b * B * np.cos(theta) * np.sin(phi) + y_b * B * np.sin(theta)])
        
        J1 = np.array([[np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi), np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)],
                       [np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.sin(phi) + np.sin(theta)*np.sin(psi)*np.cos(phi)],
                       [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]])

        J2 = np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
                      ], dtype=float)

        Zero = np.zeros((3, 3))

        J = np.block([[J1, Zero],
                      [Zero, J2]])
        
        M_inv = np.linalg.inv(M)
        
        nu_dot = M_inv @ (tau - C @ nu - D @ nu - g_eta)
        eta_dot = J @ nu

        return np.concatenate((nu_dot,eta_dot))
    
    

    

class AUV_States:
    def __init__(self, x, u=np.ndarray(shape=(6,), dtype=float)):
        self._x = x
        self._u = u

    # in n frame
    def translation_n(self):
        return np.array([self._x['pos'][0], 
                         self._x['pos'][1],
                         self._x['pos'][2]])
    # in b frame
    def translation_b(self):
        return self.R_nb().T @ self.translation_n()

    def R_nb(self):
        phi,theta,psi = self._x['pos'][3],self._x['pos'][4],self._x['pos'][5]

        J1 = np.array([[np.cos(psi)*np.cos(theta), -np.sin(psi)*np.cos(phi) + np.cos(psi)*np.sin(theta)*np.sin(phi), np.sin(psi)*np.sin(phi) + np.cos(psi)*np.sin(theta)*np.cos(phi)],
                       [np.sin(psi)*np.cos(theta), np.cos(psi)*np.cos(phi) + np.sin(psi)*np.sin(theta)*np.sin(phi), -np.cos(psi)*np.sin(phi) + np.sin(theta)*np.sin(psi)*np.cos(phi)],
                       [-np.sin(theta), np.cos(theta)*np.sin(phi), np.cos(theta)*np.cos(phi)]],dtype=float)
        
        return J1
    
    def T_euler(self):
        phi,theta,psi = self._x['pos'][3],self._x['pos'][4],self._x['pos'][5]

        J2 = np.array([
            [1, np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ],dtype=float)

        return J2
    
    def J(self):
        
        J = np.block(
                [
                    [self.R_nb(), np.zeros((3, 3))],
                    [np.zeros((3, 3)), np.self.T_euler()]
                ])
        
        return J
    
    def J_inv(self):
        J_inv = np.block(
            [
                [self.R_nb().T, np.zeros((3,3))],
                [np.zeros((3,3)), np.linalg.inv(self.T_euler())]
            ]
        )

        return J_inv


class AUV_Geometry:
    def __init__(self, length, width,height):
        self._length = length
        self._width = width
        self._height = height
        
    
    def get_plot_patch(self, state):
        length, width, height = self._length, self._width, self._height

        x, y, z = state[0], state[1], state[2]
        phi,theta,psi = state[3],state[4],state[5]

        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(phi), -math.sin(phi)],
                [0, math.sin(phi), math.cos(phi)]
            ]
        )

        Ry = np.array(
            [
                [math.cos(theta), 0, math.sin(theta)],
                [0, 1, 0],
                [-math.sin(theta), 0, math.cos(theta)]
            ]
        )

        Rz = np.array(
            [
                [math.cos(psi), -math.sin(psi), 0],
                [math.sin(psi), math.cos(psi), 0],
                [0, 0, 1]
            ]
        )
        rotation = (Rz @ Ry) @ Rx
        translation = np.array([x,y,z])

        vertices = np.array(
            [
                [
                    width / 2 ,
                    length / 2,
                    -height / 2
                ],
                [
                    width / 2 ,
                    length / 2,
                    height / 2
                ],
                [
                    -width / 2 ,
                    length / 2,
                    -height / 2
                ],
                [
                    -width / 2 ,
                    length / 2,
                    height / 2
                ],
                [
                    -width / 2 ,
                    -length / 2,
                    -height / 2
                ],
                [
                    -width / 2 ,
                    -length / 2,
                    height / 2
                ],
                [
                    width / 2 ,
                    -length / 2,
                    -height / 2
                ],
                [
                    width / 2 ,
                    -length / 2,
                    height / 2
                ]
            ]
        )

        points = translation + vertices @ rotation.T
        hull = ConvexHull(points)

        faces = [points[simplex] for simplex in hull.simplices]
        
        # poly3d = Poly3DCollection([points], alpha=0.3, edgecolor='r', facecolor='cyan', linewidths=1)

        # ax.add_collection3d(poly3d)
    
        return np.vstack(faces)
        


class AUV_System(System):
    
    def update(self,control_action):

        combined_state = np.concatenate((self._state._x['vel'], self._state._x['pos']))
        sol = solve_ivp(self._dynamics.forward_dynamics, [self._time, self._time + 0.02], combined_state, args=(control_action,), method='RK45')
        self._state._x['vel'] = sol.y[:, -1][:6] # in frame b
        self._state._x['pos'] = sol.y[:, -1][-6:] # in frame n

        self._state._u = control_action
        
        self._time += 0.02
    
    def logging(self,system_logger):
        system_logger._xs.append(self._state._x['pos'])
        system_logger._us.append(self._state._u)


    


