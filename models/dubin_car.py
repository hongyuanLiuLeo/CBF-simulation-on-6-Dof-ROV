import datetime
import numpy as np
import matplotlib.patches as patches
import math
import casadi as ca

from geometry_utils import *

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.simulation import *

class DubinCarDynamics:
    @staticmethod
    def forward_dynamics(x, u, timestep=0.1):
        """Return updated state in a form of `np.ndnumpy`"""
        x_next = np.ndarray(shape=(4,), dtype=float)
        x_next[0] = x[0] + x[2] * math.cos(x[3]) * timestep
        x_next[1] = x[1] + x[2] * math.sin(x[3]) * timestep
        x_next[2] = u[0]
        x_next[3] = x[3] + u[1] * timestep
        return x_next

    @staticmethod
    def forward_dynamics_opt(timestep):
        """Return updated state in a form of `ca.SX`"""
        x_symbol = ca.SX.sym("x", 4)
        u_symbol = ca.SX.sym("u", 2)
        x_symbol_next = x_symbol[0] + x_symbol[2] * ca.cos(x_symbol[3]) * timestep
        y_symbol_next = x_symbol[1] + x_symbol[2] * ca.sin(x_symbol[3]) * timestep
        v_symbol_next = x_symbol[2] + u_symbol[0] * timestep
        theta_symbol_next = x_symbol[3] + u_symbol[1] * timestep
        state_symbol_next = ca.vertcat(x_symbol_next, y_symbol_next, v_symbol_next, theta_symbol_next)
        return ca.Function("dubin_car_dynamics", [x_symbol, u_symbol], [state_symbol_next])
    
    @staticmethod
    def nominal_safe_controller(x, timestep, amin, amax):
        """Return updated state using nominal safe controller in a form of `np.ndnumpy`"""
        u_nom = np.zeros(shape=(2,))
        u_nom[0] = np.clip(-x[2] / timestep, amin, amax)
        return DubinCarDynamics.forward_dynamics(x, u_nom, timestep), u_nom
    
    @staticmethod
    def safe_dist(x, timestep, amin, amax, dist_margin):
        """Return a safe distance outside which to ignore obstacles"""
        # TODO: wrap params
        safe_ratio = 1.25
        brake_min_dist = (abs(x[2]) + amax * timestep) ** 2 / (2 * amax) + dist_margin
        return safe_ratio * brake_min_dist + abs(x[2]) * timestep + 0.5 * amax * timestep ** 2

class DubinCarGeometry:
    def __init__(self, length, width):
        self._length = length
        self._width = width
        self._region = RectangleRegion(-length / 2, length / 2, -width / 2, width / 2)

    def equiv_rep(self):
        return [self._region]
    
    def get_plot_patch(self, state):
        length, width = self._length, self._width
        x, y, theta = state[0], state[1], state[3]
        vertices = np.array(
            [
                [
                    x + length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    y + length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
                [
                    x + length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    y + length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    x - length / 2 * np.cos(theta) + width / 2 * np.sin(theta),
                    y - length / 2 * np.sin(theta) - width / 2 * np.cos(theta),
                ],
                [
                    x - length / 2 * np.cos(theta) - width / 2 * np.sin(theta),
                    y - length / 2 * np.sin(theta) + width / 2 * np.cos(theta),
                ],
            ]
        )
        return patches.Polygon(vertices, alpha=1.0, closed=True, fc="None", ec="tab:brown")
    
    
class DubinCarMultipleGeometry:
    def __init__(self):
        self._num_geometry = 0
        self._geometries = []
        self._regions = []

    def equiv_rep(self):
        return self._regions

    def add_geometry(self, geometry):
        self._geometries.append(geometry)
        self._regions.append(geometry._region)
        self._num_geometry += 1

    def get_plot_patch(self, state, region_idx, alpha=0.1):
        return self._geometries[region_idx].get_plot_patch(state, alpha)

class DubinCarTriangleGeometry:
    def __init__(self, vertex_points):
        self._vertex_points = vertex_points
        self._region = PolytopeRegion.convex_hull(vertex_points)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=0.5):
        x, y, theta = state[0], state[1], state[3]
        mat_rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        vertices_points_curr = (mat_rotation @ self._vertex_points.T).T + np.array([x, y])
        return patches.Polygon(vertices_points_curr, alpha=alpha, closed=True, fc="tab:brown", ec="None", linewidth=0.5)


class DubinCarPentagonGeometry:
    def __init__(self, vertex_points):
        self._vertex_points = vertex_points
        self._region = PolytopeRegion.convex_hull(vertex_points)

    def equiv_rep(self):
        return [self._region]

    def get_plot_patch(self, state, i=0, alpha=0.5):
        x, y, theta = state[0], state[1], state[3]
        mat_rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        vertices_points_curr = (mat_rotation @ self._vertex_points.T).T + np.array([x, y])
        return patches.Polygon(vertices_points_curr, alpha=alpha, closed=True, fc="tab:brown", ec="None", linewidth=0.5)


class DubinCarStates:
    def __init__(self, x, u=np.array([0.0, 0.0])):
        self._x = x
        self._u = u

    def translation(self):
        return np.array([[self._x[0]], [self._x[1]]])

    def rotation(self):
        return np.array(
            [
                [math.cos(self._x[3]), -math.sin(self._x[3])],
                [math.sin(self._x[3]), math.cos(self._x[3])],
            ]
        )

class DubinCarSystem(System):

    def get_state(self):
        return self._state._x
    

    # def logging(self,logger):
    #     logger._xs.append(self._state._x)
    #     logger._us.append(self._state._u)

    def update(self,control_action):
        self._state._x = self._dynamics.forward_dynamics(self._state._x, control_action)
        self._state._u = control_action
        
        self._time += 0.1
    
    def logging(self,system_logger):
        system_logger._xs.append(self._state._x)
        system_logger._us.append(self._state._u)
        # system_logger._xs = np.vstack((system_logger._xs,self._state._x.reshape(1,4)))
        # system_logger._us.append(self._state._u)
