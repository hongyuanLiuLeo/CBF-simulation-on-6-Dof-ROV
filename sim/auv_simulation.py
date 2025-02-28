import math
import casadi as ca
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.logger import *

class System:
    def __init__(self,time=0.0,state=None,geometry=None,dynamics=None):
        self._time = time
        self._state = state
        self._geometry = geometry
        self._dynamics = dynamics
            

class Robot:
    def __init__(self, system):
        self._system = system
        self._system_logger = SystemLogger()
    
    def set_global_path(self,global_path):
        self._global_path = global_path

    def set_global_planner(self, global_planner):
        self._global_planner = global_planner
        self._global_planner_logger = GlobalPlannerLogger()

    def set_local_planner(self, local_planner):
        self._local_planner = local_planner
        self._local_planner_logger = LocalPlannerLogger()

    def set_controller(self, controller):
        self._controller = controller
        self._controller_logger = ControllerLogger()

    def run_global_planner(self, sys, obstacles, goal_pos):
        
        self._global_path = self._global_planner.generate_path(sys, obstacles, goal_pos)
        self._global_planner.logging(self._global_planner_logger)

    def run_local_planner(self,time):
        
        self._local_trajectory = self._local_planner.generate_trajectory(self._system, self._global_path,time)
        self._local_planner.logging(self._local_planner_logger)

    def run_system(self):
        self._system.update(self._control_action)
        self._system.logging(self._system_logger)

    def run_controller(self,obs):
        self._control_action = self._controller.generate_control_input(
            self._system, self._local_trajectory, obs
        )
        # self._controller.logging(self._controller_logger)

    def run_system(self):
        self._system.update(self._control_action)
        self._system.logging(self._system_logger)

class SingleAgentSimulation:
    def __init__(self, robot, obstacles=None):
        self._robot = robot
        self._obstacles = obstacles

    def run_navigation(self, navigation_time):
        
        # self._robot.run_local_planner(navigation_time)

        while self._robot._system._time < navigation_time:
            
            self._robot.run_local_planner(navigation_time)

            # for i, point in enumerate(self._robot._local_trajectory):

            self._robot.run_controller(self._obstacles)
            self._robot.run_system()

                # self._robot.run_controller(point,self._obstacles)
                # self._robot.run_system()
    
