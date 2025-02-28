import numpy as np

class step_trajectory_generator:
    def __init__(self):
        self._local_trajectory = []

    def generate_trajectory(self, sys, global_path, time):

        if time <= 0:
            self._local_trajectory = np.array([0,0,0,0,0,0])
            

        if time > 0:
            self._local_trajectory = global_path
        

        return self._local_trajectory
    
    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)