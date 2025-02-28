import numpy as np

class lerp_trajectory_generator:
    def __init__(self):
        pass
    def generate_trajectory(self, global_path,time):
        start_point = global_path[0,:]
        end_point = global_path[-1,:]

        # print(start_point)
        # print(end_point)
        # b = 1
        t = np.linspace(0, 1, 10*time)
        self._local_trajectory = np.outer(1 - t, start_point) + np.outer(t, end_point)
        # print(self._local_trajectory)
        # a = 1
        return self._local_trajectory
    
    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)