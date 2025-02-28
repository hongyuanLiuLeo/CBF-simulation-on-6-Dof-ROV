import numpy as np

class circle2_trajectory_generator:
    def __init__(self):
        self._local_trajectory = []

    def generate_trajectory(self, sys,global_path,time):

        R = 5 # Radius of the circle
        V = 0.02 # Linear velocity

        phi_d = 0
        theta_d = 0
        

        # Compute position on the circular path
        x_d = R * np.sin(V * time)
        y_d =  R * np.cos(V * time) - R
        z_d = 0; # Assuming the circle lies on the xy-plane
        psi_d = -V*time; # Orientation (angle)
        
        self._local_trajectory = np.array([x_d,y_d,z_d,phi_d,theta_d,psi_d])
        return self._local_trajectory
    
    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)