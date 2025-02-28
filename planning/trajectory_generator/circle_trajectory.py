import numpy as np

class circle_trajectory_generator:
    def __init__(self):
        self._local_trajectory = {}

    def generate_trajectory(self, global_path,time):
        start_point = global_path[0,:]
        end_point = global_path[-1,:]

        self.radius = 5
        self.dt = 0.0001
        self.time_tol = 15

        if time > self.time_tol:
            pos = np.array([self.radius, 0, 2.5])  # Final position if time exceeds time_tol
            vel = np.zeros(3)
            acc = np.zeros(3)
        else:
            angle = self.calculate_angle(0, 2*np.pi, self.time_tol, time)
            pos = self.compute_position(angle)
            vel = self.compute_velocity(time)
            acc = (self.compute_velocity(time + self.dt) - self.compute_velocity(time)) / self.dt
        yaw = 0
        yawdot = 0
        # self._local_trajectory = {
        #     'des_pos' : pos,
        #     'des_vel' : vel,
        #     'des_acc' : acc,
        #     'des_yaw' : 
        # }
        self._local_trajectory['des_pos']=pos
        self._local_trajectory['des_vel']=vel
        self._local_trajectory['des_acc']=acc
        self._local_trajectory['des_yaw']=yaw
        self._local_trajectory['des_yawdot']=yawdot

        return self._local_trajectory
    

    def calculate_angle(self,start, end, duration, time):
        return start + (end - start) * np.clip(time / duration, 0, 1)
    
    def compute_position(self,angle):
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        z = 2.5 * angle / (2 * np.pi)
        return np.array([x, y, z])
    
    def compute_velocity(self,current_time):
        angle_now = self.calculate_angle(0, 2 * np.pi, self.time_tol, current_time)
        pos_now = self.compute_position(angle_now)
        
        angle_next = self.calculate_angle(0, 2 * np.pi, self.time_tol, current_time + self.dt)
        pos_next = self.compute_position(angle_next)
        
        return (pos_next - pos_now) / self.dt
    
    def logging(self, logger):
        logger._trajs.append(self._local_trajectory['des_pos'])