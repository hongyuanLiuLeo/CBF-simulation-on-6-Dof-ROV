import numpy as np

class ConstantSpeedTrajectoryGenerator3D:
    def __init__(self):
        # TODO: wrap params
        self._global_path_index = 0
        # TODO: number of waypoints shall equal to length of global path
        self._num_waypoint = None
        # local path
        self._reference_speed = 0.2
        self._num_horizon = 5
        self._local_path_timestep = 0.02
        self._local_trajectory = None
        self._proj_dist_buffer = 0.0005

    def generate_trajectory(self, system, global_path,time):
        # TODO: move initialization of _num_waypoint and _global_path to constructor
        if self._num_waypoint is None:
            self._global_path = global_path
            self._num_waypoint = global_path.shape[0]
        pos = system._state._x['pos'][:3]
        # TODO: pass _global_path as a reference
        return self.generate_trajectory_internal(pos, self._global_path)
    
    def generate_trajectory_internal(self, pos, global_path):
        local_index = self._global_path_index
        trunc_path = np.vstack([global_path[local_index:, :3], global_path[-1, :3]])
        curv_vec = trunc_path[1:, :] - trunc_path[:-1, :]
        curv_length = np.linalg.norm(curv_vec, axis=1)

        if curv_length[0] == 0.0:
            curv_direct = np.zeros((3,))
        else:
            curv_direct = curv_vec[0, :] / curv_length[0]
        proj_dist = np.dot(trunc_path[0, :] - pos, curv_direct)
        
        # if proj_dist >= curv_length[0] - self._proj_dist_buffer and local_index < self._num_waypoint - 1:
        #     self._global_path_index += 1 # need to move to next waypoint
        #     return self.generate_trajectory_internal(pos, global_path)

        
        if proj_dist <= 0.0 and local_index < self._num_waypoint - 1:
            proj_dist = 0.0
            self._global_path_index += 1 # need to move to next waypoint
            return self.generate_trajectory_internal(pos, global_path)

        
        self._local_trajectory = np.hstack([trunc_path[0,:],np.zeros(3)])
        # t_c = (proj_dist + self._proj_dist_buffer) / self._reference_speed
        # t_s = t_c + self._local_path_timestep * np.linspace(0, self._num_horizon - 1, self._num_horizon)

        # curv_time = np.cumsum(np.hstack([0.0, curv_length / self._reference_speed]))
        # curv_time[-1] += (
        #     t_c + 2 * self._local_path_timestep * self._num_horizon + self._proj_dist_buffer / self._reference_speed
        # )

        # path_idx = np.searchsorted(curv_time, t_s, side="right") - 1
        # path = np.vstack(
        #     [
        #         np.interp(t_s, curv_time, trunc_path[:, 0]),
        #         np.interp(t_s, curv_time, trunc_path[:, 1]),
        #         np.interp(t_s, curv_time, trunc_path[:, 2])
        #     ]
        # ).T

        # path_vel = self._reference_speed * np.ones((self._num_horizon, 1))
        # path_yaw = np.arctan2(curv_vec[path_idx, 1], curv_vec[path_idx, 0]).reshape(self._num_horizon, 1)
        # path_pitch = np.arctan2(curv_vec[path_idx, 2],np.sqrt(curv_vec[path_idx, 0]**2+curv_vec[path_idx, 1]**2)).reshape(self._num_horizon,1)
        # path_roll = np.zeros((self._num_horizon,1))

        # self._local_trajectory = np.hstack([path, path_roll, path_pitch, path_yaw])
        return self._local_trajectory

    def logging(self, logger):
        logger._trajs.append(self._local_trajectory)