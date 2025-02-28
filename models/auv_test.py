import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation

from control.PID_AUV import *
from geometry3D_utils import *
from auv import *
from sim.auv_simulation import *
from planning.trajectory_generator.step_trajectory import *
from planning.trajectory_generator.lerp_trajectory import *
from planning.trajectory_generator.circle_trajectory import *
from planning.trajectory_generator.circle2_trajectory import *
from planning.trajectory_generator.constant_speed_generator3D import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

start_point = np.array([0,0,0,0,0,0],dtype=float)
# goal_point = np.array([1,1,1,0.0,0.0,0.0],dtype=float)
goal_point = np.array([[1.0,0.0,0.0,0.0,0.0,0.0],
                       [2.0,0.0,0.0,0.0,0.0,0.0]],dtype=float)

x_init = {
    'pos':start_point,
    'vel':np.array([0,0,0,0,0,0],dtype=float)
}
robot = Robot(
    AUV_System(
        state=AUV_States(x_init,np.zeros(6)),
        geometry=AUV_Geometry(0.575,0.457,0.254), # length=0.575, width=0.457, height = 0.254
        dynamics=AUV_Dynamics()
    )

)

vertices1 = np.array([
    [0.5, -0.1, -0.1],
    [0.5, -0.1,  0.1],
    [0.5,  0.1, -0.1],
    [0.5,  0.1,  0.1],
    [0.7, -0.1, -0.1],
    [0.7, -0.1,  0.1],
    [0.7,  0.1, -0.1],
    [0.7,  0.1,  0.1]])

obs1 = PolytopeRegion3D.convex_hull(vertices1)
obstacles = [obs1]

run_time = 100
robot.set_global_path(goal_point)
robot.set_local_planner(ConstantSpeedTrajectoryGenerator3D())
robot.set_controller(PID())
sim = SingleAgentSimulation(robot,obstacles)
sim.run_navigation(run_time)

local_paths = np.vstack(sim._robot._local_planner_logger._trajs)
closedloop_traj = np.vstack(sim._robot._system_logger._xs)
input_control = np.vstack(sim._robot._system_logger._us)

print(closedloop_traj)

# fig2 = plt.figure()
time = np.linspace(0, run_time, len(closedloop_traj))
# plt.plot(time, closedloop_traj[:,0], label="System Response x")
# plt.plot(time, closedloop_traj[:,1], label="System Response y")
# plt.plot(time, closedloop_traj[:,2], label="System Response z")
# plt.plot(time, local_paths[:,0], linestyle='--',label="reference x")
# plt.plot(time, local_paths[:,1], linestyle='--',label="reference y")
# plt.plot(time, local_paths[:,2], linestyle='--',label="reference z")
# plt.xlabel("Time (s)")
# plt.ylabel("Response")
# plt.title("PID Control System Step Response")
# plt.legend()
# plt.grid()
# plt.show()

fig2 = plt.figure()
plt.plot(time, closedloop_traj[:, 0], label="System Response x")
plt.plot(time, local_paths[:, 0], linestyle='--', label="Reference x")
plt.xlabel("Time (s)")
plt.ylabel("Response (x)")
plt.title("PID Control: X Response")
plt.legend()
plt.grid()

fig3 = plt.figure()
plt.plot(time, closedloop_traj[:, 1], label="System Response y")
plt.plot(time, local_paths[:, 1], linestyle='--', label="Reference y")
plt.xlabel("Time (s)")
plt.ylabel("Response (y)")
plt.title("PID Control: Y Response")
plt.legend()
plt.grid()

fig4 = plt.figure()
plt.plot(time, closedloop_traj[:, 2], label="System Response z")
plt.plot(time, local_paths[:, 2], linestyle='--', label="Reference z")
plt.xlabel("Time (s)")
plt.ylabel("Response (z)")
plt.title("PID Control: Z Response")
plt.legend()
plt.grid()

fig5 = plt.figure()
plt.plot(time, closedloop_traj[:, 3], label="System Response roll")
plt.plot(time, local_paths[:, 3], linestyle='--', label="Reference roll")
plt.xlabel("Time (s)")
plt.ylabel("Response (roll)")
plt.title("PID Control: roll Response")
plt.legend()
plt.grid()

fig6 = plt.figure()
plt.plot(time, closedloop_traj[:, 4], label="System Response pitch")
plt.plot(time, local_paths[:, 4], linestyle='--', label="Reference pitch")
plt.xlabel("Time (s)")
plt.ylabel("Response (pitch)")
plt.title("PID Control: pitch Response")
plt.legend()
plt.grid()

fig7 = plt.figure()
plt.plot(time, closedloop_traj[:, 5], label="System Response yaw")
plt.plot(time, local_paths[:, 5], linestyle='--', label="Reference yaw")
plt.xlabel("Time (s)")
plt.ylabel("Response (yaw)")
plt.title("PID Control: yaw Response")
plt.legend()
plt.grid()


fig8 = plt.figure()
plt.plot(time, input_control)
plt.xlabel("Time (s)")
plt.ylabel("Input u")
plt.title("PID Control Input")
plt.grid()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

ax.set_xlim([-5, 5])  
ax.set_ylim([-5, 5])   
ax.set_zlim([-5, 5])  

ax.invert_zaxis()

# ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], closedloop_traj[:, 2], "k-", linewidth=0.5)

ax.scatter(start_point[0],start_point[1],start_point[2],color='green',s=30,label='Start Point')
ax.scatter(goal_point[:,0],goal_point[:,1],goal_point[:,2],color='red',s=30,label='End Point')

for obs in obstacles:
    obs.get_plot_patch(ax)

robot_patch = Poly3DCollection([np.zeros((1, 3))], alpha=0.3, edgecolor='r', facecolor='cyan', linewidths=1)
ax.add_collection3d(robot_patch)

# ax.set_xlim(min(closedloop_traj[:, 0]), max(closedloop_traj[:, 0]))
# ax.set_ylim(min(closedloop_traj[:, 1]), max(closedloop_traj[:, 1]))
# ax.set_zlim(min(closedloop_traj[:, 2]), max(closedloop_traj[:, 2]))

def update(index):
    
    polygon_patch_next = sim._robot._system._geometry.get_plot_patch(closedloop_traj[index, :])
    robot_patch.set_verts([polygon_patch_next])
    ax.add_collection3d(robot_patch)
    ax.scatter(closedloop_traj[index, 0], closedloop_traj[index, 1], closedloop_traj[index, 2], color='black', s=1)
    
    

anim = animation.FuncAnimation(fig, update, frames=np.arange(0,len(closedloop_traj)), interval=1000 * 0.1)
plt.show()

# anim = animation.FuncAnimation(fig, update, frames=np.arange(0,len(closedloop_traj),5), interval=1000 * 0.1)
# anim.save("animations/auv2.mp4", dpi=150, writer=animation.writers["ffmpeg"](fps=30))