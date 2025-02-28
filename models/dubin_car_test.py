import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import animation

from control.PID_controller import *
from geometry_utils import *
from dubin_car import *

from planning.trajectory_generator.constant_speed_generator import *


start_pos = np.array([0,0,0.785],dtype=float)
goal_pos = np.array([3,3,0.785],dtype=float)

start_pos2 = np.array([-0.5,0,0.785],dtype=float)

robot = Robot(
    DubinCarSystem(
        state=DubinCarStates(x=np.block([start_pos[:2],np.array([0.0,start_pos[2]])])),
        geometry=DubinCarGeometry(0.44,0.38),
        dynamics=DubinCarDynamics()
    )

)

robot2 = Robot(
    DubinCarSystem(
        state=DubinCarStates(x=np.block([start_pos2[:2],np.array([0.0,start_pos2[2]])])),
        geometry=DubinCarGeometry(0.44,0.38),
        dynamics=DubinCarDynamics()
    )

)


obs1 = PolytopeRegion.convex_hull(np.array([[2,2],[4,2],[4,4],[2,4]]))
obs2 = PolytopeRegion.convex_hull(np.array([[8,8],[10,8],[10,10],[8,10]]))
obs3 = PolytopeRegion.convex_hull(np.array([[7.5,4],[10,6],[12,4]]))
obs4 = PolytopeRegion.convex_hull(np.array([[-5,5],[-5,7.5],[-6,5]]))
obs5 = PolytopeRegion.convex_hull(np.array([[0,11],[2.5,10],[1.5,8],[-1.5,8],[-2.5,10]]))
obstacles = [obs1,obs2,obs3,obs4,obs5]

global_path = np.array([[0,0],[5,5],[6,10]])
global_path2 = np.array([[-0.5,0],[-5,2]])

robot.set_global_path(global_path)
robot.set_local_planner(ConstantSpeedTrajectoryGenerator())
robot.set_controller(PIDController())
sim =SingleAgentSimulation(robot,obstacles)
sim.run_navigation(50)

robot2.set_global_path(global_path2)
robot2.set_local_planner(ConstantSpeedTrajectoryGenerator())
robot2.set_controller(PIDController())
sim2 =SingleAgentSimulation(robot2,obstacles)
sim2.run_navigation(50)

fig, ax = plt.subplots()
plt.axis("equal")
# ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)
# plt.show()

#local_paths = np.vstack(sim._robot._local_planner_logger._trajs)
#print(local_paths)

# print("-----------------------------")
# print(sim._robot._system_logger._xs)
# local_paths = np.vstack(sim._robot._local_planner_logger._trajs)
# print("---------------------------------")
# print(local_paths)
# print("-----------------------------------")
# print(np.array(sim._robot._system_logger._us))
# print("-------------------------------------")

# ax.plot(sim._robot._system_logger._xs[:,0],sim._robot._system_logger._xs[:,1],"bo--", linewidth=1, markersize=4)
# ax.plot(local_paths[:,0],local_paths[:,1])
# circle = plt.Circle((0,0),10,color='g',fill=False)
# ax.add_patch(circle)

# for obs in obstacles:
#     obs_patch = obs.get_plot_patch()
#     ax.add_patch(obs_patch)

# plt.grid(True)
# plt.show()

circle = plt.Circle((0,0),10,color='g',fill=False)
ax.add_patch(circle)

ax.plot(global_path[:, 0], global_path[:, 1], "bo--", linewidth=1, markersize=4)
ax.plot(global_path2[:, 0], global_path2[:, 1], "bo--", linewidth=1, markersize=4)

local_paths = sim._robot._local_planner_logger._trajs
local_path = local_paths[0]
(reference_traj_line,) = ax.plot(local_path[:, 0], local_path[:, 1])

local_paths2 = sim2._robot._local_planner_logger._trajs
local_path2 = local_paths2[0]
(reference_traj_line2,) = ax.plot(local_path2[:, 0], local_path2[:, 1])

closedloop_traj = np.vstack(sim._robot._system_logger._xs)
ax.plot(closedloop_traj[:, 0], closedloop_traj[:, 1], "k-", linewidth=0.5, markersize=4)

closedloop_traj2 = np.vstack(sim2._robot._system_logger._xs)
ax.plot(closedloop_traj2[:, 0], closedloop_traj2[:, 1], "k-", linewidth=0.5, markersize=4)

for obs in obstacles:
    obs_patch = obs.get_plot_patch()
    ax.add_patch(obs_patch)

robot_patch = patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="tab:brown")
ax.add_patch(robot_patch)

robot_patch2 = patches.Polygon(np.zeros((1, 2)), alpha=1.0, closed=True, fc="None", ec="tab:brown")
ax.add_patch(robot_patch2)

def update(index):
    # local_path = local_paths[index]
    # reference_traj_line.set_data(local_path[:, 0], local_path[:, 1])
    # optimized_traj = optimized_trajs[index]
    # optimized_traj_line.set_data(optimized_traj[:, 0], optimized_traj[:, 1])
    polygon_patch_next = sim._robot._system._geometry.get_plot_patch(closedloop_traj[index, :])
    robot_patch.set_xy(polygon_patch_next.get_xy())

    polygon_patch_next2 = sim2._robot._system._geometry.get_plot_patch(closedloop_traj2[index, :])
    robot_patch2.set_xy(polygon_patch_next2.get_xy())
    
anim = animation.FuncAnimation(fig, update, frames=np.arange(0,len(closedloop_traj)), interval=1000 * 0.1)
anim.save("animations/world2.mp4", dpi=300, writer=animation.writers["ffmpeg"](fps=60))

# local_paths = np.vstack(sim._robot._local_planner_logger._trajs)
# ax.plot(local_paths[:,0],local_paths[:,1])




