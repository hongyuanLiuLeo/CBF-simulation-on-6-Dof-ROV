import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义起点和终点
start = np.array([0, 0, 0])  # 起点 (x1, y1, z1)
end = np.array([5, 5, 10])   # 终点 (x2, y2, z2)

# 生成时间步长 (0 到 1)
t = np.linspace(0, 1, 100)

# 线性插值生成轨迹
trajectory = np.outer(1 - t, start) + np.outer(t, end)

# 提取x, y, z坐标
x_vals = trajectory[:, 0]
y_vals = trajectory[:, 1]
z_vals = trajectory[:, 2]

# 可视化轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_vals, y_vals, z_vals, label='Linear trajectory')
ax.scatter(start[0], start[1], start[2], color='red', label='Start')
ax.scatter(end[0], end[1], end[2], color='green', label='End')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
