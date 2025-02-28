from geometry3D_utils import *

# 创建3D绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 假设 polytope 是你的 3D 凸多面体对象
polytope = PolytopeRegion3D.convex_hull(np.array(np.array([[0.228,0.218,-0.127],[0.228,0.218,0.127],
                                                        [-0.228,0.218,-0.127],[-0.228,0.218,0.127],
                                                        [-0.228,-0.218,-0.127],[-0.228,-0.218,0.127],
                                                        [0.228,-0.218,-0.127],[0.228,-0.218,0.127]])))

# 绘制多面体
polytope.get_plot_patch(ax)


ax.set_xlim([-5, 5])  # 设置 X 轴长度为 0 到 10
ax.set_ylim([-5, 5])   # 设置 Y 轴长度为 0 到 5
ax.set_zlim([-5, 5])  # 设置 Z 轴长度为 0 到 15
# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()