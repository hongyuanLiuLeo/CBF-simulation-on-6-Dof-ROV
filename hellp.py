import numpy as np
import pygame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 初始化 pygame
pygame.init()

# 定义 ROV 的动态模型
def rov_dynamics(state, control_input, dt):
    """
    ROV 的动力学模型，描述其在 3D 空间中的运动
    state: [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
           [位置和姿态(x, y, z, roll, pitch, yaw)，速度和角速度(u, v, w, p, q, r)]
    control_input: [Fx, Fy, Fz, Mx, My, Mz] 推力和力矩的输入
    dt: 时间步长
    """
    # 状态分为位置姿态和速度
    x, y, z, roll, pitch, yaw = state[:6]  # 位置和姿态
    u, v, w, p, q, r = state[6:]  # 线速度和角速度

    # 推力和力矩输入
    Fx, Fy, Fz, Mx, My, Mz = control_input

    # 假设质量和惯性矩阵为单位矩阵（简化）
    m = 1.0  # 质量
    I = np.eye(3)  # 惯性矩阵
    
    # 线速度的更新公式 (力作用)
    u_dot = Fx / m
    v_dot = Fy / m
    w_dot = Fz / m
    
    # 角速度的更新公式 (力矩作用)
    p_dot = Mx / I[0, 0]
    q_dot = My / I[1, 1]
    r_dot = Mz / I[2, 2]
    
    # 欧拉角速度的更新公式
    roll_dot = p + q * np.sin(roll) * np.tan(pitch) + r * np.cos(roll) * np.tan(pitch)
    pitch_dot = q * np.cos(roll) - r * np.sin(roll)
    yaw_dot = (q * np.sin(roll) + r * np.cos(roll)) / np.cos(pitch)
    
    # 更新速度
    u += u_dot * dt
    v += v_dot * dt
    w += w_dot * dt
    p += p_dot * dt
    q += q_dot * dt
    r += r_dot * dt
    
    # 更新位置和姿态
    x += u * dt
    y += v * dt
    z += w * dt
    roll += roll_dot * dt
    pitch += pitch_dot * dt
    yaw += yaw_dot * dt

    # 返回更新后的状态
    return np.array([x, y, z, roll, pitch, yaw, u, v, w, p, q, r])

# 初始化状态和控制输入
state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 初始状态 [x, y, z, roll, pitch, yaw, u, v, w, p, q, r]
control_input = np.array([0, 0, 0, 0, 0, 0])  # 推力和力矩输入 [Fx, Fy, Fz, Mx, My, Mz]

# 仿真参数
dt = 0.1  # 时间步长
trajectory = []

# 控制信号更新函数
def update_control_input(keys, control_input):
    # 推力控制 (W, A, S, D 控制 X 和 Y 轴方向)
    if keys[pygame.K_w]:
        control_input[0] = 10  # 增加 X 轴方向推力
    elif keys[pygame.K_s]:
        control_input[0] = -10  # 减少 X 轴方向推力
    else:
        control_input[0] = 0  # X 轴方向无推力

    if keys[pygame.K_a]:
        control_input[1] = -10  # 减少 Y 轴方向推力
    elif keys[pygame.K_d]:
        control_input[1] = 10  # 增加 Y 轴方向推力
    else:
        control_input[1] = 0  # Y 轴方向无推力

    # Z 轴推力 (上升和下降)
    if keys[pygame.K_UP]:
        control_input[2] = 10  # 增加 Z 轴方向推力（上升）
    elif keys[pygame.K_DOWN]:
        control_input[2] = -10  # 减少 Z 轴方向推力（下降）
    else:
        control_input[2] = 0  # Z 轴方向无推力

    # 力矩控制 (Q, E, Z, C 控制转动)
    if keys[pygame.K_q]:
        control_input[3] = 1  # 增加绕 X 轴的力矩（滚转）
    elif keys[pygame.K_e]:
        control_input[3] = -1  # 减少绕 X 轴的力矩
    else:
        control_input[3] = 0  # X 轴无力矩

    if keys[pygame.K_z]:
        control_input[4] = 1  # 增加绕 Y 轴的力矩（俯仰）
    elif keys[pygame.K_c]:
        control_input[4] = -1  # 减少绕 Y 轴的力矩
    else:
        control_input[4] = 0  # Y 轴无力矩

    if keys[pygame.K_LEFT]:
        control_input[5] = 1  # 增加绕 Z 轴的力矩（偏航）
    elif keys[pygame.K_RIGHT]:
        control_input[5] = -1  # 减少绕 Z 轴的力矩
    else:
        control_input[5] = 0  # Z 轴无力矩

# 设置 pygame 的屏幕
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("ROV Real-time Control")

# 初始化 Matplotlib 的 3D 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # 开启交互模式，实时更新图像

# 主循环
running = True
while running:
    # 检查事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 获取按键状态
    keys = pygame.key.get_pressed()

    # 更新控制输入
    update_control_input(keys, control_input)

    # 更新 ROV 动力学
    state = rov_dynamics(state, control_input, dt)
    trajectory.append(state[:6])  # 记录位置和姿态

    # 实时更新 3D 轨迹图
    ax.cla()  # 清除之前的绘图
    ax.plot([s[0] for s in trajectory], [s[1] for s in trajectory], [s[2] for s in trajectory], label='ROV trajectory')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()
    plt.draw()  # 绘制更新后的图像
    plt.pause(0.01)  # 短暂暂停以更新图像

# 关闭 pygame
pygame.quit()

# 最后显示完整轨迹
plt.ioff()  # 关闭交互模式
plt.show()

