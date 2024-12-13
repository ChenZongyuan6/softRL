#射线与平面交点的
#测试ppo效果并简单可视化，用于gymnasium 环境，

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from nn_environment4 import NNEnvironment
from src.nn_model import PressureToPositionNet
from scipy.interpolate import splprep, splev #B样条曲线库


# 加载训练好的模型
# model = PPO.load("ppo_trained_agent102508")
model = PPO.load("ppo_1211_circle_2.zip")

# 创建预训练神经网络模型实例并加载权重
nn_model = PressureToPositionNet()
# nn_model.load_state_dict(torch.load('nn_models/trained_model.pth'))
nn_model.load_state_dict(torch.load('nn_models/trained_nn_model10_swgt_fixRseed_centered.pth'))
nn_model.eval()
env = NNEnvironment(nn_model)

# 初始化3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # 打开交互模式

# 末端第5个点目标位置坐标（固定不变）
# target_position = np.array([415.70, 281.45, 221.13]) #[420.0, 280.0, 218.0]
# target_position = np.array([-4.2325, -71.0387, 6.9697])
# # 重映射目标点坐标
# mapped_target = np.zeros(3)
# mapped_target[0] = target_position[2]  # Z -> X
# mapped_target[1] = target_position[0]  # X -> Y
# mapped_target[2] = target_position[1]  # Y -> Z
#前面的坐标表示是没remapped的，下面直接给出remap过的坐标
target_position = np.array([-8.5773, -5.3201, -78.610267])
mapped_target = target_position

def update_plot(points, ax):
    """更新3D图形"""
    # ax.clear() #据说是2D的方法，3d用cla
    ax.cla()
    # 映射坐标轴：Z -> X, X -> Y, Y -> Z
    mapped_points = np.zeros_like(points)
    mapped_points[:, 0] = points[:, 2]  # Z -> X
    mapped_points[:, 1] = points[:, 0]  # X -> Y
    mapped_points[:, 2] = points[:, 1]  # Y -> Z

    # 绘制5个捕捉点
    ax.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], c='r', s=50)
     # 绘制目标点（绿色圆点）
    ax.scatter(mapped_target[0], mapped_target[1], mapped_target[2], c='g', marker='o', s=50)
    # 为每个点添加序号标注
    for i, point in enumerate(mapped_points):
        ax.text(point[0], point[1], point[2], f'{i+1}', fontsize=10)

    # 使用 B 样条平滑连接捕捉点
    tck, u = splprep([mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2]], s=0) #_用来丢弃最后一个返回值info（用不到
    u_fine = np.linspace(0, 1, 100)
    smooth_curve = splev(u_fine, tck)

    # 绘制 B 样条曲线
    ax.plot(smooth_curve[0], smooth_curve[1], smooth_curve[2], 'r-', linewidth=2)
    # 定义显示范围 可注释掉，换成用mid和max min定义
    x_min = -35
    x_max = 35
    y_min = -35
    y_max = 35
    z_min = -70
    z_max = 0

    # 设置每个轴的范围，使得X、Y、Z方向比例一致

    # 绘制坐标轴（蓝色宽度为1的线）
    ax.plot([x_min, x_max], [0, 0], [0, 0], color='b', linewidth=1)  # X轴
    ax.plot([0, 0], [y_min, y_max], [0, 0], color='b', linewidth=1)  # Y轴
    ax.plot([0, 0], [0, 0], [z_min, z_max], color='b', linewidth=1)  # Z轴
    # **新增：绘制平面**
    # 给定平面上的一个点和法向量
    point_on_plane = np.array([0, 0, -80])  # 假设平面过原点 (0, 0, 0)
    normal_vector = np.array([0.1, 0.1, 1])  # 假设平面的法向量为 [1, 1, 1]
    # 计算平面上的 x 和 y 的值，进而确定对应的 z 值
    x_vals = np.linspace(x_min, x_max, 10)
    y_vals = np.linspace(y_min ,y_max, 10)
    X, Y = np.meshgrid(x_vals, y_vals)
    # 根据点法式平面方程计算 Z 值
    # 公式：Ax + By + Cz + D = 0
    # 其中，A, B, C 是法向量的分量，D 是常数
    A, B, C = normal_vector
    D = -(A * point_on_plane[0] + B * point_on_plane[1] + C * point_on_plane[2])
    # 计算 Z
    Z = -(A * X + B * Y + D) / C
    # 绘制平面
    ax.plot_surface(X, Y, Z, color='c', alpha=0.3)
    # **新增：计算并绘制直线射线**
    # 取最后两个点，第4个点 P4 和第5个点 P5
    P4 = mapped_points[3]  # 第4个点
    P5 = mapped_points[4]  # 第5个点
    # 计算直线的方向向量
    direction_vector = P5 - P4  # 直线的方向向量 (P5 - P4)
    # 为了绘制射线，选择从 P4 开始，沿着方向向量延伸
    t_vals = np.linspace(0, 5, 100)  # 控制射线的长度（从 P4 延伸出 5 个单位长度）
    # 计算射线上的点
    ray_points = P4 + t_vals[:, np.newaxis] * direction_vector
    # 绘制射线
    ax.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], color='r', linewidth=1, label='Ray from P4 to P5')

    # **计算射线和平面的交点**
    # 使用射线和平面方程来计算交点
    denominator = A * direction_vector[0] + B * direction_vector[1] + C * direction_vector[2]
    numerator = -(A * P4[0] + B * P4[1] + C * P4[2] + D)
    t_intersection = numerator / denominator
    # 计算交点坐标
    intersection_point = P4 + t_intersection * direction_vector
    # 输出交点坐标
    print(
        f"Intersection point coordinates: x = {intersection_point[0]}, y = {intersection_point[1]}, z = {intersection_point[2]}")
    # 绘制交点
    ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='r', s=30,
               label='Intersection Point')


    # 显示当前 episode 和 step
    ax.text2D(0.05, 0.95, f"Episode: {episode}", transform=ax.transAxes, fontsize=8, color='black')
    ax.text2D(0.05, 0.90, f"Step: {step}", transform=ax.transAxes, fontsize=8, color='black')
    # 固定坐标轴范围
    # ax.set_xlim(210, 230)  # X轴范围
    # ax.set_ylim(415, 435)  # Y轴范围
    # ax.set_zlim(280, 350)  # Z轴范围
    # ax.set_xlim(185, 255)  # X轴范围
    # ax.set_ylim(390, 460)  # Y轴范围
    # ax.set_zlim(280, 350)  # Z轴范围
    #中心化后的坐标轴范围
    ax.set_xlim(x_min, x_max)  # X轴范围
    ax.set_ylim(y_min, y_max)  # Y轴范围
    ax.set_zlim(z_min, z_max)  # Z轴范围
    # 设置坐标轴
    ax.set_xlabel('X Coordinate (mocap data Z)')
    ax.set_ylabel('Y Coordinate (mocap data X)')
    ax.set_zlabel('Z Coordinate (mocap data Y)')
    ax.set_title('3D Visualization of 5 Points')
    plt.draw()
    plt.pause(0.2)  # 短暂停顿以刷新图像
    return intersection_point  # 返回交点坐标

# Step 2: 测试训练好的Policy并实时可视化
num_episodes = 4  # 测试1个回合
for episode in range(num_episodes):
    # obs = env.reset()  #gym环境
    obs, info = env.reset()  # Gym API，返回元组
    done = False
    truncated = False
    total_reward = 0
    step = 0

    print(f"==== Episode {episode + 1} ====")

    # while not done:
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # 打印当前状态、奖励和气压
        print(f"Step: {step}, Action: {action}, Reward: {reward}, Distance: {info.get('distance_to_target')}")
        print(f"Pressure: {info.get('pressure')}, End Effector Position: {info.get('state')[-3:]}")

        # 提取15维状态并绘制
        predicted_output = info.get('state')
        points = np.array(predicted_output).reshape(5, 3)
        # 获取交点并绘制
        intersection_point = update_plot(points, ax)

        # 获取末端执行器的XYZ坐标
        end_effector_position = np.array(info.get('state')[-3:])

        # 计算末端执行器与目标位置的欧式距离
        distance_to_target = np.linalg.norm(intersection_point - target_position)
        #
        # 如果距离小于0.2，则终止当前测试
        if distance_to_target < 0.1:
            print(f"Target reached at step {step}. Distance to target: {distance_to_target}")
            done = True

    print(f"Total Reward for Episode {episode + 1}: {total_reward}")

# 关闭交互模式
plt.ioff()
plt.show()
