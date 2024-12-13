#测试ppo效果并简单可视化
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from nn_environment4 import NNEnvironment
from src.nn_model import PressureToPositionNet
from scipy.interpolate import splprep, splev #B样条曲线库


# 加载训练好的模型
# model = PPO.load("ppo_trained_agent102508")
model = PPO.load("ppo_1206_3.zip")

# 创建环境实例
nn_model = PressureToPositionNet()
# nn_model.load_state_dict(torch.load('nn_models/trained_model.pth'))
nn_model.load_state_dict(torch.load('nn_models/trained_nn_model10_swgt_fixRseed_centered.pth'))

nn_model.eval()
env = NNEnvironment(nn_model)

# 初始化3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # 打开交互模式

# 假设目标点的坐标（固定不变）
# target_position = np.array([415.70, 281.45, 221.13]) #[420.0, 280.0, 218.0]
target_position = np.array([-4.2325, -71.0387, 6.9697])
# 重映射目标点坐标
mapped_target = np.zeros(3)
mapped_target[0] = target_position[2]  # Z -> X
mapped_target[1] = target_position[0]  # X -> Y
mapped_target[2] = target_position[1]  # Y -> Z

def update_plot(points, ax):
    """更新3D图形"""
    ax.clear()

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
    tck, u = splprep([mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2]], s=0)
    u_fine = np.linspace(0, 1, 100)
    smooth_curve = splev(u_fine, tck)

    # 绘制 B 样条曲线
    ax.plot(smooth_curve[0], smooth_curve[1], smooth_curve[2], 'r-', linewidth=2)

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
    ax.set_xlim(-35, 35)  # X轴范围
    ax.set_ylim(-60, 10)  # Y轴范围
    ax.set_zlim(-35, 35)  # Z轴范围
    # 设置坐标轴
    ax.set_xlabel('X Coordinate (mocap data Z)')
    ax.set_ylabel('Y Coordinate (mocap data X)')
    ax.set_zlabel('Z Coordinate (mocap data Y)')
    ax.set_title('3D Visualization of 5 Points')
    plt.draw()
    plt.pause(0.001)  # 短暂停顿以刷新图像

# Step 2: 测试训练好的Policy并实时可视化
num_episodes = 1  # 测试1个回合
for episode in range(num_episodes):
    # obs = env.reset()  #gym环境
    obs, info = env.reset()  # Gym API，返回元组
    done = False
    total_reward = 0
    step = 0

    print(f"==== Episode {episode + 1} ====")


    while not done:
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
        update_plot(points, ax)

    print(f"Total Reward for Episode {episode + 1}: {total_reward}")

# 关闭交互模式
plt.ioff()
plt.show()
