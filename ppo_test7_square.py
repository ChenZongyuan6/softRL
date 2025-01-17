# -----------------------
# 测试并可视化已训练的RL策略模型
# -----------------------
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
# from nn_environment11_square import NNEnvironment  # 确保与训练环境一致
# from nn_environment12_circle_10point_obs import NNEnvironment
from nn_environment13_circle_10point_obs_relpos import NNEnvironment
from nn_training.nn_model import PressureToPositionNet
from scipy.interpolate import splprep, splev  # B样条曲线库
from mpl_toolkits.mplot3d import Axes3D  # 确保3D绘图支持

# 全局变量：任务名称，确保与训练代码中的 TASK_NAME 一致
# TASK_NAME = "ppo_1224_circlenew_5_110k"
TASK_NAME = "30_ppo_0116_circlenew_2"

# -----------------------------------------------------------
# 加载已训练的RL策略模型
model_path = f"./rl_agent/{TASK_NAME}"  # 使用全局变量构建路径
model = PPO.load(model_path)

# 加载已训练好的NN模型
nn_model = PressureToPositionNet()
# nn_model.load_state_dict(torch.load('nn_models/trained_nn_model10_swgt_fixRseed_centered.pth', map_location='cpu'))
nn_model.load_state_dict(torch.load('nn_models/30_trained_nn_model_202412_4.pth', map_location='cpu'))
nn_model.eval()

# 创建测试环境
env = NNEnvironment(
    nn_model=nn_model,
    device='cpu',  # 根据需要修改设备
    max_steps=256,  # 设置测试时的最大步数
    seed=42,
    env_index=0  # 可以根据需要指定不同的环境索引
)


# -----------------------------------------------------------
# 准备可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # 打开交互模式


def update_plot(points, trajectory, ax, episode, step, target_position):
    """更新3D图形"""
    ax.clear()
    # print(f"Target position: {target_position}")

    # 映射坐标轴：Z -> X, X -> Y, Y -> Z
    # 这是根据您给定的映射方式进行的坐标变换（可根据需要修改或移除）
    mapped_points = np.zeros_like(points)
    mapped_points[:, 0] = points[:, 2]  # Z -> X
    mapped_points[:, 1] = points[:, 0]  # X -> Y
    mapped_points[:, 2] = points[:, 1]  # Y -> Z

    mapped_target = np.zeros(3)
    mapped_target[0] = target_position[2]
    mapped_target[1] = target_position[0]
    mapped_target[2] = target_position[1]

    # 绘制关键点
    ax.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], c='r', s=20, label='Keypoints')
    # 绘制目标点（绿色标记）
    ax.scatter(mapped_target[0], mapped_target[1], mapped_target[2], c='g', marker='o', s=20, label='Target')

    # 为每个点添加序号标注
    for i, point in enumerate(mapped_points):
        ax.text(point[0], point[1], point[2], f'{i + 1}', fontsize=10)

    # 使用 B 样条平滑连接关键点（如果点数过少，也可以直接连线）
    if len(points) >= 5:
        tck, u = splprep([mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2]], s=0)
        u_fine = np.linspace(0, 1, 100)
        smooth_curve = splev(u_fine, tck)
        # 绘制 B 样条曲线
        ax.plot(smooth_curve[0], smooth_curve[1], smooth_curve[2], 'r-', linewidth=2, label='softRobot')
    else:
        # 如果点数不够5，可以用简单直线连接
        ax.plot(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], 'r-', linewidth=2, label='Trajectory')

    # 绘制末端执行器轨迹
    if len(trajectory) > 1:
        trajectory = np.array(trajectory)
        trajectory_mapped = np.zeros_like(trajectory)
        trajectory_mapped[:, 0] = trajectory[:, 2]  # Z -> X
        trajectory_mapped[:, 1] = trajectory[:, 0]  # X -> Y
        trajectory_mapped[:, 2] = trajectory[:, 1]  # Y -> Z

        ax.plot(
            trajectory_mapped[:, 0],
            trajectory_mapped[:, 1],
            trajectory_mapped[:, 2],
            'r--', linewidth=0.8, label='Trajectory'
        )

    # 设置坐标轴范围
    x_min, x_max = -35, 35
    y_min, y_max = -35, 35
    z_min, z_max = -75, 5  # 根据您的目标点调整Z轴范围
    # x_min, x_max = -10, 10
    # y_min, y_max = -10, 10
    # z_min, z_max = -75, -55  # 根据您的目标点调整Z轴范围

    # 绘制坐标轴（蓝色线）
    ax.plot([x_min, x_max], [0, 0], [0, 0], color='b', linewidth=1)
    ax.plot([0, 0], [y_min, y_max], [0, 0], color='b', linewidth=1)
    ax.plot([0, 0], [0, 0], [z_min, z_max], color='b', linewidth=1)

    # 显示当前episode和step
    ax.text2D(0.05, 0.95, f"Episode: {episode}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(0.05, 0.90, f"Step: {step}", transform=ax.transAxes, fontsize=10, color='black')
    ax.text2D(0.05, 0.85, f"Target: {target_position}", transform=ax.transAxes, fontsize=10, color='green')

    # 设置坐标轴范围
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    # 设置坐标轴标签
    ax.set_xlabel('X (from Z of mocap)')
    ax.set_ylabel('Y (from X of mocap)')
    ax.set_zlabel('Z (from Y of mocap)')
    ax.set_title('3D Visualization of Keypoints')

    # 添加图例
    # ax.legend()
    ax.legend(loc='upper right')

    plt.draw()
    plt.pause(0.001)


# -----------------------------------------------------------
# 测试回合数
num_episodes = 1  # 根据需要修改
max_step = 256 # 测试专用的最大步数限制

for episode in range(1, num_episodes + 1):
    # obs, _ = env.reset()
    obs, info = env.reset()  # 修改为接收 info
    done = False
    truncated = False
    total_reward = 0
    step = 0

    # 记录action与pressure数据
    action_x = []
    action_y = []
    action_z = []
    pressure_x = []
    pressure_y = []
    pressure_z = []

    # 初始的 target_position 通过 reset 返回的 info 获取
    # 由于 reset 返回的是 (observation, info)，但在训练代码中 reset 返回的是 observation 和 {}
    # 这里需要修改环境的 reset 方法使其返回 info，或者在 reset 后获取 target_position
    # 假设环境的 reset 返回 (observation, info)
    # 如果不是，请根据实际情况调整
    # 这里假设 target_position 存储在 env.target_position
    # 从 info 中获取初始的 target_position
    #适配nn_environment12_circle
    target_position = info.get('target_position', np.zeros(3))  # 从 info 字典中获取
    # 记录末端执行器的轨迹
    trajectory = np.empty((0, 3))  # 初始化为形状 (0, 3) 的空数组

    # target_position = env.target_position  # 从环境对象直接获取，适配nn_environment11_square


    # 记录末端执行器的轨迹
    trajectory = np.empty((0, 3))  # 初始化为形状 (0, 3) 的空数组

    print(f"==== Episode {episode} ====")

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)  # 使用确定性策略
        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        # 记录action数据
        action_x.append(action[0])
        action_y.append(action[1])
        action_z.append(action[2])

        # 记录pressure数据
        pressure = info.get('pressure', np.zeros(3))
        pressure_x.append(pressure[0])
        pressure_y.append(pressure[1])
        pressure_z.append(pressure[2])

        distance_to_target = info.get('distance_to_target', None)
        target_position = info.get('target_position', target_position)
        # print(f"Step: {step}, Action: {action}, Reward: {reward:.4f}, Distance: {distance_to_target:.4f}"
        #       f"Pressure: {[f'{p:.4f}' for p in pressure]}", f"target position:{target_position}")

        # 从info中提取关键点坐标
        keypoints = info.get('state_keypoints', None)
        if keypoints is not None and len(keypoints) == 15:
            # 将15维关键点重新整形为5个点(5 x 3)
            points = np.array(keypoints).reshape(5, 3)

            # 计算最后一个关键点到目标点的距离
            end_effector_position = points[-1]
            distance_to_target = np.linalg.norm(end_effector_position - target_position)
            # print(f"Step: {step}, Action: {action}, Reward: {reward:.4f}, Distance: {distance_to_target:.4f}"
            #       f"Pressure: {[f'{p:.4f}' for p in pressure]}", f"target position:{target_position}", f"End Effector Distance to Target: {distance_to_target:.4f}")
            # 记录末端执行器位置
            trajectory = np.vstack([trajectory, end_effector_position])  # 追加末端执行器位置

            # 更新3D图像
            update_plot(points, trajectory , ax, episode, step, target_position)
        else:
            print("Warning: keypoints数据不正常或长度不为15维。")

        # 当距离小于某个阈值时提前结束
        # if distance_to_target is not None and distance_to_target < 0.2:
        #     print(f"Target reached at step {step}. Distance: {distance_to_target:.4f}")
        #     done = True
        # 添加 max_step 限制
        if step >= max_step:
            print(f"Max steps reached at step {step}.")
            truncated = True

    # 回合结束后绘制Action曲线（可选）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(action_x, label='Action X', color='r')
    plt.plot(action_y, label='Action Y', color='g')
    plt.plot(action_z, label='Action Z', color='b')
    plt.xlabel('Step')
    plt.ylabel('Action Value')
    plt.title(f'Action Values for Episode {episode}')
    plt.legend()
    plt.show()
    """

    # 如需可视化pressure数据，可在此添加相应绘图代码（可选）
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pressure_x, label='Pressure X', color='r')
    plt.plot(pressure_y, label='Pressure Y', color='g')
    plt.plot(pressure_z, label='Pressure Z', color='b')
    plt.xlabel('Step')
    plt.ylabel('Pressure Value')
    plt.title(f'Pressure Values for Episode {episode}')
    plt.legend()
    plt.show()
    """

    print(f"Total Reward for Episode {episode}: {total_reward:.4f}")

# 关闭交互模式
plt.ioff()
plt.show()
