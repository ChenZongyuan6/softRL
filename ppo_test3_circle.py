# 测试PPO效果并简单可视化，用于gymnasium环境，展示已训练好的policy对圆形轨迹的跟踪情况

import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from nn_environment6_circle import NNEnvironment  # 需确保修改后的环境代码保存在 nn_environment4.py 中
from src.nn_model import PressureToPositionNet
from scipy.interpolate import splprep, splev #B样条曲线库


# 加载训练好的模型
model = PPO.load("ppo_1211_circle_2.zip")

# 创建预训练神经网络模型实例并加载权重
nn_model = PressureToPositionNet()
nn_model.load_state_dict(torch.load('nn_models/trained_nn_model10_swgt_fixRseed_centered.pth'))
nn_model.eval()

# 创建环境实例（使用修改后的NNEnvironment，包含圆轨迹逻辑）
env = NNEnvironment(nn_model)

# 初始化3D图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # 开启交互模式

# 用于存储轨迹的列表
end_effector_positions = []
target_positions = []

num_episodes = 1  # 测试2个回合
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step = 0

    # 清空上一个回合的轨迹数据
    end_effector_positions.clear()
    target_positions.clear()

    print(f"==== Episode {episode + 1} ====")

    while not (done or truncated):
        # 使用训练好的模型预测动作
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # 从info中提取末端点位置和目标位置
        end_effector_pos = info.get('state')[-3:]
        current_target = info.get('current_target', None)

        if current_target is not None and end_effector_pos is not None:
            end_effector_positions.append(end_effector_pos)
            target_positions.append(current_target)

            # 清空并更新绘图
            ax.clear()
            ee_array = np.array(end_effector_positions)
            tgt_array = np.array(target_positions)

            # 绘制目标轨迹(红色)与末端点轨迹(蓝色)
            if len(tgt_array) > 1:
                ax.plot(tgt_array[:,0], tgt_array[:,1], tgt_array[:,2], 'r-', label='Target Trajectory')
            if len(ee_array) > 1:
                ax.plot(ee_array[:,0], ee_array[:,1], ee_array[:,2], 'b-', label='End Effector Trajectory')

            # 标记当前点
            ax.scatter(tgt_array[-1,0], tgt_array[-1,1], tgt_array[-1,2], c='r', marker='x', s=50, label='Current Target')
            ax.scatter(ee_array[-1,0], ee_array[-1,1], ee_array[-1,2], c='b', marker='o', s=50, label='Current End Effector')

            # 设置坐标轴范围（根据实验需要可调整）
            ax.set_xlim(-80, 20)
            ax.set_ylim(-110, -30)
            ax.set_zlim(-10, 10)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Episode {episode+1}, Step {step}')
            ax.legend()

            plt.draw()
            plt.pause(0.001)

        # # 如果你想在达到特定精度时终止测试
        # distance_to_target = info.get('distance_to_target', None)
        # if distance_to_target is not None and distance_to_target < 0.2:
        #     print(f"Target reached at step {step}. Distance to target: {distance_to_target}")
        #     done = True

    print(f"Total Reward for Episode {episode + 1}: {total_reward}")

plt.ioff()
plt.show()
