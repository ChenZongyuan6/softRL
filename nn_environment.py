# nn_environment.py

import gym
import numpy as np
import torch

class NNEnvironment(gym.Env):
    def __init__(self, nn_model):
        super(NNEnvironment, self).__init__()

        # 定义动作空间和状态空间
        self.action_space = gym.spaces.Box(low=np.array([30, 40, 40]), high=np.array([55, 65, 73]), dtype=np.float32) # 气压动作空间
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)  # 位置状态空间

        self.nn_model = nn_model  # 训练好的神经网络模型
        self.current_state = self.reset()

        self.target_position = np.array([420.0, 280.0, 218.0])  # 末端点的目标位置

    def step(self, action):
        # 使用 PPO 代理产生的动作（气压值）计算新的位置状态
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        new_state_tensor = self.nn_model(action_tensor)
        new_state = new_state_tensor.detach().numpy().flatten()

        # 取末端点的坐标，即 new_state 的最后3个维度
        end_effector_position = new_state[-3:]
        # 计算末端点与目标位置的距离
        distance_to_target = np.linalg.norm(end_effector_position - self.target_position)
        # 奖励：距离越小，奖励越大
        reward = 100-distance_to_target  # 使用负的距离作为奖励，越接近目标，奖励越大

        # 定义一个终止条件（例如，时间步超过一定值时终止）
        done = distance_to_target < 1  # 如果末端点距离目标小于0.01，则任务成功，终止
        done = done or self.current_state[-3] >= 1000  # 限制时间步的最大数量

        self.current_state = new_state
        return new_state, reward, done, {}

    def reset(self):
        self.current_state = np.zeros(15)  # 初始化为零状态或其他合理值
        return self.current_state

