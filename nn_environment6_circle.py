import gymnasium as gym
import numpy as np
import torch

class NNEnvironment(gym.Env):
    def __init__(self, nn_model, device='cpu'):
        super(NNEnvironment, self).__init__()

        self.action_space = gym.spaces.Box(low=-3, high=3, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.nn_model = nn_model.to(device)
        self.nn_model.eval()
        self.device = device

        # 气壓限制
        self.pressure_low = np.array([30, 40, 40])
        self.pressure_high = np.array([55, 65, 73])

        # 轨迹参数设置
        self.radius = 6.0        # 圆的半径
        self.center_x = 0.0       # 圆心X坐标
        self.center_y = -73.0     # 圆心Y坐标
        self.center_z = 0.0       # 圆心Z坐标（可以根据需要调整）
        self.omega = 0.002         # 角速度（弧度/step），可根据需要进行调整

        self.max_steps = 500

        self.reset()

    def _get_target_position(self, t):
        # 根据时间步t计算当前目标位置
        x = self.center_x + self.radius * np.cos(self.omega * t)
        y = self.center_y + self.radius * np.sin(self.omega * t)
        z = self.center_z
        return np.array([x, y, z], dtype=np.float32)

    def step(self, action):
        # 动作执行：更新气压
        self.current_pressure += action
        self.current_pressure = np.clip(self.current_pressure, self.pressure_low, self.pressure_high)

        # 使用神经网络预测新的末端点位置
        with torch.no_grad():
            pressure_tensor = torch.tensor(self.current_pressure, dtype=torch.float32).unsqueeze(0).to(self.device)
            new_state_tensor = self.nn_model(pressure_tensor)
        new_state = new_state_tensor.cpu().numpy().flatten()

        # 提取末端点位置
        end_effector_position = new_state[-3:]

        # 更新时间步
        self.t += 1
        # 根据当前时间步计算新目标点
        self.target_position = self._get_target_position(self.t)

        # 计算与目标点的距离
        distance_to_target = np.linalg.norm(end_effector_position - self.target_position)

        # 奖励函数：使用指数衰减作为距离奖励
        distance_reward = np.exp(-distance_to_target)

        # 这里可以加入其他奖励成分，例如接近目标的额外奖励等
        close_to_target_reward = 0
        termination_penalty = 0

        self.steps += 1
        if self.steps >= self.max_steps and distance_to_target > 2.0:
            # 如果超过最大步数且仍然离目标较远，可以给一个负奖励作为结束
            termination_penalty = -1

        total_reward = distance_reward + close_to_target_reward + termination_penalty

        # 是否结束
        done = False
        truncated = self.steps >= self.max_steps

        info = {
            'distance_reward': distance_reward,
            'close_to_target_reward': close_to_target_reward,
            'termination_penalty': termination_penalty,
            'distance_to_target': distance_to_target,
            'state': new_state,
            'pressure': self.current_pressure,
            'current_target': self.target_position
        }

        self.current_state = new_state

        return new_state, total_reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # 重置状态与气压
        self.current_state = np.zeros(15)
        self.current_pressure = np.array([30.0, 40.0, 40.0])
        self.steps = 0

        # 时间从0开始
        self.t = 0
        # 初始化目标位置为初始时间步下的圆轨迹点
        self.target_position = self._get_target_position(self.t)

        return self.current_state, {}
