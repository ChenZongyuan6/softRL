# o1修改
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

        self.pressure_low = np.array([30, 40, 40])
        self.pressure_high = np.array([55, 65, 73])
        # self.target_position = np.array([420.0, 280.0, 218.0]) #[46, 59, 65]
        # self.target_position = np.array([422.06, 281.39, 217.75])  # [42, 52, 55] 太小 没明显反应
        # self.target_position = np.array([415.70, 281.45, 221.13])  # [35, 50, 65]
        # self.target_position = np.array([416.30, 281.46, 221.24])  # [35, 50, 65]
        self.target_position = np.array([-4.2325, -71.0387, 6.9697])  # [30, 60, 60]
        self.max_steps = 500
        self.reset()

    def step(self, action):
        self.current_pressure += action
        self.current_pressure = np.clip(self.current_pressure, self.pressure_low, self.pressure_high)

        with torch.no_grad():
            pressure_tensor = torch.tensor(self.current_pressure, dtype=torch.float32).unsqueeze(0).to(self.device)
            new_state_tensor = self.nn_model(pressure_tensor)
        new_state = new_state_tensor.cpu().numpy().flatten()#如果你的模型在 GPU 上进行推理，生成的 new_state_tensor 可能位于 GPU 内存中。为了与 NumPy 兼容，需要将其移回 CPU。
        # pressure_tensor = torch.tensor(self.current_pressure, dtype=torch.float32).unsqueeze(0)
        # new_state_tensor = self.nn_model(pressure_tensor)
        # new_state = new_state_tensor.detach().numpy().flatten()

        end_effector_position = new_state[-3:]
        distance_to_target = np.linalg.norm(end_effector_position - self.target_position)

        epsilon = 1e-5
        # distance_reward = -np.log(distance_to_target + epsilon) / np.log(200)
        # distance_reward = -distance_to_target
        distance_reward = np.exp(-distance_to_target)

        close_to_target_reward = 0
        # if distance_to_target < 1:
        #     close_to_target_reward = 2  # 给予额外奖励

        termination_penalty = 0
        self.steps += 1
        if self.steps >= self.max_steps and distance_to_target >= 1:
            termination_penalty = -1

        total_reward = distance_reward + close_to_target_reward + termination_penalty

        # done = distance_to_target < 1 or self.steps >= self.max_steps
        # done = self.steps >= self.max_steps or distance_to_target > 20
        done = False
        truncated = self.steps >= self.max_steps
        # 返回信息
        info = {
            'distance_reward': distance_reward,
            'close_to_target_reward': close_to_target_reward,
            'termination_penalty': termination_penalty,
            'distance_to_target': distance_to_target,
            'state': new_state,
            'pressure': self.current_pressure
        }

        self.current_state = new_state

        # return new_state, total_reward, done, info
        return new_state, total_reward, done, truncated, info  # 返回五个值

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # 调用父类的 reset 方法，设置种子
        self.current_state = np.zeros(15)
        self.current_pressure = np.array([30.0, 40.0, 40.0])  # 确保在 pressure_low 范围内
        self.steps = 0
        return self.current_state, {}  # 返回 (observation, info)
    # def reset(self):
    #
    #     self.current_state = np.zeros(15)
    #     self.current_pressure = np.array([25.0, 25.0, 30.0])
    #     self.steps = 0
    #     return self.current_state
