#o1修改
import gym
import numpy as np
import torch

class NNEnvironment(gym.Env):
    def __init__(self, nn_model):
        super(NNEnvironment, self).__init__()

        self.action_space = gym.spaces.Box(low=-5, high=5, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self.nn_model = nn_model
        self.pressure_low = np.array([30, 40, 40])
        self.pressure_high = np.array([55, 65, 73])
        self.target_position = np.array([420.0, 280.0, 218.0])

        self.max_steps = 200
        self.reset()

    def step(self, action):
        self.current_pressure += action
        self.current_pressure = np.clip(self.current_pressure, self.pressure_low, self.pressure_high)

        pressure_tensor = torch.tensor(self.current_pressure, dtype=torch.float32).unsqueeze(0)
        new_state_tensor = self.nn_model(pressure_tensor)
        new_state = new_state_tensor.detach().numpy().flatten()

        end_effector_position = new_state[-3:]
        distance_to_target = np.linalg.norm(end_effector_position - self.target_position)

        epsilon = 1e-5
        distance_reward = -np.log(distance_to_target + epsilon) / np.log(100)

        close_to_target_reward = 0
        if distance_to_target < 5:
            close_to_target_reward = 0  # 给予额外奖励

        termination_penalty = 0
        self.steps += 1
        if self.steps >= self.max_steps and distance_to_target >= 1:
            termination_penalty = -1000

        total_reward = distance_reward + close_to_target_reward + termination_penalty

        done = distance_to_target < 1 or self.steps >= self.max_steps

        info = {
            'distance_reward': distance_reward,
            'close_to_target_reward': close_to_target_reward,
            'termination_penalty': termination_penalty,
            'distance_to_target': distance_to_target,
            'state': new_state,
            'pressure': self.current_pressure
        }

        self.current_state = new_state

        return new_state, total_reward, done, info

    def reset(self):
        self.current_state = np.zeros(15)
        self.current_pressure = np.array([0.0, 0.0, 0.0])
        self.steps = 0
        return self.current_state
