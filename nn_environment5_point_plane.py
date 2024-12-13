#
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

        # 定义目标交点的位置，注意，这里是原始坐标，xzy分别对应正常坐标的YZX
        self.target_position = np.array([-5.3201,  -78.6103, -8.5773])

        # 定义平面的法向量（人为设定，不再动态计算）
        # 例如：在 XY 平面内倾斜 45 度的竖直平面
        self.plane_normal = np.array([0.1, 1, 0.1])  # 法向量指向 XY 平面内的方向
        # 定义平面上的一点
        self.plane_point = np.array([0, -80, 0])  # 可选的平面上的某个固定点

        self.max_steps = 300
        self.reset()

    def calculate_intersection(self, point4, point5, normal, plane_point):
        """
        计算点4和点5连线与给定竖直平面的交点
        :param point4: 点4的坐标
        :param point5: 点5的坐标
        :param normal: 平面的法向量
        :param plane_point: 平面上的一个点
        :return: 交点坐标或 None
        """
        x4, y4, z4 = point4
        x5, y5, z5 = point5

        # 计算连线的方向向量
        line_direction = np.array([x5 - x4, y5 - y4, z5 - z4])
        line_point = np.array([x4, y4, z4])

        # 如果方向向量和法向量垂直，则连线与平面平行，没有交点
        dot_product = np.dot(line_direction, normal)
        if abs(dot_product) < 1e-6:  # 判断是否平行
            return None

        # 计算交点的t 参数
        t = np.dot(normal, (plane_point - line_point)) / dot_product
        if t < 0:  # 只考虑延长线上的交点
            return None

        # 计算交点
        intersection = line_point + t * line_direction
        return intersection

    def step(self, action):
        # 更新气压
        self.current_pressure += action
        self.current_pressure = np.clip(self.current_pressure, self.pressure_low, self.pressure_high)

        # 使用神经网络预测新的末端点位置
        pressure_tensor = torch.tensor(self.current_pressure, dtype=torch.float32).unsqueeze(0)
        new_state_tensor = self.nn_model(pressure_tensor)
        new_state = new_state_tensor.detach().numpy().flatten()

        # 提取5个点的坐标
        points = new_state.reshape(5, 3)
        point4 = points[3]
        point5 = points[4]

        # 使用固定的法向量和平面上的一点来计算交点
        intersection = self.calculate_intersection(point4, point5, self.plane_normal, self.plane_point)

        # 处理交点为 None 的情况（即射线和平面平行）
        if intersection is None:
            distance_to_target = float('inf')
        else:
            # 计算交点与目标位置的距离
            distance_to_target = np.linalg.norm(intersection - self.target_position)

        # 计算奖励
        distance_reward = np.exp(-distance_to_target)

        close_to_target_reward = 0
        # if distance_to_target < 1:
        #     close_to_target_reward = 2  # 给予额外奖励
        termination_penalty = 0
        self.steps += 1
        if self.steps >= self.max_steps and distance_to_target >= 0.5:
            termination_penalty = -1

        total_reward = distance_reward + close_to_target_reward + termination_penalty

        # 判断是否结束
        done = False
        truncated = self.steps >= self.max_steps
        # 返回信息
        info = {
            'distance_reward': distance_reward,
            'close_to_target_reward': close_to_target_reward,
            'termination_penalty': termination_penalty,
            'distance_to_target': distance_to_target,
            'state': new_state,
            'pressure': self.current_pressure,
            'intersection': intersection
        }

        self.current_state = new_state

        # return new_state, total_reward, done, info
        return new_state, total_reward, done, truncated, info  # 返回五个值

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # 调用父类的 reset 方法，设置种子
        self.current_state = np.zeros(15)
        self.current_pressure = np.array([30.0, 40.0, 40.0])
        self.steps = 0
        return self.current_state, {}  # 返回 (observation, info)
