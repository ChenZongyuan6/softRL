import gymnasium as gym
import numpy as np
import torch

class NNEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        nn_model,
        device='cpu',
        max_steps=512,
        seed=42,
        env_index=0
    ):
        super(NNEnvironment, self).__init__()

        # ============== 动作空间 ==============
        self.action_space = gym.spaces.Box(
            low=-3,
            high=3,
            shape=(3,),
            dtype=np.float32
        )

        # ============== 环形轨迹参数 ==============
        self.radius = 6.0         # 圆的半径
        self.center_x = 0.0       # 圆心 X
        self.center_y = -70.5     # 圆心 Y
        self.center_z = 0.0       # 圆心 Z
        self.omega = 0.2          # 角速度 (弧度 / step)

        # ============== 模型、设备等 ==============
        self.nn_model = nn_model.to(device)
        self.nn_model.eval()
        self.device = device

        # ============== 压力限制 ==============
        self.pressure_low = np.array([30, 40, 40], dtype=np.float32)
        self.pressure_high = np.array([55, 65, 73], dtype=np.float32)

        # ============== 观测空间修改 ==============
        # 仍是 48 维 (15关键点 + 3压力 + 10个相对目标点*3=30)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(48,),
            dtype=np.float32
        )

        # ============== 其它属性 ==============
        self.max_steps = max_steps
        self.env_index = env_index

        # 上一步的距离（可用于额外奖励）
        self.previous_distance_to_target = None

        # 设置随机种子
        self.seed(seed)

        self.reset()

    def _get_target_position(self, t: int) -> np.ndarray:
        """
        计算时间步 t 对应的圆轨迹点 (x, y, z).
        """
        x = self.center_x + self.radius * np.cos(self.omega * t)
        y = self.center_y
        z = self.center_z + self.radius * np.sin(self.omega * t)
        return np.array([x, y, z], dtype=np.float32)

    def _get_future_trajectory(self, t_start: int, end_effector_position: np.ndarray, horizon=10):
        """
        返回从 t_start 到 t_start + horizon-1 的轨迹点，每个点 3 维，共 horizon 个点。
        坐标相对于当前末端执行器位置 (end_effector_position)。
        输出 shape = (3*horizon,).
        """
        points = []
        for i in range(horizon):
            # 计算绝对坐标
            abs_pos = self._get_target_position(t_start + i)
            # 转换为相对坐标
            rel_pos = abs_pos - end_effector_position
            points.append(rel_pos)
        # 拼成 shape=(10,3) 再 flatten => (30,)
        points = np.array(points).reshape(-1)  # (30,)
        return points

    def step(self, action: np.ndarray):
        # 1) 更新压力
        self.current_pressure += action
        self.current_pressure = np.clip(
            self.current_pressure,
            self.pressure_low,
            self.pressure_high
        )

        # 2) 使用神经网络预测末端关键点
        with torch.no_grad():
            pressure_tensor = torch.tensor(
                self.current_pressure,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            new_state_tensor = self.nn_model(pressure_tensor)
        keypoints = new_state_tensor.cpu().numpy().flatten()  # 15维

        # 3) 计算当前目标点(时间步 t 对应)
        current_target = self._get_target_position(self.t)
        end_effector_position = keypoints[-3:]
        distance_to_target = np.linalg.norm(end_effector_position - current_target)

        # 4) 计算奖励
        # 使用指数衰减做距离奖励
        distance_reward = np.exp(-distance_to_target)

        improvement_reward = 0.0
        if self.previous_distance_to_target is not None:
            improvement_reward = np.exp(self.previous_distance_to_target - distance_to_target)
        self.previous_distance_to_target = distance_to_target

        # 合并奖励
        reward = distance_reward
                 #+ 0.1 * improvement_reward  # 根据需要启用

        # 5) 每个 step 都自动推进时间 t+1
        self.t += 1

        # 6) 终止判断
        self.steps += 1
        truncated = (self.steps >= self.max_steps)
        done = False  # 根据需要修改

        # 7) 组装 Observation
        # 7.1) keypoints(15 维)
        # 7.2) 当前压力(3 维)
        # 7.3) 未来10个轨迹点相对于当前末端执行器位置 => shape=(30,)
        future_points = self._get_future_trajectory(self.t, end_effector_position, horizon=10)  # shape=(30,)

        # 把它们拼起来 => 15 + 3 + 30 = 48 维
        observation = np.concatenate([
            keypoints,
            self.current_pressure,
            future_points
        ])  # shape=(48,)

        # 8) 组装 info
        info = {
            'distance_to_target': distance_to_target,
            'state_keypoints': keypoints,
            'pressure': self.current_pressure,
            'target_position': current_target,
            't': self.t
        }

        return observation, reward, done, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # 1) 重置气压、步数、时间步 t
        self.current_pressure = np.array([30.0, 40.0, 40.0], dtype=np.float32)
        self.steps = 0
        self.t = 0
        self.previous_distance_to_target = None

        # 2) 用当前压力预测初始关键点
        with torch.no_grad():
            pressure_tensor = torch.tensor(
                self.current_pressure,
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            init_state_tensor = self.nn_model(pressure_tensor)
        keypoints = init_state_tensor.cpu().numpy().flatten()  # 15维

        # 3) 计算末端执行器位置
        end_effector_position = keypoints[-3:]

        # 4) 拼装观察量 (未来10个点相对于当前末端执行器位置)
        future_points = self._get_future_trajectory(self.t, end_effector_position, horizon=10)  # shape=(30,)
        observation = np.concatenate([
            keypoints,
            self.current_pressure,
            future_points
        ])  # shape=(48,)

        # 5) 组装 info
        info = {
            'distance_to_target': None,
            'state_keypoints': keypoints,
            'pressure': self.current_pressure,
            'target_position': self._get_target_position(self.t),
            't': self.t
        }
        return observation, info

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
