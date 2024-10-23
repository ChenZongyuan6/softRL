import torch
from stable_baselines3 import PPO
from nn_environment3 import NNEnvironment  # 使用更新后的环境代码
from nn_model import PressureToPositionNet  # 导入自定义模型类
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch.utils.tensorboard import SummaryWriter

# 加载预训练的神经网络模型
nn_model = PressureToPositionNet()
nn_model.load_state_dict(torch.load('trained_model.pth'))
nn_model.eval()

# 创建环境
env = DummyVecEnv([lambda: NNEnvironment(nn_model)])


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.writer = None
        self.episode_num = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.current_distance_reward = 0
        self.current_close_to_target_reward = 0
        self.current_termination_penalty = 0

        # 初始化列表以存储气压和末端点位置
        self.pressure_channel_1 = []
        self.pressure_channel_2 = []
        self.pressure_channel_3 = []
        self.end_effector_x = []
        self.end_effector_y = []
        self.end_effector_z = []

    def _on_training_start(self):
        if self.writer is None:
            # 使用模型的日志目录
            self.writer = SummaryWriter(log_dir=self.logger.dir)

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        if infos:
            distance_to_target = infos[0].get('distance_to_target')
            if distance_to_target is not None:
                self.writer.add_scalar('Distance_to_Target/Step', distance_to_target, self.num_timesteps)

            # 累积奖励组件
            self.current_distance_reward += infos[0].get('distance_reward', 0)
            self.current_close_to_target_reward += infos[0].get('close_to_target_reward', 0)
            self.current_termination_penalty += infos[0].get('termination_penalty', 0)

        rewards = self.locals.get('rewards')
        if rewards is not None:
            self.current_episode_reward += rewards[0]
            self.current_episode_length += 1

        dones = self.locals.get('dones')
        if dones is not None and dones[0]:
            # 当前回合编号
            episode_idx = self.episode_num

            # 记录当前回合的气压
            current_pressure = infos[0].get('pressure')
            if current_pressure is not None:
                self.pressure_channel_1.append(current_pressure[0])
                self.pressure_channel_2.append(current_pressure[1])
                self.pressure_channel_3.append(current_pressure[2])
                self.writer.add_scalar('Pressure/Channel_1', current_pressure[0], episode_idx)
                self.writer.add_scalar('Pressure/Channel_2', current_pressure[1], episode_idx)
                self.writer.add_scalar('Pressure/Channel_3', current_pressure[2], episode_idx)

            # 记录当前回合的末端点 XYZ 坐标
            end_effector_position = infos[0].get('state')[-3:]
            if end_effector_position is not None:
                self.end_effector_x.append(end_effector_position[0])
                self.end_effector_y.append(end_effector_position[1])
                self.end_effector_z.append(end_effector_position[2])
                self.writer.add_scalar('End_Effector/X', end_effector_position[0], episode_idx)
                self.writer.add_scalar('End_Effector/Y', end_effector_position[1], episode_idx)
                self.writer.add_scalar('End_Effector/Z', end_effector_position[2], episode_idx)

            # 将每回合的奖励和其他信息写入 TensorBoard
            self.writer.add_scalar('Rewards/Total', self.current_episode_reward, episode_idx)
            self.writer.add_scalar('Episode Length', self.current_episode_length, episode_idx)
            self.writer.add_scalar('Rewards/Distance', self.current_distance_reward, episode_idx)
            self.writer.add_scalar('Rewards/Close_to_Target', self.current_close_to_target_reward, episode_idx)
            self.writer.add_scalar('Rewards/Termination_Penalty', self.current_termination_penalty, episode_idx)

            # 重置当前回合的变量
            self.current_episode_reward = 0
            self.current_episode_length = 0
            self.current_distance_reward = 0
            self.current_close_to_target_reward = 0
            self.current_termination_penalty = 0
            self.episode_num += 1

        return True

    def _on_rollout_end(self):
        # 在每次策略更新后获取训练指标
        logs = self.model.logger.name_to_value

        # 提取需要的指标
        policy_loss = logs.get('train/policy_gradient_loss')
        value_loss = logs.get('train/value_loss')
        entropy_loss = logs.get('train/entropy_loss')
        kl_divergence = logs.get('train/approx_kl')
        explained_variance = logs.get('train/explained_variance')

        # 将指标记录到 TensorBoard
        if policy_loss is not None:
            self.writer.add_scalar('Loss/Policy_Loss', policy_loss, self.num_timesteps)
        if value_loss is not None:
            self.writer.add_scalar('Loss/Value_Loss', value_loss, self.num_timesteps)
        if entropy_loss is not None:
            self.writer.add_scalar('Entropy', -entropy_loss, self.num_timesteps)  # 熵损失取反以表示熵值
        if kl_divergence is not None:
            self.writer.add_scalar('KL_Divergence', kl_divergence, self.num_timesteps)
        if explained_variance is not None:
            self.writer.add_scalar('Explained_Variance', explained_variance, self.num_timesteps)

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.close()


# 创建 PPO 模型
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard/")
# 创建回调
callback = CustomCallback()
# 开始训练
model.learn(total_timesteps=400000, callback=callback)
# 保存模型
model.save("ppo_trained_agent_with_tensorboard")