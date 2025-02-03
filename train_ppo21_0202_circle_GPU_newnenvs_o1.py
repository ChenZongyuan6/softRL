import torch
import numpy as np
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# === 改动：导入我们修改后的 NNEnvironment
from nn_environment17_circle_10point_GPU_o1 import NNEnvironment
from nn_training.nn_model import PressureToPositionNet  # 你的自定义模型类

# 全局变量：任务名称，便于修改路径
TASK_NAME = "35_ppo_0201_circle_1"

def create_env_fn(nn_model, device='cpu', env_index=0):
    """
    创建画圆环境的函数。
    改动：将已经加载好的 nn_model 直接传进来，不在此处重复加载。
    """
    env = NNEnvironment(
        nn_model=nn_model,
        device=device,
        max_steps=256,
        seed=42,
        env_index=env_index
    )
    return env


class CustomCallback(BaseCallback):
    """
    自定义回调：在多环境并行时，正确统计并记录每个环境的episode信息。
    """
    def __init__(self, verbose=0, save_path="ppo_trained_agent_checkpoint", save_frequency=5):
        super(CustomCallback, self).__init__(verbose)
        self.save_path = save_path
        self.save_frequency = save_frequency

        # 训练过程日志
        self.writer = None
        # Rollout计数器
        self.rollout_count = 0
        # 每个环境单独的统计
        self.episode_rewards = None   # shape = (n_envs,)
        self.episode_lengths = None   # shape = (n_envs,)

        # 已完成的episode计数(用于 x 轴递增)
        self.episodes_done = 0

    def _init_callback(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(f'runs/{TASK_NAME}')

        vec_env = self.model.get_env()
        self.n_envs = vec_env.num_envs

        self.episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.n_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards')  # shape = (n_envs,)
        dones = self.locals.get('dones')      # shape = (n_envs,)
        infos = self.locals.get('infos', [{}] * self.n_envs)

        for i in range(self.n_envs):
            self.episode_rewards[i] += rewards[i]
            self.episode_lengths[i] += 1

        ended_envs = [i for i, done in enumerate(dones) if done]

        if len(ended_envs) > 0:
            ended_rewards = []
            ended_lengths = []
            ended_distances = []

            for i in ended_envs:
                ended_rewards.append(self.episode_rewards[i])
                ended_lengths.append(self.episode_lengths[i])
                distance_to_target = infos[i].get('distance_to_target', None)
                if distance_to_target is not None:
                    ended_distances.append(distance_to_target)
                else:
                    ended_distances.append(np.nan)

            mean_reward = np.mean(ended_rewards)
            mean_length = np.mean(ended_lengths)
            mean_distance = np.mean(ended_distances)

            self.writer.add_scalar('Rewards/Total', mean_reward, self.episodes_done)
            self.writer.add_scalar('Episode Length', mean_length, self.episodes_done)
            self.writer.add_scalar('Distance_to_Target/Episode', mean_distance, self.episodes_done)

            if (self.episodes_done > 0) and (self.episodes_done % self.save_frequency == 0):
                save_path_with_episode = f"{self.save_path}_ep{self.episodes_done}.zip"
                self.model.save(save_path_with_episode)
                if self.verbose > 0:
                    print(f"Model saved at episode {self.episodes_done} to {save_path_with_episode}")

            self.episodes_done += len(ended_envs)

            for i in ended_envs:
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        return True

    def _on_rollout_end(self):
        logs = self.model.logger.name_to_value
        policy_loss = logs.get('train/policy_gradient_loss', 0)
        value_loss = logs.get('train/value_loss', 0)
        entropy_loss = logs.get('train/entropy_loss', 0)
        kl_divergence = logs.get('train/approx_kl', 0)
        explained_variance = logs.get('train/explained_variance', 0)

        self.writer.add_scalar('Loss/Policy_Loss', policy_loss, self.rollout_count)
        self.writer.add_scalar('Loss/Value_Loss', value_loss, self.rollout_count)
        self.writer.add_scalar('Loss/Entropy', -entropy_loss, self.rollout_count)
        self.writer.add_scalar('KL_Divergence', kl_divergence, self.rollout_count)
        self.writer.add_scalar('Explained_Variance', explained_variance, self.rollout_count)

        self.writer.add_scalar('Loss/Policy_Loss_t', policy_loss, self.num_timesteps)
        self.writer.add_scalar('Loss/Value_Loss_t', value_loss, self.num_timesteps)
        self.writer.add_scalar('Loss/Entropy_t', -entropy_loss, self.num_timesteps)
        self.writer.add_scalar('KL_Divergence_t', kl_divergence, self.num_timesteps)
        self.writer.add_scalar('Explained_Variance_t', explained_variance, self.num_timesteps)

        self.rollout_count += 1

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 改动：在主进程中只加载一次模型
    global_nn_model = PressureToPositionNet()
    global_nn_model.load_state_dict(
        torch.load('nn_models/35_trained_nn_model_202501.pth', map_location=device)
    )
    global_nn_model.to(device)
    global_nn_model.eval()

    # 并行环境数量
    num_envs = 48

    # === 改动：把全局模型 global_nn_model 传入 create_env_fn
    def make_env(i):
        return create_env_fn(nn_model=global_nn_model, device=device, env_index=i)

    # 创建多个并行环境
    env = SubprocVecEnv([lambda i=i: make_env(i) for i in range(num_envs)])

    model = PPO(
        'MlpPolicy',
        env,
        n_steps=4096,
        batch_size=16384,
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        device=device
    )

    callback = CustomCallback(
        save_path=f"rl_checkpoints/{TASK_NAME}/checkpoints",
        save_frequency=50
    )

    model.learn(total_timesteps=100_000_000, callback=callback)

    model.save(f"./rl_agent/{TASK_NAME}")
