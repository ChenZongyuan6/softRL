# -----------------------
# 适配“画圆”环境的训练代码 以及“画方形
# -----------------------
import torch
import numpy as np
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
# from nn_environment10_circle import NNEnvironment
# from nn_environment11_square import NNEnvironment
# from nn_environment12_circle_10point_obs import NNEnvironment
# from nn_environment13_circle_10point_obs_relpos import NNEnvironment
from nn_environment14_circle_10point_norm_obs import NNEnvironment
from nn_training.nn_model import PressureToPositionNet  # 导入自定义模型类
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
# 全局变量：任务名称，便于修改路径
TASK_NAME = "30_ppo_0116_circlenew_2"

def create_env_fn(device='cpu', env_index=0):
    """
    创建画圆环境的函数。
    """
    nn_model = PressureToPositionNet()
    nn_model.load_state_dict(
        # torch.load('nn_models/trained_nn_model10_swgt_fixRseed_centered.pth', map_location=device) #第一批0322数据训练
        torch.load('nn_models/30_trained_nn_model_202412_4.pth', map_location=device) #第二批202412数据训练
    )
    nn_model.to(device)
    nn_model.eval()

    # 使用新环境：circle 环境
    env = NNEnvironment(nn_model, device=device, max_steps=256, seed=42, env_index=env_index)  # 增加 env_index
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
        """
        stable-baselines3 会在调用 model.learn() 时先调用此函数来初始化 Callback。
        """
        # 如果没有创建 writer，则在这里创建
        if self.writer is None:
            # self.writer = SummaryWriter('runs/ppo_1222_circle_8')
            self.writer = SummaryWriter(f'runs/{TASK_NAME}')

        # 获取当前并行环境数量 n_envs
        vec_env = self.model.get_env()
        self.n_envs = vec_env.num_envs

        # 为每个环境初始化统计容器
        self.episode_rewards = np.zeros(self.n_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.n_envs, dtype=np.int32)

    def _on_training_start(self) -> None:
        """
        当训练开始时调用（比 _init_callback 略晚，通常这里也可以做一些检查）。
        """
        pass

    def _on_step(self) -> bool:
        """
        这个函数在每个 time step（环境执行一次动作）后被调用。
        这里做并行环境的 episode 统计，以及写 TensorBoard。
        """
        # 这几个对象由 stable-baselines3 在训练循环中注入
        rewards = self.locals.get('rewards')  # shape = (n_envs,)
        dones = self.locals.get('dones')      # shape = (n_envs,)
        infos = self.locals.get('infos', [{}] * self.n_envs)  # 列表，每个env都有一个info字典

        # 1) 先累加每个环境的 reward 和 step
        for i in range(self.n_envs):
            self.episode_rewards[i] += rewards[i]
            self.episode_lengths[i] += 1

        # 2) 找出本 step 中哪些环境结束了
        ended_envs = [i for i, done in enumerate(dones) if done]

        # 3) 如果有环境结束，就做统计和日志
        if len(ended_envs) > 0:
            # 先把结束环境的 episode reward / length / distance 收集起来
            ended_rewards = []
            ended_lengths = []
            ended_distances = []

            for i in ended_envs:
                ended_rewards.append(self.episode_rewards[i])
                ended_lengths.append(self.episode_lengths[i])

                # 从 info 中取 distance_to_target
                distance_to_target = infos[i].get('distance_to_target', None)
                if distance_to_target is not None:
                    ended_distances.append(distance_to_target)
                else:
                    ended_distances.append(np.nan)  # 如果没有则用 nan

            # 计算这些结束 episode 的平均值
            mean_reward = np.mean(ended_rewards)
            mean_length = np.mean(ended_lengths)
            mean_distance = np.mean(ended_distances)

            # 写入 TensorBoard (episode 级别的统计)
            # 注意：将 episodes_done 作为 x 轴，这样每次加 len(ended_envs) 次
            # 会保证 x 轴是随episode数量增加
            self.writer.add_scalar('Rewards/Total', mean_reward, self.episodes_done)
            self.writer.add_scalar('Episode Length', mean_length, self.episodes_done)
            self.writer.add_scalar('Distance_to_Target/Episode', mean_distance, self.episodes_done)

            # 如果需要保存模型，可以参考下面逻辑
            if (self.episodes_done > 0) and (self.episodes_done % self.save_frequency == 0):
                save_path_with_episode = f"{self.save_path}_ep{self.episodes_done}.zip"
                self.model.save(save_path_with_episode)
                if self.verbose > 0:
                    print(f"Model saved at episode {self.episodes_done} to {save_path_with_episode}")

            # episode 计数增加
            self.episodes_done += len(ended_envs)

            # 4) 重置结束环境的统计，为下一个回合做准备
            for i in ended_envs:
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0

        return True

    def _on_rollout_end(self):
        """
        一个 rollout 结束后（也就是 PPO 收集完 n_steps 训练数据后，进入训练时），会调用一次。
        可以在此处记录一些训练相关的损失或者其他指标。
        """
        logs = self.model.logger.name_to_value
        # 提取训练动态指标
        policy_loss = logs.get('train/policy_gradient_loss', 0)
        value_loss = logs.get('train/value_loss', 0)
        entropy_loss = logs.get('train/entropy_loss', 0)
        kl_divergence = logs.get('train/approx_kl', 0)
        explained_variance = logs.get('train/explained_variance', 0)

        # 使用 rollout_count 作为 X 轴
        self.writer.add_scalar('Loss/Policy_Loss', policy_loss, self.rollout_count)
        self.writer.add_scalar('Loss/Value_Loss', value_loss, self.rollout_count)
        self.writer.add_scalar('Loss/Entropy', -entropy_loss, self.rollout_count)  # 熵损失取负表示正熵值
        self.writer.add_scalar('KL_Divergence', kl_divergence, self.rollout_count)
        self.writer.add_scalar('Explained_Variance', explained_variance, self.rollout_count)

        # 可选：同时也以 self.num_timesteps 为 X 轴记录一份数据
        self.writer.add_scalar('Loss/Policy_Loss_t', policy_loss, self.num_timesteps)
        self.writer.add_scalar('Loss/Value_Loss_t', value_loss, self.num_timesteps)
        self.writer.add_scalar('Loss/Entropy_t', -entropy_loss, self.num_timesteps)
        self.writer.add_scalar('KL_Divergence_t', kl_divergence, self.num_timesteps)
        self.writer.add_scalar('Explained_Variance_t', explained_variance, self.num_timesteps)
        # Rollout计数器增加
        self.rollout_count += 1

    def _on_training_end(self) -> None:
        """
        训练完成后调用，用于清理资源，比如关闭 TensorBoard writer。
        """
        if self.writer is not None:
            self.writer.close()


if __name__ == '__main__':
    # 在 Windows 上进行多进程，需要使用 spawn
    mp.set_start_method('spawn', force=True)

    # 选择设备
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 并行环境数量
    num_envs = 48

    # 创建多个并行环境
    # env = SubprocVecEnv([lambda: create_env_fn(device=device) for _ in range(num_envs)])

    # 创建多个并行环境，传入 env_index
    env = SubprocVecEnv([
        lambda i=i: create_env_fn(device='cpu', env_index=i) for i in range(num_envs)
    ])
    # 初始化 PPO 模型
    model = PPO(
        'MlpPolicy',
        env,
        n_steps=4096,         # 每个环境收集多少步之后执行一次学习 #2048
        batch_size=16384,  #8192
        verbose=1,
        tensorboard_log="./ppo_tensorboard/",
        device=device
    )

    # 创建回调并设置保存路径和频率
    callback = CustomCallback(
        # save_path="rl_checkpoints/ppo_1222_circle_8/checkpoints",
        save_path=f"rl_checkpoints/{TASK_NAME}/checkpoints",
        save_frequency=50
    )

    # 开始训练
    model.learn(total_timesteps=100_000_000, callback=callback)
    # 保存最终模型
    # model.save("./rl_agent/ppo_1222_circle_8")
    model.save(f"./rl_agent/{TASK_NAME}")
