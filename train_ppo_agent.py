# train_ppo_agent.py

# from stable_baselines3 import PPO
# import torch
# from nn_environment import NNEnvironment
# from nn_model import PressureToPositionNet  # 确保导入正确的模型类
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from nn_environment import NNEnvironment
from nn_model import PressureToPositionNet  # 导入自定义模型类
from stable_baselines3.common.callbacks import BaseCallback

# # 创建模型架构
# nn_model = PressureToPositionNet()
# # 加载预训练的权重
# nn_model.load_state_dict(torch.load('trained_model.pth'))
# # 将模型设置为评估模式，避免训练时的不必要行为（如 Dropout）
# nn_model.eval()
# # 创建自定义环境实例
# env = NNEnvironment(nn_model)
#
# # 创建 PPO 代理并训练
# model = PPO('MlpPolicy', env, verbose=1)
# # 训练 PPO 代理
# model.learn(total_timesteps=100000)
#
# # 保存训练好的 PPO 模型
# model.save("ppo_trained_agent")



# 创建模型架构
nn_model = PressureToPositionNet()
# 加载预训练的权重
nn_model.load_state_dict(torch.load('trained_model.pth'))
# 将模型设置为评估模式
nn_model.eval()
# 创建自定义环境实例
env = NNEnvironment(nn_model)


# 自定义回调函数，记录损失值和奖励
class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.episode_rewards = []  # 用于记录每个回合的累计奖励
        self.losses = []  # 用于记录损失值

    def _on_step(self) -> bool:
        # 记录奖励
        if self.locals.get('rewards') is not None:
            reward = self.locals['rewards']
            self.episode_rewards.append(reward)

        # 记录损失值
        if 'loss' in self.locals:
            self.losses.append(self.locals['loss'])

        return True

    def _on_rollout_end(self):
        # 回合结束时，记录累计奖励
        total_rewards = sum(self.locals['rewards'])
        self.episode_rewards.append(total_rewards)


# 创建 PPO 代理并训练
model = PPO('MlpPolicy', env, verbose=1)
# 创建回调对象
callback = CustomCallback()
# 训练 PPO 代理，注意：根据问题难度和时间需求调整时间步
model.learn(total_timesteps=200000, callback=callback)
# 保存训练好的 PPO 模型
model.save("ppo_trained_agent_101402_100-d")

# === 训练结束后绘制奖励和损失值曲线 ===

# 1. 绘制累计奖励曲线
plt.figure(figsize=(10, 6))
plt.plot(callback.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress: Total Reward per Episode')
plt.show()

# 2. 绘制损失值曲线（如果有记录到）
if callback.losses:
    plt.figure(figsize=(10, 6))
    plt.plot(callback.losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Curve during Training')
    plt.show()
else:
    print("No losses were recorded during training.")






