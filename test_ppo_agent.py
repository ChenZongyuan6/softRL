import torch
from stable_baselines3 import PPO
from nn_environment import NNEnvironment  # 导入自定义环境
from nn_model import PressureToPositionNet  # 导入自定义模型类

# 加载已训练好的神经网络模型
nn_model = PressureToPositionNet()
nn_model.load_state_dict(torch.load('trained_model.pth'))  # 加载训练好的权重
nn_model.eval()  # 设置为评估模式

# 创建自定义环境实例
env = NNEnvironment(nn_model)

# 加载训练好的 PPO 代理模型
model = PPO.load('ppo_trained_agent')

# 测试 PPO 代理，进行多个回合的测试
num_episodes = 1  # 设定测试的回合数
for episode in range(num_episodes):
    obs = env.reset()  # 重置环境，开始新的 episode
    total_reward = 0  # 累计奖励
    for i in range(1000):  # 每个 episode 运行最多 1000 个时间步
        action, _states = model.predict(obs)  # 使用 PPO 代理预测动作
        obs, rewards, done, info = env.step(action)  # 执行动作，并返回新状态、奖励等信息
        total_reward += rewards  # 累计该 episode 的奖励

        # 打印当前时间步的末端执行器位置和奖励
        print(f"Episode {episode}, Step {i}, End effector position: {obs[-3:]}, Reward: {rewards}")

        if done:  # 如果 episode 结束（done=True），跳出循环
            print(f"Episode {episode} complete! Total reward: {total_reward}")
            break
