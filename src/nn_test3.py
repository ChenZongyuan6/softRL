import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from nn_model import PressureToPositionNet  # 从 model.py 导入网络


# 读取测试数据
def load_test_data(pressure_file, position_file, batch_size=1):  # 改为每次处理一个样本
    pressure_data = pd.read_csv(pressure_file)
    position_data = pd.read_csv(position_file)

    # 假设气压数据在第3到5列（Python索引从0开始，所以是2到5）
    X_test = torch.tensor(pressure_data.iloc[:, 2:5].values, dtype=torch.float32)

    # 假设位置数据在第1到15列（Python索引从0开始，所以是0到15）
    y_test = torch.tensor(position_data.iloc[:, 0:15].values, dtype=torch.float32)

    # 创建数据集和数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 每个批次大小为1

    return test_loader, pressure_data

# 评估神经网络，记录每个批次的loss和output
def evaluate_network_and_record_output(model, test_loader, n=3):
    model.eval()  # 设置模型为评估模式
    criterion = torch.nn.MSELoss(reduction='none')  # 使用均方误差来评估性能（不进行平均以便逐个累积）
    batch_losses = []  # 用于记录每个批次的损失
    all_outputs = []  # 用于记录每个批次的预测值
    last_n_mse = []  # 用于记录最后n个点的均方误差
    total_loss = 0.0  # 用于记录总的均方误差
    total_samples = 0  # 用于记录样本数

    with torch.no_grad():  # 禁用梯度计算以加快评估
        for inputs, targets in test_loader:
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算当前批次的损失
            batch_losses.append(loss.mean().item())  # 记录损失
            all_outputs.append(outputs.squeeze().tolist())  # 记录每个批次的预测值，压缩维度

            # 累积总的均方误差
            total_loss += loss.sum().item()
            total_samples += targets.numel()

            # 计算最后n个点的均方误差
            mse_last_n = criterion(outputs[:, -n:], targets[:, -n:]).mean().item()
            last_n_mse.append(mse_last_n)

    # 计算所有样本的总均方误差
    total_mse = total_loss / total_samples

    return batch_losses, all_outputs, last_n_mse, total_mse



# 绘制每个批次的loss曲线
def plot_loss_curve(batch_losses):
    plt.figure()
    plt.plot(range(1, len(batch_losses) + 1), batch_losses, label="Batch Loss")
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Loss Per Batch')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(base_save_path, f'batch_loss_curve.png')
    plt.savefig(save_path)


# 绘制15个通道的真实值和预测值
def plot_outputs_with_targets(all_outputs, targets):
    targets = targets.numpy()  # 转换为numpy格式
    num_plots = 5  # 总共需要绘制5张图，每张图3个子图
    output_colors = ['g', 'g', 'g']  # 预测值的颜色
    target_colors = ['b', 'b', 'b']  # 真实值的颜色

    # 逐步绘制5张图，每张图3个子图
    for i in range(num_plots):
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # 创建3行1列的子图
        start_idx = i * 3  # 起始通道索引
        end_idx = start_idx + 3  # 结束通道索引

        for j, ax in enumerate(axes):  # 循环3个子图
            channel_idx = start_idx + j  # 当前通道的索引

            # 绘制当前通道的预测值和真实值，使用不同颜色
            ax.plot(range(len(all_outputs)), [output[channel_idx] for output in all_outputs], color=output_colors[j],
                    label=f"Output Channel {channel_idx + 1}", linestyle='-')
            ax.plot(range(len(targets)), targets[:, channel_idx], color=target_colors[j],
                    label=f"Target Channel {channel_idx + 1}", linestyle='-')

            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'Channel {channel_idx + 1}')
            ax.legend(loc='upper right')
            ax.grid(True)

        # 设置每张图的标题
        fig.suptitle(f'Channels {start_idx + 1} to {end_idx}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局避免重叠
        save_path = os.path.join(base_save_path, f'outputs_vs_targets_channels_{start_idx + 1}_to_{end_idx}.png')
        plt.savefig(save_path)  # 保存图像为文件


def plot_error(all_outputs, targets):
    targets = targets.numpy()  # 转换为numpy格式
    num_plots = 5  # 总共需要绘制5张图，每张图3个子图
    error_colors = ['r', 'r', 'r']  # 误差的颜色（红色表示误差）

    # 逐步绘制5张图，每张图3个子图
    for i in range(num_plots):
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))  # 创建3行1列的子图
        start_idx = i * 3  # 起始通道索引
        end_idx = start_idx + 3  # 结束通道索引

        for j, ax in enumerate(axes):  # 循环3个子图
            channel_idx = start_idx + j  # 当前通道的索引

            # 计算误差并绘制误差散点图
            errors = [output[channel_idx] - targets[idx, channel_idx] for idx, output in enumerate(all_outputs)]
            ax.scatter(range(len(errors)), errors, color=error_colors[j], label=f"Error Channel {channel_idx + 1}", marker='o')

            ax.set_xlabel('Sample Index')
            ax.set_ylabel(f'Error (Channel {channel_idx + 1})')
            ax.legend(loc='upper right')
            ax.grid(True)

        # 设置每张图的标题
        fig.suptitle(f'Error for Channels {start_idx + 1} to {end_idx}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局避免重叠
        save_path = os.path.join(base_save_path, f'errors_channels_{start_idx + 1}_to_{end_idx}.png')
        plt.savefig(save_path)  # 保存图像为文件
# 绘制最后n个点的均方误差曲线
def plot_last_n_mse(last_n_mse, n):
    plt.figure()
    plt.plot(range(1, len(last_n_mse) + 1), last_n_mse, label=f'Last {n} Points MSE', color='purple')
    plt.xlabel('Batch Number')
    plt.ylabel('MSE')
    plt.title(f'MSE for Last {n} Points Per Batch')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(base_save_path, f'last_{n}_points_mse_curve.png')
    plt.savefig(save_path)



# 主程序
if __name__ == "__main__":
    # 定义图像基础保存路径
    base_save_path = "D:/A Research/softRL/outputs/trained_nn_model9_same_weight_fixRseed/"
    # 动态创建目录
    os.makedirs(base_save_path, exist_ok=True)

    # 文件路径（假设测试数据与训练数据文件格式相同）
    pressure_file = "D:/A Research/softRL/filtered_pressure_data_302.csv"
    position_file = "D:/A Research/softRL/filtered_position_data.csv"
    #中心化
    # position_file = "D:/A Research/softRL/filtered_position_data_centered.csv"

    # 加载测试数据，设置batch_size为1
    test_loader, pressure_data = load_test_data(pressure_file, position_file, batch_size=1)

    # 创建网络模型
    model = PressureToPositionNet()
    # 加载训练好的模型参数
    # model.load_state_dict(torch.load('../nn_models/best_model.pth'))
    model.load_state_dict(torch.load('../nn_models/trained_nn_model9_same_weight_fixRseed.pth'))
    print("模型已加载")


    # 评估模型并记录每个批次的loss、输出、最后n个点的均方误差，以及总的均方误差
    n = 6  # 设置最后n个点的数量
    batch_losses, all_outputs, last_n_mse, total_mse = evaluate_network_and_record_output(model, test_loader, n=n)

    # 输出总的均方误差
    print(f"Total Mean Squared Error (MSE) for all samples: {total_mse:.4f}")

    # 输出每个批次的损失曲线
    plot_loss_curve(batch_losses)
    # 获取所有目标值（测试集中的所有样本标签）
    targets = torch.cat([targets for _, targets in test_loader], dim=0)
    # 分别绘制15个通道的预测值和真实值
    plot_outputs_with_targets(all_outputs, targets)
    # 分别绘制15个通道的误差
    plot_error(all_outputs, targets)
    # 绘制最后n个点的均方误差曲线
    plot_last_n_mse(last_n_mse, n)
    # 确保所有图形保持显示直到手动关闭
    plt.show()


