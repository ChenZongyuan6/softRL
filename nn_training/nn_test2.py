#计算测试集Loss，输出并对比每个batch的气压数据和loss，为方便观察单个数据情况，把batch_size设为1
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from nn_model import PressureToPositionNet  # 从 model.py 导入网络


# 读取测试数据
def load_test_data(pressure_file, position_file, batch_size=1):  # 修改batch_size为6
    # 加载气压输入数据和位置输出数据
    pressure_data = pd.read_csv(pressure_file)
    position_data = pd.read_csv(position_file)

    # 假设气压数据在第3到5列（Python索引从0开始，所以是2到5）
    X_test = torch.tensor(pressure_data.iloc[:, 2:5].values, dtype=torch.float32)  # 读取第3到5列气压数据

    # 假设位置数据在第1到15列（Python索引从0开始，所以是0到15）
    y_test = torch.tensor(position_data.iloc[:, 0:15].values, dtype=torch.float32)  # 读取第1到15列位置数据

    # 创建数据集和数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 不打乱顺序

    return test_loader, pressure_data  # 返回DataLoader和原始气压数据

# 评估神经网络并记录每个批次的loss，最终输出总平均loss
def evaluate_network_and_record_loss(model, test_loader):
    model.eval()  # 设置模型为评估模式
    criterion = torch.nn.MSELoss()  # 使用均方误差来评估性能
    batch_losses = []  # 用于记录每个批次的损失
    total_loss = 0.0  # 累加所有批次的损失

    with torch.no_grad():  # 禁用梯度计算以加快评估
        for inputs, targets in test_loader:
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算当前批次的损失
            batch_losses.append(loss.item())  # 记录损失
            total_loss += loss.item()  # 累加每个批次的损失

    avg_loss = total_loss / len(test_loader)  # 计算总的平均loss
    print(f"Total Average Loss: {avg_loss:.4f}")  # 输出总的平均loss
    return batch_losses, avg_loss

#####按折线图绘制
# # 绘制loss曲线
# def plot_loss_curve(batch_losses):
#     plt.figure()
#     plt.plot(range(1, len(batch_losses) + 1), batch_losses, label="Batch Loss")  # 绘制每个批次的损失
#     plt.xlabel('Batch Number')  # 横坐标为批次编号
#     plt.ylabel('Loss')  # 纵坐标为损失值
#     plt.title('Loss Per Batch')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('batch_loss_curve.png')  # 保存图像为文件
#     # plt.show() 在主程序中调用一次即可
#
# # 修改：分别可视化气压数据的三个维度，并同时绘制loss曲线
# def plot_pressure_data_with_loss(pressure_data, batch_losses):
#     x_range = range(1, len(pressure_data) + 1)
#     loss_range = range(1, len(batch_losses) + 1)
#
#     # 绘制Pressure X + Loss
#     fig, ax1 = plt.subplots()  # 创建包含两个y轴的图表
#     ax1.plot(x_range, pressure_data.iloc[:, 2], 'b-', label='Pressure X')  # 绘制第3列气压数据
#     ax1.set_xlabel('Sample Index')
#     ax1.set_ylabel('Pressure X', color='b')
#     ax1.tick_params(axis='y', labelcolor='b')
#
#     ax2 = ax1.twinx()  # 创建第二个y轴，绘制loss曲线
#     ax2.plot(loss_range, batch_losses, 'r-', label='Loss')  # 绘制loss曲线
#     ax2.set_ylabel('Loss', color='r')
#     ax2.tick_params(axis='y', labelcolor='r')
#
#     plt.title('Pressure Data (X) and Loss Over Time')
#     fig.tight_layout()  # 调整布局以避免标签重叠
#     plt.savefig('pressure_x_and_loss.png')
#
#     # 绘制Pressure Y + Loss
#     fig, ax1 = plt.subplots()  # 创建包含两个y轴的图表
#     ax1.plot(x_range, pressure_data.iloc[:, 3], 'g-', label='Pressure Y')  # 绘制第4列气压数据
#     ax1.set_xlabel('Sample Index')
#     ax1.set_ylabel('Pressure Y', color='g')
#     ax1.tick_params(axis='y', labelcolor='g')
#
#     ax2 = ax1.twinx()  # 创建第二个y轴，绘制loss曲线
#     ax2.plot(loss_range, batch_losses, 'r-', label='Loss')  # 绘制loss曲线
#     ax2.set_ylabel('Loss', color='r')
#     ax2.tick_params(axis='y', labelcolor='r')
#
#     plt.title('Pressure Data (Y) and Loss Over Time')
#     fig.tight_layout()  # 调整布局以避免标签重叠
#     plt.savefig('pressure_y_and_loss.png')
#
#     # 绘制Pressure Z + Loss
#     fig, ax1 = plt.subplots()  # 创建包含两个y轴的图表
#     ax1.plot(x_range, pressure_data.iloc[:, 4], 'c-', label='Pressure Z')  # 绘制第5列气压数据
#     ax1.set_xlabel('Sample Index')
#     ax1.set_ylabel('Pressure Z', color='c')
#     ax1.tick_params(axis='y', labelcolor='c')
#
#     ax2 = ax1.twinx()  # 创建第二个y轴，绘制loss曲线
#     ax2.plot(loss_range, batch_losses, 'r-', label='Loss')  # 绘制loss曲线
#     ax2.set_ylabel('Loss', color='r')
#     ax2.tick_params(axis='y', labelcolor='r')
#
#     plt.title('Pressure Data (Z) and Loss Over Time')
#     fig.tight_layout()  # 调整布局以避免标签重叠
#     plt.savefig('pressure_z_and_loss.png')

#####按散点图绘制# 绘制loss散点图
def plot_loss_curve(batch_losses):
    plt.figure()
    plt.scatter(range(1, len(batch_losses) + 1), batch_losses, label="Batch Loss")  # 使用散点图代替折线图
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.title('Loss Per Batch')
    plt.legend()
    plt.grid(True)
    plt.savefig('batch_loss_curve.png')

# 修改：分别可视化气压数据的三个维度，并同时绘制loss散点图
def plot_pressure_data_with_loss(pressure_data, batch_losses):
    x_range = range(1, len(pressure_data) + 1)
    loss_range = range(1, len(batch_losses) + 1)

    # 绘制Pressure X + Loss
    fig, ax1 = plt.subplots()
    ax1.scatter(x_range, pressure_data.iloc[:, 2], color='b', label='Pressure X')  # 使用散点图绘制第3列气压数据
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Pressure X', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.scatter(loss_range, batch_losses, color='r', label='Loss')  # 使用散点图绘制loss
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Pressure Data (X) and Loss Over Time')
    fig.tight_layout()
    plt.savefig('pressure_x_and_loss.png')

    # 绘制Pressure Y + Loss
    fig, ax1 = plt.subplots()
    ax1.scatter(x_range, pressure_data.iloc[:, 3], color='g', label='Pressure Y')  # 使用散点图绘制第4列气压数据
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Pressure Y', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()
    ax2.scatter(loss_range, batch_losses, color='r', label='Loss')  # 使用散点图绘制loss
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Pressure Data (Y) and Loss Over Time')
    fig.tight_layout()
    plt.savefig('pressure_y_and_loss.png')

    # 绘制Pressure Z + Loss
    fig, ax1 = plt.subplots()
    ax1.scatter(x_range, pressure_data.iloc[:, 4], color='c', label='Pressure Z')  # 使用散点图绘制第5列气压数据
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Pressure Z', color='c')
    ax1.tick_params(axis='y', labelcolor='c')

    ax2 = ax1.twinx()
    ax2.scatter(loss_range, batch_losses, color='r', label='Loss')  # 使用散点图绘制loss
    ax2.set_ylabel('Loss', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Pressure Data (Z) and Loss Over Time')
    fig.tight_layout()
    plt.savefig('pressure_z_and_loss.png')

# 主程序
if __name__ == "__main__":
    # 文件路径（假设测试数据与训练数据文件格式相同）
    # pressure_file = "D:/A Research/softRL/Dat03022_processed_3d_win30.csv"
    # position_file = "D:/A Research/softRL/mocapdata_3022_28226_processed.csv"
    # pressure_file = "D:/A Research/softRL/train_test_sep/pressure_test.csv"
    # position_file = "D:/A Research/softRL/filtered_position_data.csv"
    pressure_file = "D:/A Research/softRL/filtered_pressure_data_302.csv"
    position_file = "D:/A Research/softRL/filtered_position_data.csv"
    # 加载测试数据，设置batch_size为6
    test_loader, pressure_data = load_test_data(pressure_file, position_file, batch_size=1)

    # 创建网络模型
    model = PressureToPositionNet()

    # 加载训练好的模型参数
    model.load_state_dict(torch.load('../nn_models/nn_model_1123best.pth'))
    # model.load_state_dict(torch.load('../nn_models/best_model.pth'))   ###此处需要修改
    print("模型已加载")

    # 评估模型并记录每个批次的损失，同时计算总的平均损失
    batch_losses, avg_loss = evaluate_network_and_record_loss(model, test_loader)

    # 输出每个批次的损失曲线
    plot_loss_curve(batch_losses)
    # 分别可视化气压数据的3个通道和Loss
    plot_pressure_data_with_loss(pressure_data, batch_losses)
    # 确保所有图形保持显示直到手动关闭
    plt.show()