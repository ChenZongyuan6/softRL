'''
在nn_test3.py基础上进行改进，在测试nn网络输出的时候，不再用整个输出共15维来评估，而是只用15维中的最后3个维度（含义上对应最末端一个目标点的三维坐标），
计算输出值的最后三个维度和测试数据中给出的对应真实值的最后三个维度的欧氏距离，代替原代码中的损失（loss），要输出的图和原代码类似，还是计算并输出
每个batch的误差和总的平均误差，并做和上述相同的图，包括loss的图，以及绘制Pressure X Y Z  分别和 Loss对比的图，共4张
'''
# 计算测试集误差，输出并对比每个batch的气压数据和误差，为方便观察单个数据情况，把batch_size设为1
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from nn_model import PressureToPositionNet  # 从 nn_models.py 导入网络

# 读取测试数据
def load_test_data(pressure_file, position_file, batch_size=1):
    # 加载气压输入数据和位置输出数据
    pressure_data = pd.read_csv(pressure_file)
    position_data = pd.read_csv(position_file)

    # 假设气压数据在第3到5列（Python索引从0开始，所以是2到5）
    X_test = torch.tensor(pressure_data.iloc[:, 2:5].values, dtype=torch.float32)  # 读取第3到5列气压数据

    # 假设位置数据在第1到15列（Python索引从0开始，所以是0到15）
    y_test = torch.tensor(position_data.iloc[:, 0:15].values, dtype=torch.float32)  # 读取第1到15列位置数据

    # 创建数据集和数据加载器
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 不打乱顺序

    return test_loader, pressure_data  # 返回DataLoader和原始气压数据

# 评估神经网络并记录每个批次的误差，最终输出总平均误差
def evaluate_network_and_record_error(model, test_loader):
    model.eval()  # 设置模型为评估模式
    batch_errors = []  # 用于记录每个批次的误差
    total_error = 0.0  # 累加所有批次的误差

    with torch.no_grad():  # 禁用梯度计算以加快评估
        for inputs, targets in test_loader:
            outputs = model(inputs)  # 前向传播

            # 取输出和目标的最后三个维度
            # outputs_last3 = outputs[:, -3:]
            # targets_last3 = targets[:, -3:]
            #某3个维度 0:3 3:6 6:9 9:12 12:15
            outputs_last3 = outputs[:, 12:15]
            targets_last3 = targets[:, 12:15]

            # 计算欧氏距离作为误差
            error = torch.norm(outputs_last3 - targets_last3, dim=1).item()
            batch_errors.append(error)  # 记录误差
            total_error += error  # 累加每个批次的误差

    avg_error = total_error / len(test_loader)  # 计算总的平均误差
    print(f"Total Average Error: {avg_error:.4f}")  # 输出总的平均误差
    return batch_errors, avg_error

# 绘制误差散点图
def plot_error_curve(batch_errors):
    plt.figure()
    plt.scatter(range(1, len(batch_errors) + 1), batch_errors, label="Batch Error")
    plt.xlabel('Batch Number')
    plt.ylabel('Error')
    plt.title('Error Per Batch')
    plt.legend()
    plt.grid(True)
    plt.savefig('batch_error_curve.png')

# 修改：分别可视化气压数据的三个维度，并同时绘制误差散点图
def plot_pressure_data_with_error(pressure_data, batch_errors):
    x_range = range(1, len(pressure_data) + 1)
    error_range = range(1, len(batch_errors) + 1)

    # 绘制Pressure X + Error
    fig, ax1 = plt.subplots()
    ax1.scatter(x_range, pressure_data.iloc[:, 2], color='b', label='Pressure X')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Pressure X', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.scatter(error_range, batch_errors, color='r', label='Error')
    ax2.set_ylabel('Error', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Pressure Data (X) and Error Over Time')
    fig.tight_layout()
    plt.savefig('pressure_x_and_error.png')

    # 绘制Pressure Y + Error
    fig, ax1 = plt.subplots()
    ax1.scatter(x_range, pressure_data.iloc[:, 3], color='g', label='Pressure Y')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Pressure Y', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    ax2 = ax1.twinx()
    ax2.scatter(error_range, batch_errors, color='r', label='Error')
    ax2.set_ylabel('Error', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Pressure Data (Y) and Error Over Time')
    fig.tight_layout()
    plt.savefig('pressure_y_and_error.png')

    # 绘制Pressure Z + Error
    fig, ax1 = plt.subplots()
    ax1.scatter(x_range, pressure_data.iloc[:, 4], color='c', label='Pressure Z')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Pressure Z', color='c')
    ax1.tick_params(axis='y', labelcolor='c')

    ax2 = ax1.twinx()
    ax2.scatter(error_range, batch_errors, color='r', label='Error')
    ax2.set_ylabel('Error', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title('Pressure Data (Z) and Error Over Time')
    fig.tight_layout()
    plt.savefig('pressure_z_and_error.png')

# 主程序
if __name__ == "__main__":
    # 文件路径（假设测试数据与训练数据文件格式相同）
    pressure_file = "D:/A Research/softRL/filtered_pressure_data_302.csv"  ###
    # position_file = "D:/A Research/softRL/data_centered/filtered_position_data_centered.csv"
    position_file = "D:/A Research/softRL/filtered_position_data.csv"    ###
    # 加载测试数据，设置batch_size为1
    test_loader, pressure_data = load_test_data(pressure_file, position_file, batch_size=1)

    # 创建网络模型
    model = PressureToPositionNet()

    # 加载训练好的模型参数
    # model.load_state_dict(torch.load('../nn_models/nn_model_1123best.pth'))
    # model.load_state_dict(torch.load('../nn_models/trained_nn_model8_same_weight.pth'))
    model.load_state_dict(torch.load('../nn_models/trained_nn_model10_swgt_fixRseed_centered.pth'))
    print("模型已加载")

    # 评估模型并记录每个批次的误差，同时计算总的平均误差
    batch_errors, avg_error = evaluate_network_and_record_error(model, test_loader)

    # 输出每个批次的误差曲线
    plot_error_curve(batch_errors)
    # 分别可视化气压数据的3个通道和误差
    plot_pressure_data_with_error(pressure_data, batch_errors)
    # 确保所有图形保持显示直到手动关闭
    plt.show()
