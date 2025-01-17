#只计算测试集loss并输出
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

from nn_model import PressureToPositionNet  # 从 model.py 导入网络

# 读取测试数据
def load_test_data(pressure_file, position_file, batch_size=32):
    # 加载气压输入数据和位置输出数据
    pressure_data = pd.read_csv(pressure_file)
    position_data = pd.read_csv(position_file)

    # 假设气压数据在第3到5列（Python索引从0开始，所以是2到5）
    X_test = torch.tensor(pressure_data.iloc[:, 2:5].values, dtype=torch.float32)  # 读取第3到5列气压数据

    # 假设位置数据在第1到15列（Python索引从0开始，所以是0到15）
    y_test = torch.tensor(position_data.iloc[:, 0:15].values, dtype=torch.float32)  # 读取第1到15列位置数据

    # 创建数据集和数据加载器
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


# 评估神经网络
def evaluate_network(model, test_loader):
    model.eval()  # 设置模型为评估模式（禁用dropout等）
    total_loss = 0.0
    criterion = torch.nn.MSELoss()  # 使用均方误差来评估性能
    with torch.no_grad():  # 禁用梯度计算以加快评估
        for inputs, targets in test_loader:
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            total_loss += loss.item()  # 累加损失

    avg_loss = total_loss / len(test_loader)  # 计算平均损失
    print(f"Test Loss (MSE): {avg_loss:.4f}")


# 主程序
if __name__ == "__main__":
    # 文件路径（假设测试数据与训练数据文件格式相同）
    # pressure_file = "D:/A Research/softRL/Dat0302_processed_3d_win30_thr0.5.csv"
    # position_file = "D:/A Research/softRL/mocapdata_processed.csv"
    pressure_file = "D:/A Research/softRL/filtered_pressure_data_302.csv"
    position_file = "D:/A Research/softRL/filtered_position_data.csv"
    # 加载测试数据
    test_loader = load_test_data(pressure_file, position_file)
    # 创建网络模型
    model = PressureToPositionNet()
    # 加载训练好的模型参数
    model.load_state_dict(torch.load('../nn_models/trained_model.pth'))   ###此处需要修改
    print("模型已加载")

    # 评估模型在测试集上的性能
    evaluate_network(model, test_loader)
