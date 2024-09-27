import torch
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # 导入 matplotlib
import time

from nn_model import PressureToPositionNet  # 从model.py导入网络


# 读取气压输入和位置输出的数据
def load_data(pressure_file, position_file, batch_size=32):
    # 加载气压输入数据和位置输出数据
    pressure_data = pd.read_csv(pressure_file)
    position_data = pd.read_csv(position_file)

    # 假设气压输入和位置输出数据分别在第2到4列和第2到16列
    X = torch.tensor(pressure_data.iloc[:, 2:5].values, dtype=torch.float32)  # 气压输入
    y = torch.tensor(position_data.iloc[:, 0:15].values, dtype=torch.float32)  # 位置输出

    # 创建数据集和数据加载器
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


# 训练神经网络
def train_network(model, dataloader, epochs=100, lr=0.001):
    # 使用Adam优化器和均方误差损失函数
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 用于保存每个epoch的平均loss
    loss_history = []

    # 记录训练开始时间
    start_time = time.time()

    # 训练过程
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)  # 将平均损失添加到列表中
        # 打印每个epoch的平均损失
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")

    # 记录训练结束时间
    end_time = time.time()
    # 计算总的训练时间
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    # 保存训练好的模型
    torch.save(model.state_dict(), "trained_model.pth")
    print("模型已保存到 trained_model.pth")

    # 绘制loss收敛曲线
    plt.figure()  # 新建一个图形窗口
    plt.plot(range(1, epochs + 1), loss_history, label='Loss')  # 绘制loss历史
    plt.xlabel('Epoch')  # 设置横坐标
    plt.ylabel('Loss')  # 设置纵坐标
    plt.title('Training Loss Over Epochs')  # 图形标题
    plt.ylim(0, 10)  # 设置纵轴的范围为0到10
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线
    plt.savefig('loss_convergence.png')  # 将图像保存为文件
    plt.show()  # 显示图像

# 主程序
if __name__ == "__main__":
    # 文件路径
    pressure_file = "D:/A Research/softRL/Dat03022_processed_3d_win30.csv"
    position_file = "D:/A Research/softRL/mocapdata_3022_28226_processed.csv"

    # 加载数据
    dataloader = load_data(pressure_file, position_file)
    # 创建网络模型
    model = PressureToPositionNet()
    # 训练模型
    train_network(model, dataloader, epochs=100, lr=0.001)
