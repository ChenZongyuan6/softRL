#20241120 在nn_training的基础上，划分出10%的验证集，每几个episode检查一次，若验证集loss超过patience个episode没有降低，则早停
import torch
import random
import numpy as np
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split  # 导入random_split
import matplotlib.pyplot as plt
import time
import os

from nn_model import PressureToPositionNet  # 从model.py导入网络


# 读取气压输入和位置输出的数据
def load_data(pressure_file, position_file, batch_size=32):
    # 加载气压输入数据和位置输出数据
    pressure_data = pd.read_csv(pressure_file)
    position_data = pd.read_csv(position_file)

    # 假设气压输入和位置输出数据分别在第2到4列和第2到16列
    X = torch.tensor(pressure_data.iloc[:, 2:5].values, dtype=torch.float32)  # 气压输入
    y = torch.tensor(position_data.iloc[:, 0:15].values, dtype=torch.float32)  # 位置输出

    # 创建数据集
    dataset = TensorDataset(X, y)

    # **修改部分：将数据集随机分成90%的训练集和10%的验证集**
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader  # 返回训练集和验证集的加载器

def weighted_mse_loss(output, target):
    """
    自定义加权均方误差损失函数
    :param output: 模型预测的输出
    :param target: 实际的目标值
    :return: 加权后的MSE损失
    """
    # 定义权重
    weights = torch.ones_like(target)  # 创建一个与 target 张量形状相同的全1张量
    weights[:, 9:12] = 1.0  # 强化第10到12维的权重
    weights[:, 12:15] = 1.0  # 强化第13到15维的权重
    # 计算加权均方误差
    loss = torch.mean(weights * (output - target) ** 2)
    return loss

# 训练神经网络
def train_network(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
    # 使用Adam优化器和均方误差损失函数
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # 用于保存每个epoch的平均loss
    loss_history = []
    val_loss_history = []

    # **修改部分：设置早停法的参数**
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    # 初始化变量，用于缓存最佳模型状态
    best_model_state = None  # 缓存最佳模型的状态

    # 记录训练开始时间
    start_time = time.time()

    # 训练过程
    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, targets in train_loader:
            # 将数据移到设备
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)  # 将平均损失添加到列表中
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_loss}")

        # **修改部分：每隔5个epoch验证一次模型在验证集上的loss**
        if (epoch + 1) % 5 == 0:
            model.eval()  # 设置模型为评估模式
            val_running_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    # 确保验证数据也在同一设备上
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)  # 移动验证数据到GPU/CPU
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_targets)
                    val_running_loss += val_loss.item()
            avg_val_loss = val_running_loss / len(val_loader)
            val_loss_history.append(avg_val_loss)
            print(f"Validation Loss after Epoch [{epoch + 1}]: {avg_val_loss}")

            # **修改部分：早停机制**
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_since_improvement = 0
                # 缓存当前最好的模型状态到内存，不立即保存到磁盘
                best_model_state = model.state_dict()
                print("Validation loss improved, caching model state...")
            else:
                epochs_since_improvement += 1
                print(f"No improvement in validation loss for {epochs_since_improvement} epochs.")
                if epochs_since_improvement >= patience:
                    print("Early stopping triggered.")
                    break

    # 保存最佳模型
    if best_model_state is not None:
        save_path = "../nn_models/trained_nn_model10_swgt_fixRseed_centered.pth"  ###############
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f"Best model saved to {save_path}.")

    # 记录训练结束时间
    end_time = time.time()
    # 计算总的训练时间
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    # **修改部分：绘制训练和验证的loss曲线**
    # 定义基础保存路径
    base_save_path = "D:/A Research/softRL/outputs/trained_nn_model10_swgt_fixRseed_centered" ##########
    # 动态创建目录
    os.makedirs(base_save_path, exist_ok=True)

    # 保存最终的 loss 曲线图像
    save_path = os.path.join(base_save_path, "loss_convergence.png")  # 固定保存文件名

    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Training Loss')
    val_epochs = list(range(5, epochs + 1, 5))
    plt.plot(val_epochs[:len(val_loss_history)], val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)

    # 保存图像
    plt.savefig(save_path)  # 保存到指定路径
    plt.show()

    print(f"Loss convergence plot saved to {save_path}")


def set_random_seed(seed):
    # 设置Python内置的随机数生成器
    random.seed(seed)

    # 设置NumPy的随机种子
    np.random.seed(seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed)

    # 如果有GPU，设置GPU的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU

    # 设置为每次初始化时使用相同的种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 主程序
if __name__ == "__main__":
    # 设置随机种子
    set_random_seed(42)
    # 检查设备，优先使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 文件路径
    pressure_file = "D:/A Research/softRL/Dat03022_processed_3d_win30.csv"   ###
    # position_file = "D:/A Research/softRL/mocapdata_3022_28226_processed.csv"   ###
    position_file = "D:/A Research/softRL/data_centered/mocapdata_3022_centered.csv" ###

    # 加载数据
    train_loader, val_loader = load_data(pressure_file, position_file)
    # # 创建网络模型
    # model = PressureToPositionNet()
    # 创建网络模型并移动到设备
    model = PressureToPositionNet().to(device)

    # 训练模型
    train_network(model, train_loader, val_loader, epochs=1000, lr=0.001)
