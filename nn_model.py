import torch
import torch.nn as nn


# 定义神经网络
class PressureToPositionNet(nn.Module):
    def __init__(self):
        super(PressureToPositionNet, self).__init__()
        # 输入层到第一个隐藏层，100个神经元，激活函数为Sigmoid
        self.fc1 = nn.Linear(3, 100)
        self.sigmoid = nn.Sigmoid()

        # 第一个隐藏层到第二个隐藏层，100个神经元，线性传递函数
        self.fc2 = nn.Linear(100, 100)

        # 第二个隐藏层到输出层，15个输出通道（5组，每组3维）
        self.fc3 = nn.Linear(100, 15)

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))  # 第一个隐藏层，激活函数为Sigmoid
        x = self.fc2(x)  # 第二个隐藏层，线性传递
        x = self.fc3(x)  # 输出层，线性传递
        return x


# 简单测试网络
def test_network():
    # 创建一个简单的模型实例
    model = PressureToPositionNet()

    # 生成一些随机输入（假设输入为3通道气压数据）
    test_input = torch.randn(1, 3)  # batch_size=1, 输入3个通道

    # 通过网络前向传递
    output = model(test_input)

    # 打印输出
    print("输入：", test_input)
    print("输出：", output)


# 如果单独运行这个文件，测试网络结构
if __name__ == "__main__":
    test_network()
