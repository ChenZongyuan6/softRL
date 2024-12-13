#本代码是对任意给定气压，输出
import torch
import numpy as np
import matplotlib.pyplot as plt
from nn_model import PressureToPositionNet  # 确保导入正确的模型类

# 加载训练好的神经网络模型
nn_model = PressureToPositionNet()
# nn_model.load_state_dict(torch.load('../nn_models/trained_model.pth'))  # 加载预训练的权重
# nn_model.load_state_dict(torch.load('../nn_models/best_model.pth'))
nn_model.load_state_dict(torch.load('../nn_models/trained_nn_model10_swgt_fixRseed_centered.pth'))
nn_model.eval()  # 设置模型为评估模式

# 输入的气压数据（假设是一个包含3个气压通道的向量）
pressure_input = np.array([30, 60, 60], dtype=np.float32)  # 示例气压输入[35, 50, 65]
pressure_input_tensor = torch.tensor(pressure_input).unsqueeze(0)  # 转换为张量，并增加批次维度

# 使用神经网络进行预测

with torch.no_grad():  # 在评估模式下不需要计算梯度
    predicted_output = nn_model(pressure_input_tensor)

# 将输出转换为 numpy 数组，并去掉批次维度
predicted_output = predicted_output.numpy().flatten()

# 输出15维结果，其中每3个数为一个点的坐标
points = predicted_output.reshape(5, 3)  # 5个点的x, y, z坐标
print("Predicted 5 points (x, y, z):")
for i, point in enumerate(points):
    print(f"Point {i+1}: x = {point[0]}, y = {point[1]}, z = {point[2]}")

# 在三维图中可视化这5个点，并添加序号标注
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 对坐标进行重新映射：
# - 原 X 数据映射到 Y 轴
# - 原 Y 数据映射到 Z 轴
# - 原 Z 数据映射到 X 轴
mapped_points = np.zeros_like(points)
mapped_points[:, 0] = points[:, 2]  # 原 Z 轴数据映射到 X 轴
mapped_points[:, 1] = points[:, 0]  # 原 X 轴数据映射到 Y 轴
mapped_points[:, 2] = points[:, 1]  # 原 Y 轴数据映射到 Z 轴（竖直方向）

# 绘制5个点的映射后坐标
ax.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], c='r', marker='o', s=100)

# 为每个点添加序号标注
for i, point in enumerate(mapped_points):
    ax.text(point[0], point[1], point[2], f'{i+1}', color='blue', fontsize=12)

# 设置每个轴的范围，使得X、Y、Z方向比例一致
max_range = np.array([mapped_points[:, 0].max() - mapped_points[:, 0].min(),
                      mapped_points[:, 1].max() - mapped_points[:, 1].min(),
                      mapped_points[:, 2].max() - mapped_points[:, 2].min()]).max() / 2.0

mid_x = (mapped_points[:, 0].max() + mapped_points[:, 0].min()) * 0.5
mid_y = (mapped_points[:, 1].max() + mapped_points[:, 1].min()) * 0.5
mid_z = (mapped_points[:, 2].max() + mapped_points[:, 2].min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# 统一坐标轴单位尺寸
ax.set_box_aspect([1, 1, 1])  # 设置 X, Y, Z 轴的比例相同

# 设置坐标轴标签
ax.set_xlabel('X Coordinate (mocap data Z)')
ax.set_ylabel('Y Coordinate (mocap data X)')
ax.set_zlabel('Z Coordinate (mocap data Y)')

# 设置图形标题
ax.set_title('3D Visualization of 5 Predicted Points with Axis Mapping')
# **新增：绘制坐标轴线 (X=0, Y=0, Z=0)**
# X轴: 连接点(-max_range, 0, 0) 和 (max_range, 0, 0)
ax.plot([mid_x - max_range, mid_x + max_range], [0, 0], [0, 0], color='b', linewidth=1, label='X Axis')
# Y轴: 连接点(0, -max_range, 0) 和 (0, max_range, 0)
ax.plot([0, 0], [-max_range,max_range], [0, 0], color='b', linewidth=1, label='Y Axis')
# Z轴: 连接点(0, 0, -max_range) 和 (0, 0, max_range)
ax.plot([0, 0], [0, 0], [mid_z - max_range, mid_z + max_range], color='b', linewidth=1, label='Z Axis')

# 显示3D图
plt.show()
