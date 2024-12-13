import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev #B样条曲线库
from nn_model import PressureToPositionNet  # 确保导入正确的模型类

# 给定平面上的一个点和法向量
#本代码从nn网络收到的坐标都转化成了标准坐标系
point_on_plane = np.array([0, 0, -80])  # 假设平面过原点 (0, 0, 0)
normal_vector = np.array([0.1, 0.1, 1])  # 假设平面的法向量为 [1, 1, 1]

# 加载训练好的神经网络模型
nn_model = PressureToPositionNet()
nn_model.load_state_dict(torch.load('../nn_models/trained_nn_model10_swgt_fixRseed_centered.pth'))
nn_model.eval()  # 设置模型为评估模式

# 输入的气压数据（假设是一个包含3个气压通道的向量）
pressure_input = np.array([30, 60, 60], dtype=np.float32)  # 示例气压输入
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
ax.scatter(mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2], c='orange', marker='o', s=50)

# 为每个点添加序号标注
for i, point in enumerate(mapped_points):
    ax.text(point[0], point[1], point[2], f'{i+1}', color='blue', fontsize=10)

    # 使用 B 样条平滑连接捕捉点
    tck, u = splprep([mapped_points[:, 0], mapped_points[:, 1], mapped_points[:, 2]], s=1) #_用来丢弃最后一个返回值info（用不到
    u_fine = np.linspace(0, 1, 100)
    smooth_curve = splev(u_fine, tck)

    # 绘制 B 样条曲线
    ax.plot(smooth_curve[0], smooth_curve[1], smooth_curve[2], color='orange', linestyle='-', linewidth=2)

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
ax.plot([mid_x - max_range, mid_x + max_range], [0, 0], [0, 0], color='b', linewidth=1, label='X Axis')
ax.plot([0, 0], [-max_range,max_range], [0, 0], color='b', linewidth=1, label='Y Axis')
ax.plot([0, 0], [0, 0], [mid_z - max_range, mid_z + max_range], color='b', linewidth=1, label='Z Axis')

# **新增：绘制平面**

# 计算平面上的 x 和 y 的值，进而确定对应的 z 值
x_vals = np.linspace(mid_x - max_range, mid_x + max_range, 10)
y_vals = np.linspace(mid_y - max_range, mid_y + max_range, 10)
X, Y = np.meshgrid(x_vals, y_vals)
# 根据点法式平面方程计算 Z 值
# 公式：Ax + By + Cz + D = 0
# 其中，A, B, C 是法向量的分量，D 是常数
A, B, C = normal_vector
D = -(A * point_on_plane[0] + B * point_on_plane[1] + C * point_on_plane[2])
# 计算 Z
Z = -(A * X + B * Y + D) / C
# 绘制平面
ax.plot_surface(X, Y, Z, color='c', alpha=0.3)

# **新增：计算并绘制直线射线**
# 取最后两个点，第4个点 P4 和第5个点 P5
P4 = mapped_points[3]  # 第4个点
P5 = mapped_points[4]  # 第5个点
# 计算直线的方向向量
direction_vector = P5 - P4  # 直线的方向向量 (P5 - P4)
# 为了绘制射线，选择从 P4 开始，沿着方向向量延伸
t_vals = np.linspace(0, 5, 100)  # 控制射线的长度（从 P4 延伸出 5 个单位长度）
# 计算射线上的点
ray_points = P4 + t_vals[:, np.newaxis] * direction_vector
# 绘制射线
ax.plot(ray_points[:, 0], ray_points[:, 1], ray_points[:, 2], color='r', linewidth=1, label='Ray from P4 to P5')

# **计算射线和平面的交点**
# 使用射线和平面方程来计算交点
denominator = A * direction_vector[0] + B * direction_vector[1] + C * direction_vector[2]
numerator = -(A * P4[0] + B * P4[1] + C * P4[2] + D)
t_intersection = numerator / denominator
# 计算交点坐标
intersection_point = P4 + t_intersection * direction_vector
# 输出交点坐标
print(f"Intersection point coordinates: x = {intersection_point[0]}, y = {intersection_point[1]}, z = {intersection_point[2]}")
# 绘制交点
ax.scatter(intersection_point[0], intersection_point[1], intersection_point[2], color='r', s=30, label='Intersection Point')


# 显示3D图
plt.show()
