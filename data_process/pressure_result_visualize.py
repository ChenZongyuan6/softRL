import pandas as pd
import matplotlib.pyplot as plt

def visualize_data(original_file, processed_file, num_points=10000):
    # 读取原始数据和处理后的有效数据
    df = pd.read_csv(original_file).iloc[:num_points]  # 只取前 num_points 个数据
    result_df = pd.read_csv(processed_file)

    # 准备X, Y, Z方向的数据
    x_data = df.iloc[:, 1]
    y_data = df.iloc[:, 2]
    z_data = df.iloc[:, 3]

    # 准备有效数据段的X, Y, Z方向数据
    valid_x = []
    valid_y = []
    valid_z = []
    valid_indices = []

    for _, row in result_df.iterrows():
        start_time = row['Start Time']
        duration = row['Duration']
        # 找到有效数据段对应的行索引
        indices = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] <= start_time + duration)].index
        valid_indices.extend(indices)
        valid_x.extend(x_data[indices])
        valid_y.extend(y_data[indices])
        valid_z.extend(z_data[indices])

    # 可视化X方向数据
    plt.figure(figsize=(10, 6))
    plt.plot(x_data.index, x_data, label='Original X Data', color='blue')
    plt.scatter(valid_indices, valid_x, label='Valid X Data', color='red', s=10)
    plt.title('X Data with Valid Segments')
    plt.legend()

    # 可视化Y方向数据
    plt.figure(figsize=(10, 6))
    plt.plot(y_data.index, y_data, label='Original Y Data', color='blue')
    plt.scatter(valid_indices, valid_y, label='Valid Y Data', color='red', s=10)
    plt.title('Y Data with Valid Segments')
    plt.legend()

    # 可视化Z方向数据
    plt.figure(figsize=(10, 6))
    plt.plot(z_data.index, z_data, label='Original Z Data', color='blue')
    plt.scatter(valid_indices, valid_z, label='Valid Z Data', color='red', s=10)
    plt.title('Z Data with Valid Segments')
    plt.legend()

    # 显示所有图像，并等待用户手动关闭窗口
    plt.show()

input_data = "D:/A Research/softRL/Dat03022_input.csv"
processed_data = "D:/A Research/softRL/Dat03022_processed_3d_win30.csv"
# 示例调用
visualize_data(input_data, processed_data, num_points=5000)
