#采用的版本
import pandas as pd
import matplotlib.pyplot as plt


def process_data(input_file, output_file, window_size=30, threshold1=0.5, threshold2=0.6, threshold3=0.5):
    # 读取CSV文件，并过滤掉任何包含空值的行
    df = pd.read_csv(input_file).dropna()

    # 初始化变量
    valid_data = []
    i = 0
    while i < len(df) - window_size + 1:
        # 取出当前窗口内的X, Y, Z方向数据
        x_window = df.iloc[i:i + window_size, 1]
        y_window = df.iloc[i:i + window_size, 2]
        z_window = df.iloc[i:i + window_size, 3]

        # 检查X, Y, Z方向数据的波动是否都在±0.2以内，即最大值 - 最小值 ≤ 0.4
        if (x_window.max() - x_window.min() <= threshold1 and
                y_window.max() - y_window.min() <= threshold2 and
                z_window.max() - z_window.min() <= threshold3):

            # 计算有效数据段的开始时间和持续时间
            start_time = df.iloc[i, 0]
            duration = df.iloc[i + window_size - 1, 0] - df.iloc[i, 0]

            # 计算该段X, Y, Z方向数据的均值
            x_mean = x_window.mean()
            y_mean = y_window.mean()
            z_mean = z_window.mean()

            # 记录有效数据段
            valid_data.append([start_time, duration, x_mean, y_mean, z_mean])

            # 跳过这个有效数据段
            i += window_size
        else:
            # 如果不满足波动条件，则继续检查下一个数据点
            i += 1

    # 创建结果数据表
    result_df = pd.DataFrame(valid_data, columns=['Start Time', 'Duration', 'X Mean', 'Y Mean', 'Z Mean'])

    # 输出结果到新的CSV文件
    result_df.to_csv(output_file, index=False, header=False)

    return df, result_df  # 返回原始数据和有效数据段


def visualize_data(df, result_df, num_points=20000):
    # 只取前 num_points 个数据进行可视化
    df = df.iloc[:num_points]

    # 准备X, Y, Z方向的数据
    x_data = df.iloc[:, 1]
    y_data = df.iloc[:, 2]
    z_data = df.iloc[:, 3]

    # 准备有效数据段
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


# 示例调用

input_data = "D:/A Research/softRL/Dat0302_input.csv"
processed_data = "D:/A Research/softRL/Dat0302_processed_3d_win30_2.csv"
df, result_df =process_data(input_data, processed_data)

visualize_data(df, result_df, num_points=20000)
