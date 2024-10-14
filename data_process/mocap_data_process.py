#这个是可用版本
import pandas as pd

def extract_and_average_mocap_data(valid_indices_file, mocap_file, mocap_average_output_file):
    # 读取第一个文件，提取有效数据的开始索引和长度
    valid_df = pd.read_csv(valid_indices_file).dropna()

    # 假设第0列为开始时间，第1列为持续时间（行号和段长）
    start_indices = valid_df.iloc[:, 0].astype(int).tolist()
    durations = valid_df.iloc[:, 1].astype(int).tolist()

    # 读取动捕数据的CSV文件
    mocap_df = pd.read_csv(mocap_file)

    # 初始化一个空的DataFrame来存放有效的动捕数据段
    valid_mocap_data = pd.DataFrame()

    # 初始化列表来存放每段动捕数据的均值
    mocap_segment_means = []

    # 逐段提取有效的动捕数据
    for start, duration in zip(start_indices, durations):
        # 根据开始索引和持续时间提取动捕数据段
        mocap_segment = mocap_df.iloc[start:start + duration]

        # 将提取到的有效动捕数据段存入 valid_mocap_data
        valid_mocap_data = pd.concat([valid_mocap_data, mocap_segment], ignore_index=True)

        # 计算每段动捕数据第2到第16列（即15个分量）的均值
        mocap_means = mocap_segment.iloc[:, 1:17].mean()
        mocap_segment_means.append(mocap_means)

    # 将每段动捕数据的均值保存到新的DataFrame中
    mocap_means_df = pd.DataFrame(mocap_segment_means)

    # 保存动捕数据的均值结果到新的表格中
    mocap_means_df.to_csv(mocap_average_output_file, index=False)
    print(f"每段动捕数据的均值已保存到: {mocap_average_output_file}")

# processed_pressure_data = "D:/A Research/softRL/Dat03022_processed_3d_win30.csv"
# mocap_data_input = "D:/A Research/softRL/mocapdata_input_3022_282260.csv"
# mocap_data_processed = "D:/A Research/softRL/mocapdata_3022_28226_processed.csv"
processed_pressure_data = "D:/A Research/softRL/Dat0302_processed_3d_win30_2.csv"
mocap_data_input = "D:/A Research/softRL/mocapdata_input.csv"
mocap_data_processed = "D:/A Research/softRL/mocapdata_processed.csv"
# 示例调用
extract_and_average_mocap_data(processed_pressure_data, mocap_data_input, mocap_data_processed)
