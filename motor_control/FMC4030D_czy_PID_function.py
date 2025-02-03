# -*- coding: UTF-8 -*-
from ctypes import *
import time
import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题
# 加载动态链接库
fmc4030 = windll.LoadLibrary('FMC4030Lib-x64-20220329/FMC4030-DLL.dll')

# 定义设备状态类，用于获取设备状态数据
class machine_status(Structure):
    _fields_ = [
        ("realPos", c_float * 3),
        ("realSpeed", c_float * 3),
        ("inputStatus", c_int32 * 1),
        ("outputStatus", c_int32 * 1),
        ("limitNStatus", c_int32 * 1),
        ("limitPStatus", c_int32 * 1),
        ("machineRunStatus", c_int32 * 1),
        ("axisStatus", c_int32 * 3),
        ("homeStatus", c_int32 * 1),
        ("file", c_ubyte * 600)
    ]

# 初始化串口通信
def initialize_serial(port, baudrate=115200, timeout=2.0):
    master = modbus_rtu.RtuMaster(
        serial.Serial(port, baudrate=baudrate, bytesize=8, parity='N', stopbits=1, xonxoff=0)
    )
    master.set_timeout(timeout)
    master.set_verbose(True)
    return master

# 读取串口数据
def read_serial_data(master):
    try:
        # 从寄存器读取 12 个值
        data = master.execute(1, cst.READ_INPUT_REGISTERS, 0, 12)

        # 数据转换
        processed_data = [
            ((value - 819) / (4095.0 - 816)) * 200.0
            for value in data
        ]

        # 返回最后 6 维
        return processed_data[-6:]
    except Exception as e:
        print(f"读取数据时发生错误: {e}")
        return None

# 连接控制器函数
def connect_controller():
    id = 0
    ip = "192.168.0.30"
    port = 8088
    ms = machine_status()

    # 连接控制器
    connect_result = fmc4030.FMC4030_Open_Device(id, c_char_p(bytes(ip, 'utf-8')), port)
    if connect_result != 0:
        print(f"设备连接失败，错误代码：{connect_result}")
        return None, None

    print("设备连接成功")
    return id, ms

# 关闭控制器函数
def close_controller(id):
    fmc4030.FMC4030_Close_Device(id)
    print("设备连接关闭")

# 封装的三轴控制函数
def control_three_axes(target_values):
    """
    控制三轴运动到指定气压目标值

    Args:
        target_values (list): 三个通道的目标气压值，例如 [21, 25, 32.0]

    Returns:
        tuple: (总运动时间, 最终误差列表)
    """
    # 初始化串口
    serial_master = initialize_serial('COM4')

    # 连接控制器
    id, ms = connect_controller()
    if id is None or ms is None:
        return None, None

    # PID参数
    P = 150
    D = 100

    # 初始化控制变量
    position = [0, 0, 0]
    velocity = [0, 0, 0]
    e_last = [0, 0, 0]

    # 记录程序开始时的初始时间
    start_time = time.perf_counter()

    while True:
        # 读取串口数据
        serial_data = read_serial_data(serial_master)
        if serial_data:
            # 计算误差 e
            e = [serial_data[-6 + i] - target_values[i] for i in range(3)]
            print(f"误差：{e}")

            # 更新位置和速度
            for i in range(3):
                if e[i] >= 0:
                    position[i] = -100 #这里100算是随便设了一个值，满足一个循环中不会走完就行
                else:
                    position[i] = 100

                velocity[i] = P * abs(e[i]) + D * abs(e_last[i] - e[i])

                # 判断是否停止轴运动
                if -0.3 <= e[i] <= 0.3:
                    print(f"误差在范围内，停止轴 {i}")
                    fmc4030.FMC4030_Stop_Single_Axis(id, i, 1)
                else:
                    print(f"更新轴 {i} 的位置和速度：位置={position[i]}, 速度={velocity[i]}")
                    fmc4030.FMC4030_Jog_Single_Axis(id, i, c_float(position[i]), c_float(velocity[i]), c_float(4000), c_float(4000), 1)

            # 更新上一时刻误差
            e_last = e[:]

            # 判断是否所有轴均满足停止条件
            if all(-0.3 <= e[i] <= 0.3 for i in range(3)):
                print("所有轴满足停止条件，退出循环")
                break

        time.sleep(0.01)  # 防止过高的轮询频率

    # 记录程序结束时间
    end_time = time.perf_counter()
    total_time = end_time - start_time

    # 关闭串口
    serial_master.close()

    # 关闭控制器连接
    close_controller(id)

    return total_time, e

# 示例调用
if __name__ == "__main__":
    target_values = [0, 0, 0]
    total_time, final_error = control_three_axes(target_values)
    if total_time is not None and final_error is not None:
        print(f"总运动时间：{total_time:.2f} 秒")
        print(f"最终误差：{final_error}")
