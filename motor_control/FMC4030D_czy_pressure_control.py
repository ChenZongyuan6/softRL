# -*- coding: UTF-8 -*-
from ctypes import *
import time
import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

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

# 主要程序
if __name__ == "__main__":
    ms = machine_status()

    # 给控制器编号，此 ID 号唯一
    id = 0
    ip = "192.168.0.30"
    port = 8088

    # 串口阈值设定
    thresholds = [5.0, 7.0, 10.0]  # 倒数第6、5、4维对应的阈值

    # 初始化串口
    serial_master = initialize_serial('COM4')

    # 连接控制器
    connect_result = fmc4030.FMC4030_Open_Device(id, c_char_p(bytes(ip, 'utf-8')), port)
    if connect_result == 0:
        print("设备连接成功")
    else:
        print(f"设备连接失败，错误代码：{connect_result}")

    # 为三个电机发送驱动指令
    print(fmc4030.FMC4030_Jog_Single_Axis(id, 0, c_float(1500), c_float(1000), c_float(4000), c_float(4000), 1))  # 轴1
    print(fmc4030.FMC4030_Jog_Single_Axis(id, 1, c_float(1500), c_float(600), c_float(4000), c_float(4000), 1))  # 轴2
    print(fmc4030.FMC4030_Jog_Single_Axis(id, 2, c_float(1500), c_float(300), c_float(4000), c_float(4000), 1))  # 轴3

    # 等待三个轴全部运动完成
    while True:
        # 检查三个轴的状态
        stop_axis_1 = fmc4030.FMC4030_Check_Axis_Is_Stop(id, 0)
        stop_axis_2 = fmc4030.FMC4030_Check_Axis_Is_Stop(id, 1)
        stop_axis_3 = fmc4030.FMC4030_Check_Axis_Is_Stop(id, 2)

        # 输出三个轴的实时位置
        fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
        # print(f"轴1位置：{ms.realPos[0]:.2f}, 轴2位置：{ms.realPos[1]:.2f}, 轴3位置：{ms.realPos[2]:.2f}")

        # 读取串口数据
        serial_data = read_serial_data(serial_master)
        if serial_data:
            print("串口数据：", serial_data)

            # 检查串口数据是否超过阈值，停止对应轴
            for axis, threshold, value in zip([0, 1, 2], thresholds, serial_data[-6:-3]):
                if value >= threshold:
                    print(f"串口数据超阈值，停止轴 {axis}：值 {value:.2f} >= 阈值 {threshold}")
                    fmc4030.FMC4030_Stop_Single_Axis(id, axis, 1)

        # 如果三个轴都完成运动，则退出循环
        if stop_axis_1 == 1 and stop_axis_2 == 1 and stop_axis_3 == 1:
            break
        # time.sleep(0.01)  # 防止过高的轮询频率

    # 关闭串口
    serial_master.close()

    # 关闭控制器连接
    print("设备", fmc4030.FMC4030_Close_Device(id), "连接关闭")
