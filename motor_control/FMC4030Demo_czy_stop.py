# -*- coding: UTF-8 -*-
from ctypes import *
import time

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

ms = machine_status()

# 给控制器编号，此ID号唯一
id = 0
ip = "192.168.0.30"
port = 8088
# 连接控制器
connect_result = fmc4030.FMC4030_Open_Device(id, c_char_p(bytes(ip, 'utf-8')), port)
if connect_result == 0:
    print("设备连接成功")
else:
    print(f"设备连接失败，错误代码：{connect_result}")

# 记录开始时间
start_time = time.time()

# 为三个电机发送驱动指令
print(fmc4030.FMC4030_Jog_Single_Axis(id, 0, c_float(-400), c_float(80), c_float(300), c_float(3000), 1))  # 轴1
print(fmc4030.FMC4030_Jog_Single_Axis(id, 1, c_float(-400), c_float(80), c_float(300), c_float(3000), 1))  # 轴2
print(fmc4030.FMC4030_Jog_Single_Axis(id, 2, c_float(-400), c_float(80), c_float(300), c_float(4000), 1))  # 轴3

# 等待 2 秒后停止 2 号轴
time.sleep(2)  # 延迟 2 秒
print("2 秒已到，停止 2 号轴")
fmc4030.FMC4030_Stop_Single_Axis(id, 2, 1)  # 停止 2 号轴

# 等待三个轴全部运动完成
while True:
    # 检查三个轴的状态
    stop_axis_1 = fmc4030.FMC4030_Check_Axis_Is_Stop(id, 0)
    stop_axis_2 = fmc4030.FMC4030_Check_Axis_Is_Stop(id, 1)
    stop_axis_3 = fmc4030.FMC4030_Check_Axis_Is_Stop(id, 2)

    # 输出三个轴的实时位置
    fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
    print(f"轴1位置：{ms.realPos[0]:.2f}, 轴2位置：{ms.realPos[1]:.2f}, 轴3位置：{ms.realPos[2]:.2f}")

    # 如果三个轴都完成运动，则退出循环
    if stop_axis_1 == 1 and stop_axis_2 == 1 and stop_axis_3 == 1:
        break

    time.sleep(0.02)  # 防止过高的轮询频率

# 记录结束时间并计算总耗时
end_time = time.time()
print(f"三个轴运动完成，总耗时 {end_time - start_time:.2f} 秒")

# 关闭控制器连接，使用完成一定调用此函数释放资源
print("设备", fmc4030.FMC4030_Close_Device(id), "连接关闭")
