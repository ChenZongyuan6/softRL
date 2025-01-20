# -*- coding: UTF-8 -*-
from ctypes import *
import time
import socket

# fmc4030 = windll.LoadLibrary('G:/Project/AMC4030_V2/DLL/FMC4030-Dll/Release/FMC4030-Dll.dll')
fmc4030 = windll.LoadLibrary('FMC4030Lib-x64-20220329/FMC4030-DLL.dll')

# 定义设备状态类，用于获取设备状态数据
# struct machine_status{
#    float realPos[3];
#   float realSpeed[3];
#   unsigned int inputStatus;
#   unsigned int outputStatus;
#   unsigned int limitNStatus;
#  unsigned int limitPStatus;
#  unsigned int machineRunStatus;
#  unsigned int axisStatus[MAX_AXIS];
#  unsigned int homeStatus;
#  char file[20][30];
#  };
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

# 控制器单轴运动
start_time_axis_1 = time.time()  # 记录轴1开始时间
print(fmc4030.FMC4030_Jog_Single_Axis(id, 0, c_float(-400), c_float(400), c_float(2000), c_float(2000), 1))
# 等待轴运行完成，过程中不断获取轴实际位置并输出
while fmc4030.FMC4030_Check_Axis_Is_Stop(id, 0) == 0:
    fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
    print(ms.realPos[0])
    time.sleep(0.1)
end_time_axis_1 = time.time()  # 记录轴1结束时间
print(f"轴1运动完成，耗时 {end_time_axis_1 - start_time_axis_1:.2f} 秒")

start_time_axis_2 = time.time()  # 记录轴2开始时间
print(fmc4030.FMC4030_Jog_Single_Axis(id, 1, c_float(-400), c_float(500), c_float(500), c_float(500), 1))
# 等待轴运行完成，过程中不断获取轴实际位置并输出
while fmc4030.FMC4030_Check_Axis_Is_Stop(id, 1) == 0:
    fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
    print(ms.realPos[1])
    time.sleep(0.1)
end_time_axis_2 = time.time()  # 记录轴2结束时间
print(f"轴2运动完成，耗时 {end_time_axis_2 - start_time_axis_2:.2f} 秒")
# 关闭控制器连接，使用完成一定调用此函数释放资源
print("设备", fmc4030.FMC4030_Close_Device(id),"连接关闭")
