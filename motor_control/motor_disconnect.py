import ctypes
import time
# 加载DLL

dll = ctypes.CDLL('FMC4030Lib-x64-20220329/FMC4030-Dll.dll')
#

# 定义函数原型
# int FMC4030_Close_Device(int device_index);
dll.FMC4030_Close_Device.argtypes = [ctypes.c_int]
dll.FMC4030_Close_Device.restype = ctypes.c_int

# 断开设备连接
device_index = 0  # 设备索引
result = dll.FMC4030_Close_Device(device_index)

# 检查返回结果
if result == 0:
    print("设备断开连接成功")
else:
    print(f"设备断开连接失败，错误代码：{result}")