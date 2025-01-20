import ctypes
import time
# 加载DLL

dll = ctypes.CDLL('FMC4030Lib-x64-20220329/FMC4030-Dll.dll')
#
# 定义函数原型（如果有必要，可以根据函数的实际参数类型定义）
# 假设 FMC4030_Open_Device 函数的签名是:
# int FMC4030_Open_Device(int device_index, char *ip, int port)

# 定义调用函数的参数类型和返回值类型
dll.FMC4030_Open_Device.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
dll.FMC4030_Open_Device.restype = ctypes.c_int

# 连接设备
device_index = 0  # 设备索引，0 表示第一个设备
ip_address = b'192.168.0.30'  # 设备 IP 地址，注意字符串需要转换为字节串
port = 8088  # 端口号

# 调用连接设备的函数
result = dll.FMC4030_Open_Device(device_index, ip_address, port)

# 输出结果
if result == 0:
    print("设备连接成功")
else:
    print(f"设备连接失败，错误代码：{result}")
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