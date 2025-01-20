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
ip_address = b'192.168.0.35'  # 设备 IP 地址，注意字符串需要转换为字节串
port = 8088  # 端口号

# 调用连接设备的函数
connect_result = dll.FMC4030_Open_Device(device_index, ip_address, port)

# 输出结果
if connect_result == 0:
    print("设备连接成功")
else:
    print(f"设备连接失败，错误代码：{connect_result}")

# 定义函数原型
# 假设函数原型为:
# int FMC4030_Jog_Single_Axis(int id, int axis, double pos, double speed, double acc, double dec, int direction);
dll.FMC4030_Jog_Single_Axis.argtypes = [
    ctypes.c_int,    # id
    ctypes.c_int,    # axis
    ctypes.c_float, # pos
    ctypes.c_float, # speed
    ctypes.c_float, # acc
    ctypes.c_float, # dec
    ctypes.c_int     # direction
]
dll.FMC4030_Jog_Single_Axis.restype = ctypes.c_int  # 返回值类型为整数
# 定义函数原型
# 假设函数原型为:
# int FMC4030_Check_Axis_Is_Stop(int id, int axis);
dll.FMC4030_Check_Axis_Is_Stop.argtypes = [ctypes.c_int, ctypes.c_int]
dll.FMC4030_Check_Axis_Is_Stop.restype = ctypes.c_int  # 返回值类型为整数

# 调用函数
id = 0         # 设备 ID
axis = 0       # 控制轴
pos = 100.0     # 目标位置
speed = 500.0  # 速度
acc = 500.0    # 加速度
dec = 500.0    # 减速度
mode = 1  # 运行模式


move_result = dll.FMC4030_Jog_Single_Axis(id, 1, 200, speed, acc, dec, mode)
time.sleep(0.1)
move_result2 = dll.FMC4030_Jog_Single_Axis(id, 2, 500, speed, acc, dec, mode)
time.sleep(0.1)
# # result = dll.FMC4030_Jog_Single_Axis(0, 1, 200, 1000, 500, 500, 1)
# # 循环等待直到停止
# while True:
#     stop = dll.FMC4030_Check_Axis_Is_Stop(id, axis)
#     if stop == 1:  # 如果返回值为 1，表示轴已停止
#         print("轴已停止")
#         break
#     else:
#         print("轴未停止，等待中...")
#     time.sleep(0.1)  # 避免过高的轮询频率，稍作延迟
#
# # 检查返回值
# if result == 0:
#     print("单轴 Jog 操作成功")
# else:
#     print(f"单轴 Jog 操作失败，错误代码：{result}")


# # 定义函数原型
# # int FMC4030_Close_Device(int device_index);
# dll.FMC4030_Close_Device.argtypes = [ctypes.c_int]
# dll.FMC4030_Close_Device.restype = ctypes.c_int
# 断开设备连接
disconnect_result = dll.FMC4030_Close_Device(id)

# 检查返回结果
if disconnect_result == 0:
    print("设备",id,"断开连接成功")
else:
    print(f"设备断开连接失败，错误代码：{disconnect_result}")
