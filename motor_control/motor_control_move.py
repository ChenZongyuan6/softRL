import ctypes

# 加载 DLL
dll = ctypes.CDLL('FMC4030-Dll.dll')

# 定义函数原型
# 假设函数原型为:
# int FMC4030_Jog_Single_Axis(int id, int axis, double pos, double speed, double acc, double dec, int direction);
dll.FMC4030_Jog_Single_Axis.argtypes = [
    ctypes.c_int,    # id
    ctypes.c_int,    # axis
    ctypes.c_double, # pos
    ctypes.c_double, # speed
    ctypes.c_double, # acc
    ctypes.c_double, # dec
    ctypes.c_int     # direction
]
dll.FMC4030_Jog_Single_Axis.restype = ctypes.c_int  # 返回值类型为整数

# 调用函数
id = 0         # 设备 ID
axis = 0       # 控制轴
pos = 20.0     # 目标位置
speed = 500.0  # 速度
acc = 500.0    # 加速度
dec = 500.0    # 减速度
direction = 1  # 方向

# 执行 Jog 操作
result = dll.FMC4030_Jog_Single_Axis(id, axis, pos, speed, acc, dec, direction)

# 检查返回值
if result == 0:
    print("单轴 Jog 操作成功")
else:
    print(f"单轴 Jog 操作失败，错误代码：{result}")
