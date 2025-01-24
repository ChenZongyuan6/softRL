import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu

def initialize_serial(port, baudrate=115200, timeout=2.0):
    """
    初始化串口通信。
    """
    master = modbus_rtu.RtuMaster(
        serial.Serial(port, baudrate=baudrate, bytesize=8, parity='N', stopbits=1, xonxoff=0)
    )
    master.set_timeout(timeout)
    master.set_verbose(True)
    return master

def read_serial_data(master):
    """
    读取串口数据并返回最后 6 维值。

    Args:
        master: 已初始化的 modbus_rtu.RtuMaster 对象。

    Returns:
        list: 数据的最后 6 维值。
    """
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

# 示例使用
if __name__ == "__main__":
    # 初始化串口
    master = initialize_serial('COM4')

    # 读取一次数据
    result = read_serial_data(master)
    if result:
        print("读取到的数据最后6维:", result)

    # 关闭串口
    master.close()
