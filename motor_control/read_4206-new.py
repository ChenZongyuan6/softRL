import serial
import modbus_tk.defines as cst
from modbus_tk import modbus_rtu
import time

def waitSerial(hour, min, sec):
    return hour*3600+min*60+sec

second = waitSerial(0,0,0.1)
master = modbus_rtu.RtuMaster(serial.Serial('COM4', baudrate=115200, bytesize=8, parity='N', stopbits=1, xonxoff=0))
master.set_timeout(2.0)
master.set_verbose(True)

while True:
    while master.close():
        pass
    else:
        data = master.execute(1, cst.READ_INPUT_REGISTERS, 0, 12)
        time.sleep(second)
        index = 0
        datVal = [0 for i in range(100)]

        temp = [0 for i in range(len(data))]
        for index in range(len(data)):
            temp[index] = ((data[index] -819)/ (4095.0-816)) * 200.0
            # datVal[index] = data[index]&hex(4095)
            datVal[index] = data[index] & 0xFFF

        print(temp)