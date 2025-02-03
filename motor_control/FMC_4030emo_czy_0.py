
# # 控制器单轴运动
# start_time_axis_1 = time.time()  # 记录轴1开始时间
# print(fmc4030.FMC4030_Jog_Single_Axis(id, 0, c_float(400), c_float(800), c_float(3000), c_float(3000), 1))
# # 等待轴运行完成，过程中不断获取轴实际位置并输出
# while fmc4030.FMC4030_Check_Axis_Is_Stop(id, 0) == 0:
#     fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
#     print(ms.realPos[0])
#     # time.sleep(0.1)
# end_time_axis_1 = time.time()  # 记录轴1结束时间
# print(f"轴1运动完成，耗时 {end_time_axis_1 - start_time_axis_1:.2f} 秒")
#
# start_time_axis_2 = time.time()  # 记录轴2开始时间
# print(fmc4030.FMC4030_Jog_Single_Axis(id, 1, c_float(400), c_float(800), c_float(3000), c_float(3000), 1))
# # 等待轴运行完成，过程中不断获取轴实际位置并输出
# while fmc4030.FMC4030_Check_Axis_Is_Stop(id, 1) == 0:
#     fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
#     print(ms.realPos[1])
#     time.sleep(0.1)
# end_time_axis_2 = time.time()  # 记录轴2结束时间
# print(f"轴2运动完成，耗时 {end_time_axis_2 - start_time_axis_2:.2f} 秒")
#
# start_time_axis_3 = time.time()  # 记录轴3开始时间
# print(fmc4030.FMC4030_Jog_Single_Axis(id, 2, c_float(400), c_float(800), c_float(3000), c_float(3000), 1))
# # 等待轴运行完成，过程中不断获取轴实际位置并输出
# while fmc4030.FMC4030_Check_Axis_Is_Stop(id, 2) == 0:
#     fmc4030.FMC4030_Get_Machine_Status(id, pointer(ms))
#     print(ms.realPos[2])
#     time.sleep(0.1)
# end_time_axis_3 = time.time()  # 记录轴3结束时间
# print(f"轴3运动完成，耗时 {end_time_axis_3 - start_time_axis_3:.2f} 秒")

