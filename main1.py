import os
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import namedtuple

from actuator_state import calculate_state
from src.dataset.uav import PX4_ActuatorMotorDataReader
from src.dataset.uav import VOXL_IMUDataReader
from src.common.datatypes import SensorType
from data_manager import DataManager
from drone_extended_kalman_filter import DroneExtendedKalmanFilter
import actuator_state
from downsampling import process_imu, proces_actuator

if __name__ == "__main__":
    data_manager = DataManager()

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "UAV")

    #load and save data to manager
    motor = PX4_ActuatorMotorDataReader(
        path="data/UAV/log0001/px4/09_00_22_actuator_motors_0.csv",
        sensor_type=SensorType.PX4_ACTUATOR_MOTORS
    )

    voxl_imu0 = VOXL_IMUDataReader(
        path="data/UAV/log0001/run/mpa/imu0/data.csv",
        sensor_type=SensorType.VOXL_IMU0)

    data_manager.save_actuator_data(motor)
    data_manager.save_imu_data(voxl_imu0)

    filter = DroneExtendedKalmanFilter()

    voxl_datareader = iter(voxl_imu0)
    motor_datareader = iter(motor)

    prev_time_imu = -1
    prev_time_motor = -1

    i = 0

    t = []
    acc = []

    while True:
        try:
            #if i == 100:
                #break
            u = process_imu(voxl_datareader)

            acc.append(u.a[1])
            t.append(u.timestamp)

            #get dt for imu sensor
            if prev_time_imu == -1:
                prev_time_imu = u.timestamp - 0.001

            dt_imu = u.timestamp - prev_time_imu
            dt_imu /= 1e9

            prev_time_imu = u.timestamp

            print('dt_imu:')
            print(dt_imu)

            #get the next statevector x
            x = filter.predict(u, 0.035)

            actuator_data = proces_actuator(motor_datareader)

            if prev_time_motor == -1:
                prev_time_motor = actuator_data.timestamp - 0.001

            dt_actuator = actuator_data.timestamp - prev_time_motor
            dt_actuator /= 1e9

            prev_time_motor = actuator_data.timestamp

            print('dt_thrust:')
            print(dt_actuator)

            prev_state = data_manager.get_old_actuator_state()
            q = x[[6, 7, 8, 9]]

            s = actuator_state.calculate_state(actuator_data, q, prev_state, 0.035)
            data_manager.save_actuator_state(s)

            z = s[[0, 1, 2, 3, 4, 5]]

            filter.update_f(z)

            data_manager.save_imu_state(filter.get_x())

            data_manager.update_idx()
            i += 1

            #if (data_manager.imu_idx % 5000 == 0):
                #data_manager.show()

        except Exception as e:
            print(e)
            print('StopIteration')
            break

    data_manager.show()

    plt.plot(t, acc)
    plt.show()

