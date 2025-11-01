import os
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from actuator_state import calculate_state
from src.dataset.uav import PX4_ActuatorMotorDataReader
from src.dataset.uav import VOXL_IMUDataReader
from src.common.datatypes import SensorType
from data_manager import DataManager
from drone_extended_kalman_filter import DroneExtendedKalmanFilter
import actuator_state

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
    filter_datareader = iter(motor)

    prev_time = -1

    while True:
        try:
            u = next(voxl_datareader)

            if (prev_time == -1):
                prev_time = u.timestamp - 0.001
            dt = u.timestamp - prev_time
            dt /= 1e9

            prev_time = u.timestamp

            x = filter.predict(u, dt)

            data_manager.save_imu_state(x)

            actuator_data = next(filter_datareader)
            print(actuator_data)

            data_manager.print_actuator_states()

            prev_state = data_manager.get_old_actuator_state()
            q = x[[6, 7, 8, 9]]

            #s = actuator_state.calculate_state(actuator_data, q, prev_state)
            #data_manager.save_actuator_state(s)

            #z = s[[0, 1, 2, 3, 4, 5]]
            #print(z)

            #filter.update_f(z)

            data_manager.update_idx()

        except StopIteration:
            break

    data_manager.show()