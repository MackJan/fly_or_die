import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter

class IMU:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.filter = KalmanFilter()


class Thrust:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.RPM_max = 15000
        self.idx = 0

        self.m = 1.075
        self.b = 0.000022

        self.data[['p_x', 'p_y', 'p_z'][self.idx]] = np.array([0, 0, 0])
        self.data[['v_x', 'v_y', 'v_z'][self.idx]] = np.array([0, 0, 0])
        self.data[['a_x', 'a_y', 'a_z'][self.idx]] = np.array([0, 0, 0])

    def calculate_state(self, qw, qx, qy, qz):
        self.idx += 1

        time = self.data['timestamp'][self.idx]
        c0 = self.data['control0'][self.idx]
        c1 = self.data['control1'][self.idx]
        c2 = self.data['control2'][self.idx]
        c3 = self.data['control3'][self.idx]

        omega_max = self.RPM_max * (2 * np.pi / 60)

        #angular velocities
        w0 = c0 * omega_max
        w1 = c1 * omega_max
        w2 = c2 * omega_max
        w3 = c3 * omega_max

        #thrusts
        t0 = w0 * self.b
        t1 = w1 * self.b
        t2 = w2 * self.b
        t3 = w3 * self.b

        #total thrust
        t = t0 + t1 + t2 + t3
        T = np.array([0,0,t])

        R = self.quat_to_mat(qw, qx, qy, qz)

        F = R @ np.array([0.,0.,T])

        A = F / self.m

        dt = self.data['timestamp'][self.idx] - self.data['timestamp'][self.idx - 1]

        self.data[['a_x', 'a_y', 'a_z'][self.idx]] = A
        self.data[['v_x', 'v_y', 'v_z'][self.idx]] = (self.data[['v_x', 'v_y', 'v_z'][self.idx -1]] +
                                                      self.data[['a_x', 'a_y', 'a_z'][self.idx - 1]] * dt)
        self.data[['p_x', 'p_y', 'p_z'][self.idx]] = (self.data[['p_x', 'p_y', 'p_z'][self.idx - 1]] +
                                                      self.data[['v_x', 'v_y', 'v_z'][self.idx - 1]] * dt +
                                                      self.data[['a_x', 'a_y', 'a_z'][self.idx - 1]] * dt ** 2)

    def get_state(self):
        return np.array( self.data[['p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z'][self.idx]])

    def quat_to_mat(self, qw, qx, qy, qz):
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

def main():
    imu = IMU('data/UAV/log0001/run/mpa/data.csv')
    thrust = Thrust('data/UAV/log0001/px4/09_00_22_actuator_motors_0.csv')

if __name__=="__main__":
    main()
