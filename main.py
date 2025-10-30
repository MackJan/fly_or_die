import pandas as pd
import numpy as np
import sympy as sp
from filterpy.kalman import ExtendedKalmanFilter
from src.common.ekf_model import fx_numpy, Jx_numpy

class IMU(ExtendedKalmanFilter):
    def __init__(self, filename):
        ExtendedKalmanFilter.__init__(self, dim_x=16, dim_z=6, dim_u=6)
        self.data = pd.read_csv(filename)

        # build (or import) the symbolic-derived numeric functions
        self.define()

        # numeric lambdified functions (from the notebook / ekf_model)
        self.fx_func = fx_numpy
        self.Jx_func = Jx_numpy

        cols = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'bwx', 'bwy', 'bwz', 'bax', 'bay', 'baz']

        self.states = pd.DataFrame(columns=cols)

        self.states.loc[0] = np.array([0, 0, 0,  # position
                                       0, 0, 0,  # velocity
                                       1, 0, 0, 0,  # quaternion
                                       0, 0, 0,  # gyro bias
                                       0, 0, 0])

        self.idx = 0

        # gravity as vector
        self.g = np.array([0., 0., -9.81])

        # default covariance and process noise (can be overridden)
        self.P = np.eye(16) * 0.1
        self.Q = np.eye(16) * 1e-3

    def quat_multiply_s(self, dq, q):
        qw1, qx1, qy1, qz1 = dq
        qw2, qx2, qy2, qz2 = q
        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
        return sp.Matrix([qw, qx, qy, qz])

    def delta_q_s(self, omega, dt):
        theta = omega.norm() * dt  # calculating the total rotation angle
        if theta == 0:  # there is no rotation if theta =0
            return sp.Matrix([1, 0, 0, 0])  # nullpöörlemine
        axis = omega / omega.norm()  # normalize angular velocity to get rotation axis
        w = sp.cos(theta / 2)  # scalar part of quaternion
        xyz = axis * sp.sin(theta / 2)  # vector part of quaternion
        return sp.Matrix([w, *xyz])  # return quaternion representing the rotation

    def define(self):
        px, py, pz, vx, vy, vz, qw, qx, qy, qz = sp.symbols(
            'p_{x} p_{y} p_{z} v_{x} v_{y} v_{z} q_{w} q_{x} q_{y} q_{z}')
        bwx, bwy, bwz, bax, bay, baz = sp.symbols('b_{wx} b_{wy} b_{wz} b_{ax} b_{ay} b_{az}')

        x = sp.Matrix([px, py, pz, vx, vy, vz, qw, qx, qy, qz, bwx, bwy, bwz, bax, bay, baz])

        p = sp.Matrix([
            px, py, pz
        ])

        v = sp.Matrix([
            vx, vy, vz
        ])
        q = sp.Matrix([
            qw, qx, qy, qz
        ])

        bw = sp.Matrix([
            bwx, bwy, bwz,
        ])

        ba = sp.Matrix([
            bax, bay, baz
        ])

        dt = sp.symbols('dt')

        R = sp.Matrix([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        ax, ay, az = sp.symbols('a_{x} a_{y} a_{z}')
        wx, wy, wz = sp.symbols('w_{x} w_{y} w_{z}')

        am = sp.Matrix([
            ax, ay, az
        ])

        wm = sp.Matrix([
            wx, wy, wz
        ])

        g = sp.Matrix([
            0, 0, -9.81
        ])

        u = sp.Matrix([
            ax, ay, az, wx, wy, wz
        ])

        omega = wm - bw

        dq = self.delta_q_s(omega, dt)

        norm = sp.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)

        position_t = p + (v * dt + 0.5 * (R * (am - ba) + g) * dt ** 2)
        velocity_t = v + (R * (am - ba) + g) * dt
        quaternion_t = self.quat_multiply_s(dq, q)
        quaternion_t = quaternion_t / norm
        bw_t = bw
        ba_t = ba

        self.fxu = sp.Matrix.vstack( position_t, velocity_t, quaternion_t, bw_t, ba_t)

        self.F_j = self.fxu.jacobian([x,u])

    def input(self):
        self.idx += 1

        row = self.data.iloc[self.idx]
        am = np.array([row['AX(m/s2)'], row['AY(m/s2)'], row['AZ(m/s2)']])
        wm = np.array([row['GX(rad/s)'], row['GY(rad/s)'], row['GZ(rad/s)']])
        self.predict((am, wm))

    def predict(self, u):
        # Ensure numeric state vector exists (filterpy uses self.x)
        if not hasattr(self, 'x') or self.x is None:
            # try to initialize from stored DataFrame state if present
            if 0 in self.states.index:
                self.x = self.states.iloc[0].values.astype(float)
            else:
                self.x = np.zeros(16, dtype=float)
                self.x[6] = 1.0

        # compute dt in seconds from nanosecond timestamps
        dt = (self.data.iloc[self.idx]['timestamp(ns)'] -
              self.data.iloc[self.idx - 1]['timestamp(ns)']) / 1e9

        am, wm = u
        u_vec = np.concatenate([np.asarray(am).ravel(), np.asarray(wm).ravel()])

        # predict state and linearization using lambdified functions
        x_pred = self.fx_func(self.x, u_vec, dt)
        F = self.Jx_func(self.x, u_vec, dt)

        # normalize quaternion part
        q = x_pred[6:10]
        q = q / np.linalg.norm(q)
        x_pred[6:10] = q

        # commit prediction
        self.x = x_pred
        # log predicted state
        self.states.loc[self.idx] = x_pred
        print("F:", F.shape)
        print("P:", self.P.shape)
        print("Q:", self.Q.shape)

        # covariance update (discrete-time)
        self.P = F @ self.P @ F.T + self.Q * dt

    def quat_multiply(self, dq, q):
        qw1, qx1, qy1, qz1 = dq
        qw2, qx2, qy2, qz2 = q
        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
        return np.array([qw, qx, qy, qz])

    def delta_q(self, omega, dt):
        theta = np.linalg.norm(omega) * dt  # calculating the total rotation angle
        if theta == 0:  # there is no rotation if theta =0
            return np.array([1, 0, 0, 0])  # nullpöörlemine
        axis = omega / np.linalg.norm(omega)  # normalize angular velocity to get rotation axis
        w = np.cos(theta / 2)  # scalar part of quaternion
        xyz = axis * np.sin(theta / 2)  # vector part of quaternion
        return np.array([w, *xyz])  # return quaternion representing the rotation

    def mat_from_quat(self, q):
        qw, qx, qy, qz = q
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

    def f(self, x, u):
        p = np.array(x[['px', 'py', 'pz']], dtype=float).flatten()  # shape (3,)
        v = np.array(x[['vx', 'vy', 'vz']], dtype=float).flatten()  # shape (3,)
        q = np.array(x[['qw', 'qx', 'qy', 'qz']], dtype=float).flatten()
        ba = np.array(x[['bax', 'bay', 'baz']], dtype=float).flatten()
        bw = np.array(x[['bwx', 'bwy', 'bwz']], dtype=float).flatten()

        am, wm = u

        R = self.mat_from_quat(q)

        dt = self.data.iloc[self.idx]['timestamp(ns)'] - self.data.iloc[self.idx- 1]['timestamp(ns)']

        omega = wm - bw

        dq = self.delta_q(omega, dt)

        norm = np.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)

        position_t = p + (v * dt + 0.5 * (R * (am - ba) + self.g) * dt ** 2)
        velocity_t = v + (R * (am - ba) + self.g) * dt
        quaternion_t = self.quat_multiply(dq, q)
        quaternion_t = quaternion_t / norm

        bw_t = bw  # gyroscope bias at time t
        ba_t = ba  # accelometer bias at time t

        position_t = (position_t)
        velocity_t = (velocity_t)
        quaternion_t = np.ravel(quaternion_t)
        bw_t = np.ravel(bw_t)
        ba_t = np.ravel(ba_t)

        print(position_t.size)
        print(velocity_t.size)
        print(quaternion_t.size)
        print(bw_t.size)
        print(ba_t.size)

        # Entire drone kinematics equation
        x = np.concatenate((
            position_t, velocity_t, quaternion_t, bw_t, ba_t
        ))

        return x

    def get_quaternion(self):
        return np.array(self.states.iloc[self.idx][['qw','qx','qy','qz']])

class Thrust:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.RPM_max = 15000
        self.idx = 0

        self.m = 1.075
        self.b = 0.000022

        self.data[['p_x', 'p_y', 'p_z']] = 0
        self.data[['v_x', 'v_y', 'v_z']] = 0
        self.data[['a_x', 'a_y', 'a_z']] = 0

    def calculate_state(self, q):
        qw, qx, qy, qz = q
        self.idx += 1
        print(self.data.columns)

        time = self.data['timestamp'][self.idx]
        c0 = self.data['control[0]'][self.idx]
        c1 = self.data['control[1]'][self.idx]
        c2 = self.data['control[2]'][self.idx]
        c3 = self.data['control[3]'][self.idx]

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

        F = R @ T

        A = F / self.m

        dt = self.data['timestamp'][self.idx] - self.data['timestamp'][self.idx - 1]

        self.data.loc[self.idx, ['a_x', 'a_y', 'a_z']] = A
        self.data.loc[self.idx, ['v_x', 'v_y', 'v_z']] = (self.data.loc[self.idx-1, ['v_x', 'v_y', 'v_z']].values +
                                                          self.data.loc[self.idx-1, ['a_x', 'a_y', 'a_z']].values * dt)
        self.data.loc[self.idx, ['p_x', 'p_y', 'p_z']] = (self.data.loc[self.idx-1, ['p_x', 'p_y', 'p_z']].values +
                                                          self.data.loc[self.idx-1, ['v_x', 'v_y', 'v_z']].values * dt +
                                                          self.data.loc[self.idx-1, ['a_x', 'a_y', 'a_z']].values * dt ** 2)

    def get_state(self):
        return np.array( self.data[['p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z'][self.idx]])

    def quat_to_mat(self, qw, qx, qy, qz):
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

def Hx(x):
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])

def main():
    imu = IMU('data/UAV/log0001/run/mpa/imu0/data.csv')
    thrust = Thrust('data/UAV/log0001/px4/09_00_22_actuator_motors_0.csv')

    imu.x = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    #imu.P = np.diag([.1, .1, .1])
    #imu.R = np.diag([.1, .1, .1])
    imu.P = np.eye(16) * 0.1   # keep consistent with 16x16 state dimension
    imu.R = np.eye(6) * 0.1    # measurement covariance (6x6, matches dim_z)

    while True:
        imu.input()

        thrust.calculate_state(imu.get_quaternion())
        z = thrust.get_state()

        H = np.zeros((6, 16))
        np.fill_diagonal(H, 1)
        imu.update(z, HJacobian=lambda x: H, Hx=Hx)
#        imu.update(z, HJacobian=H, Hx=Hx)


if __name__=="__main__":
    main()
