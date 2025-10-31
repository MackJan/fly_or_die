import pandas as pd
import numpy as np
import sympy as sp
from filterpy.kalman import ExtendedKalmanFilter
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have Qt installed
import matplotlib.pyplot as plt
import os
import sys
import math
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Add src directory to path (if needed)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset.uav import PX4_GPSDataReader
from src.common.datatypes import SensorType

class IMU(ExtendedKalmanFilter):
    def __init__(self, filename):
        ExtendedKalmanFilter.__init__(self, dim_x=16, dim_z=6, dim_u=6)
        self.data = pd.read_csv(filename)

        self.length = self.data.shape[0]

        px, py, pz, vx, vy, vz, qw, qx, qy, qz = sp.symbols(
            'px py pz vx vy vz qw qx qy qz')
        bwx, bwy, bwz, bax, bay, baz = sp.symbols('bwx bwy bwz bax bay baz')

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

        ax, ay, az = sp.symbols('ax ay az')
        wx, wy, wz = sp.symbols('wx wy wz')

        am = sp.Matrix([
            ax, ay, az
        ])

        wm = sp.Matrix([
            wx, wy, wz
        ])

        g = sp.Matrix([
            0, 0, 9.81
        ])

        u = sp.Matrix([
            ax, ay, az, wx, wy, wz
        ])

        self.subs = {px: 0, py: 0, pz: 0,
                     vx: 0, vy: 0, vz: 0,
                     qw: 1, qx: 0, qy: 0, qz: 0,
                     bwx: 0, bwy: 0, bwz: 0,
                     bax: 0, bay: 0, baz: 0,

                     dt: 0}

        self.p_x, self.p_y, self.p_z = px, py, pz
        self.v_x, self.v_y, self.v_z = vx, vy, vz
        self.q_w, self.q_x, self.q_y, self.q_z = qw, qx, qy, qz
        self.bw_x, self.bw_y, self.bw_z = bwx, bwy, bwz
        self.ba_x, self.ba_y, self.ba_z = bax, bay, baz
        self.a_x, self.a_y, self.a_z = ax, ay, az
        self.w_x, self.w_y, self.w_z = wx, wy, wz
        self.dt = dt

        omega = wm - bw

        dq = self.delta_q_s(omega, dt)

        norm = sp.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)

        position_t = p + (v * dt + 0.5 * (R * (am - ba) + g) * dt ** 2)
        velocity_t = v + (R * (am - ba) + g) * dt
        quaternion_t = self.quat_multiply_s(dq, q)
        quaternion_t = quaternion_t / norm
        bw_t = bw
        ba_t = ba

        self.fxu = sp.Matrix.vstack(position_t, velocity_t, quaternion_t, bw_t, ba_t)

        #self.F_j = self.fxu.jacobian(sp.Matrix.vstack(x,u))
        self.F_j = self.fxu.jacobian(x)

        self.f_func = sp.lambdify((x, u, dt), self.fxu, 'numpy')

        #print(self.F_j.free_symbols)
        #print(self.F_j.atoms(sp.Derivative))
        self.F_j = self.F_j.doit()
        self.F_j = self.F_j.xreplace({d: 0 for d in self.F_j.atoms(sp.Derivative)})
        self.F_func = sp.lambdify((x, u, dt), self.F_j, 'numpy')

        cols = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'bwx', 'bwy', 'bwz', 'bax', 'bay', 'baz']

        self.states = pd.DataFrame(columns=cols,dtype=float)

        self.states.loc[0] = np.array([0., 0., 0.,  # position
                                       0., 0., 0.,  # velocity
                                       1., 0., 0., 0.,  # quaternion
                                       0., 0., 0.,  # gyro bias
                                       0.01, -0.02, 0.])

        self.idx = 0

        self.g = np.array([0., 0., 9.81])

        self.dt = 0

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

    def input(self):
        self.idx += 1
        if self.idx >= self.length:
            return False

        row = self.data.iloc[self.idx]
        am = np.array([row['AX(m/s2)'], row['AY(m/s2)'], row['AZ(m/s2)']])
        wm = np.array([row['GX(rad/s)'], row['GY(rad/s)'], row['GZ(rad/s)']])
        self.predict((am, wm))

        return True

    def predict(self, u):
        # ensure previous state exists
        if self.idx not in self.states.index:
            self.states.loc[self.idx] = self.states.loc[self.idx - 1]

        dt_ns = (self.data.iloc[self.idx]['timestamp(ns)'] -
                 self.data.iloc[self.idx - 1]['timestamp(ns)'])
        self.dt = dt_ns / 1e9

        if self.dt <= 0:
            # nothing to propagate on this step (first sample or bad timestamps)
            self.x = self.states.iloc[self.idx - 1].values.astype(float)
            self.states.loc[self.idx] = self.x
            return

        # use previous state for prediction
        x_prev = self.states.iloc[self.idx - 1].values.astype(float)
        am, wm = u
        u_vec = np.concatenate([am, wm])

        #debugging
        am, wm = u
        q = x_prev[6:10]  # [qw,qx,qy,qz] as you use
        R = self.mat_from_quat(q)  # numeric 3x3
        
        acc_body = am  # measured specific force (what you read from IMU)
        #acc_world = R @ (acc_body - x_prev[13:16]) - self.g  # what you use
        acc_world = R @ (acc_body) - self.g  # what you use

        #print("idx", self.idx, "dt(s)", self.dt)
        #print("am (body)", np.round(am, 3))
        #print("gyro (rad/s)", np.round(wm, 4))
        #print("quat (qw,qx,qy,qz)", np.round(q, 5))
        #print("R@am", np.round(R @ am, 3))
        #print("acc_world = R@(am-bias)+g", np.round(acc_world, 3))
        #print("pos prev[:3]", np.round(x_prev[:3], 3))

        # nonlinear prediction via lambdified function
        x_pred = np.array(self.f_func(x_prev, u_vec, self.dt), dtype=float).flatten()
        if(self.idx%100==1):
            print("pos pred[:3]", np.round(x_pred[:3], 3))
        q = x_pred[6:10]
        q = q / (np.linalg.norm(q) + 1e-12)
        x_pred[6:10] = q

        # store predicted state
        self.x = x_pred
        self.states.loc[self.idx] = self.x

        # numeric Jacobian (linearization point is x_prev)
        F = np.array(self.F_func(x_prev, u_vec, self.dt), dtype=float)
        #print("F", F)
        # sanity checks (optional, remove prints after debugging)
        # print("dt", self.dt, "x_prev[:6]", x_prev[:6], "u", u_vec)
        # print("x_pred[:6]", x_pred[:6])
        # print("F shape", F.shape, "P shape", self.P.shape)

        Q = np.zeros((16, 16))
        Q[0:3, 0:3] += np.eye(3) * 1e-1  # position
        Q[3:6, 3:6] += np.eye(3) * 1e-1  # velocity
        Q[6:10, 6:10] += np.eye(4) * 1e-2  # quat
        Q[10:16, 10:16] += np.eye(6) * 1e-8  # biases
        self.Q = Q

        # propagate covariance
        self.P = F @ self.P @ F.T + self.Q
        #print("P:", self.P)     

    def mat_from_quat(self, q):
        qw, qx, qy, qz = q
        return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

    def get_quaternion(self):
        return np.array(self.states.iloc[self.idx][['qw','qx','qy','qz']])

    #def print_pos(self):
        #print(self.x[[0, 1, 2]])


    def plot_3d_trajectory(self, z_measurements=None):
        """
        Plot the raw GPS trajectory, filtered trajectory, and z measurements in 3D.
        
        Args:
            z_measurements: Array of z measurement positions from the main loop
        """
        # Get filtered positions (EKF data)
        filtered_positions = np.array(self.states[['px', 'py', 'pz']])
        #for i in range(len(self.states)):
        #    if i%100==0:
        #        print(self.states.iloc[i])
        # Get raw GPS positions (from the RawGPS3DPlotter logic)
        raw_gps_positions = []
        ref_lat = None
        ref_lon = None
        ref_alt = None
        
        # Initialize GPS data reader
        gps_file = "data/UAV/log0001/px4/09_00_22_sensor_gps_0.csv"
        gps_reader = PX4_GPSDataReader(
            path=gps_file,
            sensor_type=SensorType.PX4_GPS
        )
        
        # Function to convert GPS to XYZ (same as in RawGPS3DPlotter)
        def lat_lon_alt_to_xyz(lat, lon, alt, ref_lat, ref_lon, ref_alt):
            if ref_lat is None:
                return 0.0, 0.0, 0.0, lat, lon, alt
            
            lat_rad = math.radians(lat)
            ref_lat_rad = math.radians(ref_lat)
            R = 6371000  # Earth radius in meters
            
            x = R * math.cos(ref_lat_rad) * math.radians(lon - ref_lon)
            y = R * math.radians(lat - ref_lat)
            z = alt - ref_alt
            
            return x, y, z, ref_lat, ref_lon, ref_alt
        
        # Read raw GPS data
        for data in gps_reader:
            lat = data.lat / 1e7
            lon = data.lon / 1e7
            alt = data.alt / 1e3
            
            # Skip invalid GPS data
            if lat == 0.0 or lon == 0.0 or abs(lat) < 1e-6 or abs(lon) < 1e-6:
                continue
            
            x, y, z, ref_lat, ref_lon, ref_alt = lat_lon_alt_to_xyz(
                lat, lon, alt, ref_lat, ref_lon, ref_alt
            )
            raw_gps_positions.append([x, y, z])
        
        raw_gps_positions = np.array(raw_gps_positions)
        
        # Create 3D plot
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot raw GPS trajectory (blue)
        ax.plot(raw_gps_positions[:, 0], raw_gps_positions[:, 1], raw_gps_positions[:, 2],
                'b.-', alpha=0.3, label='Raw GPS Trajectory', markersize=2, linewidth=1)
        
        # Plot filtered/EKF trajectory (red)
        ax.plot(filtered_positions[:, 0], filtered_positions[:, 1], filtered_positions[:, 2],
                'r-', alpha=0.8, label='EKF Filtered Trajectory', markersize=3, linewidth=2)
        
        # Plot z measurements trajectory (green/cyan)
        if z_measurements is not None and len(z_measurements) > 0:
            ax.plot(z_measurements[:, 0], z_measurements[:, 1], z_measurements[:, 2],
                    'g-', alpha=0.6, label='Thrust Model Measurements', markersize=2, linewidth=1.5)
        
        # Mark start point (green)
        ax.scatter(filtered_positions[0, 0], filtered_positions[0, 1], filtered_positions[0, 2],
                c='green', s=200, marker='o', label='Start Point', edgecolors='black', linewidths=2)
        
        # Mark end point (red)
        ax.scatter(filtered_positions[-1, 0], filtered_positions[-1, 1], filtered_positions[-1, 2],
                c='red', s=200, marker='s', label='End Point', edgecolors='black', linewidths=2)
        
        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (meters)', fontsize=12, fontweight='bold')
        ax.set_title('Trajectory Comparison: Raw GPS vs EKF vs Thrust Model', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        
        # Set equal aspect ratio using combined data
        all_positions = [raw_gps_positions, filtered_positions]
        if z_measurements is not None and len(z_measurements) > 0:
            all_positions.append(z_measurements)
        all_positions = np.vstack(all_positions)
        
        max_range = np.array([
            all_positions[:, 0].max() - all_positions[:, 0].min(),
            all_positions[:, 1].max() - all_positions[:, 1].min(),
            all_positions[:, 2].max() - all_positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
        mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
        mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"Raw GPS Points: {len(raw_gps_positions)}\n"
        stats_text += f"EKF Filtered Points: {len(filtered_positions)}\n"
        if z_measurements is not None and len(z_measurements) > 0:
            stats_text += f"Thrust Model Points: {len(z_measurements)}\n"
        stats_text += f"Start: ({filtered_positions[0, 0]:.2f}, {filtered_positions[0, 1]:.2f}, {filtered_positions[0, 2]:.2f})\n"
        stats_text += f"End: ({filtered_positions[-1, 0]:.2f}, {filtered_positions[-1, 1]:.2f}, {filtered_positions[-1, 2]:.2f})"
        
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig

    def show(self, z_measurements=None):
        """Display the plot."""
        self.plot_3d_trajectory(z_measurements)
        plt.show()

class Thrust:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

        self.length = self.data.shape[0]

        self.RPM_max = 15000
        self.idx = 0

        self.m = 1.075
        self.b = 0.22 #0.000022

        self.data[['p_x', 'p_y', 'p_z']] = 0.
        self.data[['v_x', 'v_y', 'v_z']] = 0.
        self.data[['a_x', 'a_y', 'a_z']] = 0.

        #print(self.data.head)
        #print(self.data.columns)

    def calculate_state(self, q):
        qw, qx, qy, qz = q
        self.idx += 1

        if self.idx >= self.length:
            return False

        time = self.data.iloc[self.idx]['timestamp']
        c0 = self.data.iloc[self.idx]['control[0]']
        c1 = self.data.iloc[self.idx]['control[1]']
        c2 = self.data.iloc[self.idx]['control[2]']
        c3 = self.data.iloc[self.idx]['control[3]']

        omega_max = self.RPM_max * (2. * np.pi / 60.)

        #angular velocities
        w0 = c0 * omega_max
        w1 = c1 * omega_max
        w2 = c2 * omega_max
        w3 = c3 * omega_max

        #thrusts
        t0 = w0 #** 2
        t1 = w1 #** 2 
        t2 = w2 #** 2 
        t3 = w3 #** 2

        #total thrust
        t = self.b *(t0 + t1 + t2 + t3)
        T = np.array([0., 0., t])

        R = self.quat_to_mat(qw, qx, qy, qz)
        #print("R", R)
        F = R @ T

        A = F / self.m

        dt = self.data.iloc[self.idx]['timestamp'] - self.data.iloc[self.idx - 1]['timestamp']
        dt /= 1e9

        self.data.loc[self.idx, ['a_x', 'a_y', 'a_z']] = A
        self.data.loc[self.idx, ['v_x', 'v_y', 'v_z']] = (
                self.data.loc[self.idx - 1, ['v_x', 'v_y', 'v_z']].values +
                self.data.loc[self.idx - 1, ['a_x', 'a_y', 'a_z']].values * dt
        )
        self.data.loc[self.idx, ['p_x', 'p_y', 'p_z']] = (
                self.data.loc[self.idx - 1, ['p_x', 'p_y', 'p_z']].values +
                self.data.loc[self.idx-1, ['v_x', 'v_y', 'v_z']].values * dt +
                self.data.loc[self.idx-1, ['a_x', 'a_y', 'a_z']].values *0.5 * dt ** 2
        )

        return True

    def get_state(self):
        return np.array( self.data.iloc[self.idx][['p_x', 'p_y', 'p_z', 'v_x', 'v_y', 'v_z']])

    def quat_to_mat(self, qw, qx, qy, qz):
        return np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

def Hx(x):
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])

def H_f(x):
    H = np.zeros((6, 16))
    np.fill_diagonal(H, 1)
    return H



def main():
    imu = IMU('data/UAV/log0001/run/mpa/imu0/data.csv')
    thrust = Thrust('data/UAV/log0001/px4/09_00_22_actuator_motors_0.csv')

    imu.x = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
    imu.P = np.zeros((16, 16))
    np.fill_diagonal(imu.P, 0.1)
    imu.R = np.zeros((6, 6))
    np.fill_diagonal(imu.R, 100)

    i=0
    # Initialize array to collect z measurements
    z_measurements = []

    while True:
        #print("step", i)
        i+=1
        if not imu.input():
            break
        if i%5==0:
            if not thrust.calculate_state(imu.get_quaternion()):
                break
            z = thrust.get_state()
            # Collect z measurements (first 3 elements are position x, y, z)
            z_measurements.append(z[:3])  # Only append x, y, z position
            # Print the most recently appended z values
            #print(f"Iteration {i}: z = {z[:3]}")
        
            imu.update(z, HJacobian=H_f, Hx=Hx)

        #imu.print_pos()

    
    # Convert to numpy array
    z_measurements = np.array(z_measurements)

    imu.show(z_measurements=z_measurements)

if __name__=="__main__":
    main()
