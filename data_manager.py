import os
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from src.dataset.uav import PX4_ActuatorMotorDataReader
from src.dataset.uav import VOXL_IMUDataReader
from src.common.datatypes import SensorType

class DataManager:
    def __init__(self):
        self.imu_data = []
        self.actuator_data = []

        self.imu_states = []
        self.imu_states.append(np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        self.actuator_states = []
        self.actuator_states.append(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]))

        self.imu_idx = 1
        self.actuator_idx = 1

        self.data_length = 0

        self.imu_positions = []
        self.thrust_positions = []

    def end(self):
        return self.imu_idx >= self.data_length

    def update_idx(self):
        self.imu_idx += 1
        self.actuator_idx += 1

    def get_u(self):
        return self.imu_data[self.imu_idx]

    def get_actuator_data(self):
        return self.actuator_data[self.actuator_idx]

    def get_old_actuator_state(self):
        if self.actuator_idx - 1 < 0 or self.actuator_idx - 1 >= len(self.actuator_states):
            print("Warning: actuator_states empty, returning zeros")
            return np.zeros(9)
        return self.actuator_states[self.actuator_idx - 1]

    def get_imu_dt(self):
        return self.imu_data[self.imu_idx] - self.imu_data[self.imu_idx - 1]

    def save_imu_data(self, imu_data):
        self.imu_data.append(imu_data)
        self.data_length = len(self.imu_data)

    def save_actuator_data(self, actuator_data):
        self.actuator_data.append(actuator_data)

    def save_imu_state(self, x):
        self.imu_states.append(x)
        self.imu_positions.append(x[[0,1,2]])

    def save_actuator_state(self, x):
        self.actuator_states.append(x)
        self.thrust_positions.append(x[[0,1,2]])

    def print_imu_data(self):
        print(self.imu_data)

    def print_actuator_data(self):
        print(self.actuator_data)

    def print_actuator_states(self):
        print(self.actuator_states)

    def print_imu_states(self):
        print(self.imu_states)

    def plot_imu_positions(self):
        #print(self.positions)
        raw_positions = np.array(self.imu_positions)
        z_measurements = np.array(self.thrust_positions)

        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot raw GPS trajectory
        ax.plot(raw_positions[:, 0], raw_positions[:, 1], raw_positions[:, 2],
                'b.-', alpha=0.7, label='Raw GPS Trajectory', markersize=4, linewidth=1.5)

        # Plot z measurements trajectory (green/cyan)
        if z_measurements is not None and len(z_measurements) > 0:
            ax.plot(z_measurements[:, 0], z_measurements[:, 1], z_measurements[:, 2],
                    'g-', alpha=0.6, label='Thrust Model Measurements', markersize=2, linewidth=1.5)

        # Mark start point (green)
        ax.scatter(raw_positions[0, 0], raw_positions[0, 1], raw_positions[0, 2],
                   c='green', s=200, marker='o', label='Start Point', edgecolors='black', linewidths=2)

        # Mark end point (red)
        ax.scatter(raw_positions[-1, 0], raw_positions[-1, 1], raw_positions[-1, 2],
                   c='red', s=200, marker='s', label='End Point', edgecolors='black', linewidths=2)

        # Set labels and title
        ax.set_xlabel('X (meters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (meters)', fontsize=12, fontweight='bold')
        ax.set_title('Raw GPS Trajectory in 3D (No Filtering)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)

        # Set equal aspect ratio for better visualization
        max_range = np.array([
            raw_positions[:, 0].max() - raw_positions[:, 0].min(),
            raw_positions[:, 1].max() - raw_positions[:, 1].min(),
            raw_positions[:, 2].max() - raw_positions[:, 2].min()
        ]).max() / 2.0

        mid_x = (raw_positions[:, 0].max() + raw_positions[:, 0].min()) * 0.5
        mid_y = (raw_positions[:, 1].max() + raw_positions[:, 1].min()) * 0.5
        mid_z = (raw_positions[:, 2].max() + raw_positions[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add statistics text box
        stats_text = f"Total GPS Points: {len(raw_positions)}\n"
       # stats_text += f"Start: ({raw_positions[0, 0]:.2f}, {raw_positions[0, 1]:.2f}, {raw_positions[0, 2]:.2f})\n"
        #stats_text += f"End: ({raw_positions[-1, 0]:.2f}, {raw_positions[-1, 1]:.2f}, {raw_positions[-1, 2]:.2f})"

        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes,
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        return fig

    def plot_linear_accelerations(self):
        """Plot individual linear accelerations (x, y, z) over time in 2D."""
        if not self.imu_data:
            print("No IMU data to plot.")
            return

        # Extract timestamps and acceleration components
        timestamps = [d.timestamp for d in self.imu_data]
        ax_data = [d.a[0] for d in self.imu_data]
        ay_data = [d.a[1] for d in self.imu_data]
        az_data = [d.a[2] for d in self.imu_data]

        # Create 2D plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, ax_data, label='Accel X', color='r', linewidth=1.2)
        plt.plot(timestamps, ay_data, label='Accel Y', color='g', linewidth=1.2)
        plt.plot(timestamps, az_data, label='Accel Z', color='b', linewidth=1.2)

        # Labels and grid
        plt.title('Linear Accelerations Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Acceleration (m/s²)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show()

    def show(self):
        """Display the plot."""
        self.plot_imu_positions()
        plt.show()

