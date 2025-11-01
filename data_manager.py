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

        self.positions = []

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
        self.positions.append(x[[0,1,2]])

    def save_actuator_state(self, x):
        self.actuator_states.append(x)

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
        raw_positions = np.array(self.positions)

        # Create 3D plot
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot raw GPS trajectory
        ax.plot(raw_positions[:, 0], raw_positions[:, 1], raw_positions[:, 2],
                'b.-', alpha=0.7, label='Raw GPS Trajectory', markersize=4, linewidth=1.5)

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

    def show(self):
        """Display the plot."""
        self.plot_imu_positions()
        plt.show()

