from collections import namedtuple
import numpy as np

IMU_SAMPLES = 35
ACTUATOR_SAMPLES = 10

IMUData = namedtuple('IMUData', ['timestamp', 'a', 'w'])
ThrustData = namedtuple('data',['timestamp', 'c0', 'c1', 'c2', 'c3'])

def process_imu(reader):
    lin = []
    ang = []
    samples = []

    for _ in range(IMU_SAMPLES):
        s = next(reader)
        samples.append(s)
        lin.append(s.a)
        ang.append(s.w)

    lin = np.array(lin)
    ang = np.array(ang)

    # Compute averages
    a_avg = np.mean(lin, axis=0)
    w_avg = np.mean(ang, axis=0)

    # Take first timestamp
    timestamp = samples[0].timestamp

    # Return immutable namedtuple instance
    return IMUData(timestamp=timestamp, a=a_avg, w=w_avg)

def proces_actuator(reader):
        samples = []
        c0 = []
        c1 = []
        c2 = []
        c3 = []

        for _ in range(ACTUATOR_SAMPLES):
            s = next(reader)
            samples.append(s)
            c0.append(s.c0)
            c1.append(s.c1)
            c2.append(s.c2)
            c3.append(s.c3)

        c0 = np.array(c0)
        c1 = np.array(c1)
        c2 = np.array(c2)
        c3 = np.array(c3)


        # Compute averages
        c0_avg = np.mean(c0, axis=0)
        c1_avg = np.mean(c1, axis=0)
        c2_avg = np.mean(c2, axis=0)
        c3_avg = np.mean(c3, axis=0)

        # Take first timestamp
        timestamp = samples[0].timestamp

        # Return immutable namedtuple instance
        return ThrustData(timestamp=timestamp, c0=c0_avg, c1=c1_avg, c2=c2_avg, c3=c3_avg)