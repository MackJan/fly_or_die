import numpy as np

RPM_MAX = 15000
b = 0.000022
m = 1.075

def quat_to_mat(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
    ])

def calculate_state(motor_data, q, prev_state):
    dt, c0, c1, c2, c3 = motor_data
    a_old = prev_state[0:3]
    v_old = prev_state[3:6]
    p_old = prev_state[6:9]

    omega_max = RPM_MAX * (2. * np.pi / 60.)

    # angular velocities
    w0 = c0 * omega_max
    w1 = c1 * omega_max
    w2 = c2 * omega_max
    w3 = c3 * omega_max

    # thrusts
    t0 = w0 * b
    t1 = w1 * b
    t2 = w2 * b
    t3 = w3 * b

    # total thrust
    t = t0 + t1 + t2 + t3
    T = np.array([[0.], [0.], [t]])

    R = quat_to_mat(q)

    print(R)
    print(T)

    F = R @ T

    a_new = F / m

    v_new = v_old + (a_old + a_new) / 2 * dt

    p_new = p_old + v_old * dt + 1/2 * a_old * dt**2

    return np.concatenate((p_new, v_new, a_new))