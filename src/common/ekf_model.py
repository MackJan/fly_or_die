"""EKF model: build symbolic dynamics and expose fast NumPy wrappers.

This module mirrors the symbolic expressions from the IMU notebook but
provides numeric lambdified functions that `main.py` can import and call.

The lambdified functions accept explicit scalar arguments in the order
of the symbols: x (16 scalars), u (6 scalars), dt (1 scalar).
To simplify runtime usage, this module exposes wrapper functions that
take numpy arrays: fx_numpy(x_vec, u_vec, dt) and Jx_numpy(x_vec, u_vec, dt).
"""
from typing import Sequence
import numpy as np
import sympy as sp

# Build symbols
px, py, pz, vx, vy, vz, qw, qx, qy, qz = sp.symbols(
    'p_{x} p_{y} p_{z} v_{x} v_{y} v_{z} q_{w} q_{x} q_{y} q_{z}')
bwx, bwy, bwz, bax, bay, baz = sp.symbols('b_{wx} b_{wy} b_{wz} b_{ax} b_{ay} b_{az}')

x_syms = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bwx, bwy, bwz, bax, bay, baz]

ax, ay, az, wx, wy, wz = sp.symbols('a_{x} a_{y} a_{z} w_{x} w_{y} w_{z}')
u_syms = [ax, ay, az, wx, wy, wz]

dt = sp.symbols('dt')

# Helper: rotation matrix from quaternion (sympy)
R = sp.Matrix([
    [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
    [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
    [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
])

# State vectors (sympy)
px_s, py_s, pz_s = px, py, pz
vx_s, vy_s, vz_s = vx, vy, vz
q_s = sp.Matrix([qw, qx, qy, qz])
bw_s = sp.Matrix([bwx, bwy, bwz])
ba_s = sp.Matrix([bax, bay, baz])

am = sp.Matrix([ax, ay, az])
wm = sp.Matrix([wx, wy, wz])

# gravity
g = sp.Matrix([0, 0, -9.81])

# corrected angular velocity
omega = wm - bw_s

# delta quaternion (small-angle exact formula)
theta = sp.sqrt(omega.dot(omega)) * dt
# safe axis (avoid exact division by zero by adding a tiny numeric regularizer)
# using a small numeric constant here keeps the expression algebraic and lambdifiable
eps = sp.N(1e-12)
axis = omega / (sp.sqrt(omega.dot(omega)) + eps)

# delta quaternion
w = sp.cos(theta / 2)
xyz = axis * sp.sin(theta / 2)
dq = sp.Matrix([w, xyz[0], xyz[1], xyz[2]])

# quaternion multiplication (dq * q)
qw1, qx1, qy1, qz1 = dq[0], dq[1], dq[2], dq[3]
qw2, qx2, qy2, qz2 = qw, qx, qy, qz
qpw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
qpx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
qpy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
qpz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2
q_new = sp.Matrix([qpw, qpx, qpy, qpz])

# normalize new quaternion explicitly (normalize q_new rather than the old q)
norm_q_new = sp.sqrt(q_new.dot(q_new))
q_new = q_new / (norm_q_new + sp.N(1e-12))

# kinematic updates
position_t = sp.Matrix([px_s, py_s, pz_s]) + (sp.Matrix([vx_s, vy_s, vz_s]) * dt +
                                               sp.Rational(1, 2) * (R * (am - ba_s) + g) * dt ** 2)
velocity_t = sp.Matrix([vx_s, vy_s, vz_s]) + (R * (am - ba_s) + g) * dt

fx_sym = sp.Matrix.vstack(position_t, velocity_t, q_new, bw_s, ba_s)

# Jacobians
Jx_sym = fx_sym.jacobian(x_syms)
Ju_sym = fx_sym.jacobian(u_syms)

# Lambdify the functions to numpy
fx_lamb = sp.lambdify([*x_syms, *u_syms, dt], fx_sym, modules=['numpy'])
Jx_lamb = sp.lambdify([*x_syms, *u_syms, dt], Jx_sym, modules=['numpy'])
Ju_lamb = sp.lambdify([*x_syms, *u_syms, dt], Ju_sym, modules=['numpy'])


def _to_args(x: Sequence[float], u: Sequence[float], dt_val: float):
    """Flatten inputs into the argument list for the lambdified functions."""
    x_list = list(np.asarray(x).ravel())
    u_list = list(np.asarray(u).ravel())
    return [*x_list, *u_list, float(dt_val)]


def fx_numpy(x: np.ndarray, u: np.ndarray, dt_val: float) -> np.ndarray:
    """Return the predicted state vector (length 16) as a numpy array."""
    args = _to_args(x, u, dt_val)
    out = fx_lamb(*args)
    return np.asarray(out, dtype=float).ravel()


def Jx_numpy(x: np.ndarray, u: np.ndarray, dt_val: float) -> np.ndarray:
    """Return the state Jacobian (16x16) as a numpy array."""
    args = _to_args(x, u, dt_val)
    out = Jx_lamb(*args)
    return np.asarray(out, dtype=float)


def Ju_numpy(x: np.ndarray, u: np.ndarray, dt_val: float) -> np.ndarray:
    """Return the input Jacobian (16x6) as a numpy array."""
    args = _to_args(x, u, dt_val)
    out = Ju_lamb(*args)
    return np.asarray(out, dtype=float)
