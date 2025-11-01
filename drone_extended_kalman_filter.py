import math
from filterpy.kalman import ExtendedKalmanFilter as EKF
import numpy as np
import sympy as sp
import scipy.linalg as linalg
import scipy as sc
import matrices

def Hx(x):
    return np.array([x[0], x[1], x[2], x[3], x[4], x[5]])

def H_f(x):
    H = np.zeros((6, 16))
    np.fill_diagonal(H, 1)
    return H

class DroneExtendedKalmanFilter(EKF):
    def __init__(self):
        EKF.__init__(self, dim_x=16, dim_z=6, dim_u=6)

        self.x = np.array([0. ,0. ,0., 0.01,0.01,0., 1., 0., 0., 0., 0.,0.,0., 0.01,-0.02,0.])

        #define symbolic equations
        px, py, pz, vx, vy, vz, qw, qx, qy, qz = sp.symbols(
            'px py pz vx vy vz qw qx qy qz')
        bwx, bwy, bwz, bax, bay, baz = sp.symbols('bwx bwy bwz bax bay baz')
        ax, ay, az = sp.symbols('ax ay az')
        wx, wy, wz = sp.symbols('wx wy wz')
        dt = sp.symbols('dt')

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

        R = sp.Matrix([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

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

        omega = wm - bw

        dq = self.delta_q_s(omega, dt)

        norm = sp.sqrt(qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)

        position_t = p + (v * dt + 0.5 * (R * (am - ba) + g) * dt ** 2)
        velocity_t = v + (R * (am - ba) + g) * dt
        quaternion_t = self.quat_multiply_s(dq, q)
        quaternion_t = quaternion_t / norm
        bw_t = bw
        ba_t = ba

        self.R = np.diag(6, 0.01)

        self.fxu = sp.Matrix.vstack(position_t, velocity_t, quaternion_t, bw_t, ba_t)
        print(self.fxu)

        # self.F_j = self.fxu.jacobian(sp.Matrix.vstack(x,u))
        self.F_j = self.fxu.jacobian(x)



        self.f_func = sp.lambdify((x, u, dt), self.fxu, 'numpy')



        self.F_j = self.F_j.doit()
        self.F_j = self.F_j.xreplace({d: 0 for d in self.F_j.atoms(sp.Derivative)})

        print(self.F_j)

        self.F_func = sp.lambdify((x, u, dt), self.F_j, 'numpy')

    def predict(self, u, dt):
        i = np.concatenate((u.a, u.w))

        if self.x_prior[6] == 0:
            self.x_prior[6] = 1

        x_pred = np.array(self.f_func(self.x_prior, i, dt), dtype=float).flatten()

        self.x = x_pred

        #self.x = matrices.evaluate_f(self.x_prior, u, dt)

        F = np.array(self.F_func(self.x, i, dt), dtype=float)
        #F = sc.differentiate.jacobian(self.fxu, [self.x_prior,i,dt])
        #F = matrices.evaluate_F(self.x_prior, u, dt)

        self.P = F @ self.P @ F.T + self.Q

        return self.x

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

    def update_f(self, z):
        #H = H_f(self.x)

        #print(H.shape)
        #print("rank(H) =", np.linalg.matrix_rank(H))
        #print("eig(H)  =", np.linalg.eigvalsh(H))

        #PHT = np.dot(self.P, H.T)

        #print(PHT.shape)
        #print("rank(PHT) =", np.linalg.matrix_rank(PHT))
        #print("eig(PHT)  =", np.linalg.eigvalsh(PHT))

        #self.S = np.dot(H, PHT) + self.R

        #print(self.S.shape)
        #print("rank(S) =", np.linalg.matrix_rank(self.S))
        #print("eig(S)  =", np.linalg.eigvalsh(self.S))

        #epsilon = 1e-6
        #self.S += np.eye(self.S.shape[0]) + epsilon

        #self.K = PHT.dot(linalg.inv(self.S))

        self.update(z, HJacobian=H_f, Hx=Hx)

    def get_x(self):
        return self.x


