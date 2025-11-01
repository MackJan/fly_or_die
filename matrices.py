import numpy as np

def evaluate_F(x, u, dt):
    px, py, pz, vx, vy, vz, qw, qx, qy, qz, bwx, bwy, bwz, bax, bay, baz = x.flatten()

    ax, ay, az = u.a
    wx, wy, wz = u.w

    F = np.array([
        [1, 0, 0, dt, 0, 0,
         dt**2*(1.0*qy*(az - baz) - 1.0*qz*(ay - bay)),
         dt**2*(1.0*qy*(ay - bay) + 1.0*qz*(az - baz)),
         dt**2*(1.0*qw*(az - baz) + 1.0*qx*(ay - bay) - 2.0*qy*(ax - bax)),
         dt**2*(-1.0*qw*(ay - bay) + 1.0*qx*(az - baz) - 2.0*qz*(ax - bax)),
         0, 0, 0,
         dt**2*(1.0*qy**2 + 1.0*qz**2 - 0.5),
         dt**2*(1.0*qw*qz - 1.0*qx*qy),
         dt**2*(-1.0*qw*qy - 1.0*qx*qz)],
        [0, 1, 0, 0, dt, 0,
         dt**2*(-1.0*qx*(az - baz) - 1.0*qz*(ax - bax)),
         dt**2*(-1.0*qw*(az - baz) - 2.0*qx*(ay - bay) + 1.0*qy*(ax - bax)),
         dt**2*(1.0*qx*(ax - bax) + 1.0*qz*(az - baz)),
         dt**2*(-1.0*qw*(ax - bax) + 1.0*qy*(az - baz) - 2.0*qz*(ay - bay)),
         0, 0, 0,
         dt**2*(1.0*qw*qz - 1.0*qx*qy),
         dt**2*(1.0*qx**2 + 1.0*qz**2 - 0.5),
         dt**2*(1.0*qw*qx - 1.0*qy*qz)],
        [0, 0, 1, 0, 0, dt,
         dt**2*(1.0*qx*(ay - bay) - 1.0*qy*(ax - bax)),
         dt**2*(1.0*qw*(ay - bay) - 2.0*qx*(az - baz) + 1.0*qz*(ax - bax)),
         dt**2*(-1.0*qw*(ax - bax) - 2.0*qy*(az - baz) + 1.0*qz*(ay - bay)),
         dt**2*(1.0*qx*(ax - bax) + 1.0*qy*(ay - bay)),
         0, 0, 0,
         dt**2*(1.0*qw*qy - 1.0*qx*qz),
         dt**2*(-1.0*qw*qx - 1.0*qy*qz),
         dt**2*(1.0*qx**2 + 1.0*qy**2 - 0.5)],
        [0, 0, 0, 1, 0, 0,
         dt*(2*qy*(az - baz) - 2*qz*(ay - bay)),
         dt*(2*qy*(ay - bay) + 2*qz*(az - baz)),
         dt*(2*qw*(az - baz) + 2*qx*(ay - bay) - 4*qy*(ax - bax)),
         dt*(-2*qw*(ay - bay) + 2*qx*(az - baz) - 4*qz*(ax - bax)),
         0, 0, 0,
         dt*(2*qy**2 + 2*qz**2 - 1),
         dt*(2*qw*qz - 2*qx*qy),
         dt*(-2*qw*qy - 2*qx*qz)],
        [0, 0, 0, 0, 1, 0,
         dt*(-2*qx*(az - baz) - 2*qz*(ax - bax)),
         dt*(-2*qw*(az - baz) - 4*qx*(ay - bay) + 2*qy*(ax - bax)),
         dt*(2*qx*(ax - bax) + 2*qz*(az - baz)),
         dt*(-2*qw*(ax - bax) + 2*qy*(az - baz) - 4*qz*(ay - bay)),
         0, 0, 0,
         dt*(2*qw*qz - 2*qx*qy),
         dt*(2*qx**2 + 2*qz**2 - 1),
         dt*(2*qw*qx - 2*qy*qz)],
        [0, 0, 0, 0, 0, 1,
         dt*(2*qx*(ay - bay) - 2*qy*(ax - bax)),
         dt*(2*qw*(ay - bay) - 4*qx*(az - baz) + 2*qz*(ax - bax)),
         dt*(-2*qw*(ax - bax) - 4*qy*(az - baz) + 2*qz*(ay - bay)),
         dt*(2*qx*(ax - bax) + 2*qy*(ay - bay)),
         0, 0, 0,
         dt*(2*qw*qy - 2*qx*qz),
         dt*(-2*qw*qx - 2*qy*qz),
         dt*(2*qx**2 + 2*qy**2 - 1)],
        [0, 0, 0, 0, 0, 0,
         -qw*(qw*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qx*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qy*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qz*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(qw**2 + qx**2 + qy**2 + qz**2),
         -qx*(qw*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qx*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qy*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qz*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) - (-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qy*(qw*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qx*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qy*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qz*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) - (-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qz*(qw*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qx*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qy*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qz*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) - (-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         qx*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         qy*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         qz*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         0, 0, 0],
        [0, 0, 0, 0, 0, 0,
         -qw*(qw*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qy*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + (-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qx*(qw*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qy*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(qw**2 + qx**2 + qy**2 + qz**2),
         -qy*(qw*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qy*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) - (-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qz*(qw*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qy*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + (-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qw*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qz*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         qy*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         0, 0, 0],
        [0, 0, 0, 0, 0, 0,
         -qw*(qw*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qz*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + (-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qx*(qw*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qz*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + (-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qy*(qw*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qz*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(qw**2 + qx**2 + qy**2 + qz**2),
         -qz*(qw*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qx*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2) - qz*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) - (-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         qz*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qw*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qx*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         0, 0, 0],
        [0, 0, 0, 0, 0, 0,
         -qw*(qw*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qx*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + (-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qx*(qw*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qx*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) - (-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qy*(qw*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qx*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + (-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qz*(qw*(-bwz + wz)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) - qx*(-bwy + wy)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qy*(-bwx + wx)*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2) + qz*np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2))/(qw**2 + qx**2 + qy**2 + qz**2)**(3/2) + np.cos(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/np.sqrt(qw**2 + qx**2 + qy**2 + qz**2),
         -qy*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         qx*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         -qw*np.sin(dt*np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)/2)/(np.sqrt(np.abs(bwx - wx)**2 + np.abs(bwy - wy)**2 + np.abs(bwz - wz)**2)*np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)),
         0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])

    return F

def evaluate_f(x, u, dt):
    px, py, pz, vx, vy, vz, qw, qx, qy, qz, bwx, bwy, bwz, bax, bay, baz = x.flatten()

    ax, ay, az = u.a
    wx, wy, wz = u.w
    return np.array([[dt ** 2 * (0.5 * (ax - bax) * (-2 * qy ** 2 - 2 * qz ** 2 + 1) + 0.5 * (ay - bay) * (
                -2 * qw * qz + 2 * qx * qy) + 0.5 * (az - baz) * (2 * qw * qy + 2 * qx * qz)) + dt * vx + px],
            [dt ** 2 * (0.5 * (ax - bax) * (-2 * qw * qz + 2 * qx * qy) + 0.5 * (ay - bay) * (
                        -2 * qx ** 2 - 2 * qz ** 2 + 1) + 0.5 * (az - baz) * (
                                    -2 * qw * qx + 2 * qy * qz)) + dt * vy + py], 
            [dt ** 2 * (0.5 * (ax - bax) * (-2 * qw * qy + 2 * qx * qz) + 0.5 * (ay - bay) * (
                        2 * qw * qx + 2 * qy * qz) + 0.5 * (az - baz) * (
                                -2 * qx ** 2 - 2 * qy ** 2 + 1) + 4.905) + dt * vz + pz], [dt * (
                    (ax - bax) * (-2 * qy ** 2 - 2 * qz ** 2 + 1) + (ay - bay) * (-2 * qw * qz + 2 * qx * qy) + (
                        az - baz) * (2 * qw * qy + 2 * qx * qz)) + vx], [dt * (
                    (ax - bax) * (-2 * qw * qz + 2 * qx * qy) + (ay - bay) * (-2 * qx ** 2 - 2 * qz ** 2 + 1) + (
                        az - baz) * (-2 * qw * qx + 2 * qy * qz)) + vy], [dt * (
                    (ax - bax) * (-2 * qw * qy + 2 * qx * qz) + (ay - bay) * (2 * qw * qx + 2 * qy * qz) + (
                        az - baz) * (-2 * qx ** 2 - 2 * qy ** 2 + 1) + 9.81) + vz], [(qw * np.cos(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) - qx * (-bwx + wx) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) - qy * (-bwy + wy) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) - qz * (-bwz + wz) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2)) / np.sqrt(
            qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)], [(qw * (-bwx + wx) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) + qx * np.cos(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) - qy * (-bwz + wz) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) + qz * (-bwy + wy) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2)) / np.sqrt(
            qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)], [(qw * (-bwy + wy) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) + qx * (-bwz + wz) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) + qy * np.cos(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) - qz * (-bwx + wx) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2)) / np.sqrt(
            qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)], [(qw * (-bwz + wz) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) - qx * (-bwy + wy) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) + qy * (-bwx + wx) * np.sin(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2) / np.sqrt(
            np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) + qz * np.cos(
            dt * np.sqrt(np.abs(bwx - wx) ** 2 + np.abs(bwy - wy) ** 2 + np.abs(bwz - wz) ** 2) / 2)) / np.sqrt(
            qw ** 2 + qx ** 2 + qy ** 2 + qz ** 2)], [bwx], [bwy], [bwz], [bax], [bay], [baz]])