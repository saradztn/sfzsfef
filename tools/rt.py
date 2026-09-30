"""Quaternion / matrix helpers + skeleton loading."""
import math, json, struct

# ---- quat (x,y,z,w) ----
def q_mul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz)

def q_conj(a):
    return (-a[0], -a[1], -a[2], a[3])

def q_norm(a):
    n = math.sqrt(sum(v*v for v in a)) or 1.0
    return (a[0]/n, a[1]/n, a[2]/n, a[3]/n)

def q_from_mat(rows):
    """rows = [[r00,r01,r02],[r10..],[r20..]] (row-major, row = image of axis)."""
    m00, m01, m02 = rows[0]; m10, m11, m12 = rows[1]; m20, m21, m22 = rows[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s; x = 0.25 * s
        y = (m01 + m10) / s; z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s; y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s; y = (m12 + m21) / s; z = 0.25 * s
    return q_norm((x, y, z, w))

def q_to_mat(q):
    x, y, z, w = q_norm(q)
    return [
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w),   1 - 2*(x*x + y*y)],
    ]

def mat_mul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def mat_vec(A, v):
    return tuple(sum(A[i][k]*v[k] for k in range(3)) for i in range(3))

def mat_t(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]

def mat_norm_rows(A):
    out = []
    for r in A:
        n = math.sqrt(sum(v*v for v in r)) or 1.0
        out.append([v/n for v in r])
    return out

# ---- skeleton from player.dff json ----
def load_skel(path):
    return skel_from_frames(json.load(open(path)))


def skel_from_frames(frames):
    sk = {}
    for f in frames:
        if f['bone'] is None:
            continue
        # RW stores rows = images of axes; column-vector M = transpose
        R = mat_norm_rows(mat_t(f['rot']))
        sk[f['bone']] = {
            'frame': f['idx'], 'parent_frame': f['parent'],
            'R': R, 'q': q_from_mat(R), 'T': tuple(f['pos']),
        }
    # link parents by frame
    byframe = {f['idx']: f for f in frames}
    for f in frames:
        if f['bone'] is None: continue
        pf = f['parent']
        pb = None
        while pf >= 0:
            if byframe[pf]['bone'] is not None:
                pb = byframe[pf]['bone']; break
            pf = byframe[pf]['parent']
        sk[f['bone']]['parent'] = pb
    for b, d in sk.items():
        if d['parent'] is None:
            d['parent'] = -1  # bone 0 root (parent is clump root)
    return sk

def skel_world(sk, local_q=None, root_T=None):
    """Compute world rotations & positions. local_q: {bone: quat} overrides."""
    out = {}
    order = sorted(sk.keys(), key=lambda b: 0 if sk[b]['parent'] == -1 else 1)
    # simple BFS from root
    order = []
    roots = [b for b in sk if sk[b]['parent'] == -1]
    stack = list(roots)
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in sk:
            if sk[c]['parent'] == b:
                stack.append(c)
    for b in order:
        d = sk[b]
        q = local_q.get(b, d['q']) if local_q else d['q']
        T = root_T if (root_T is not None and d['parent'] == -1) else d['T']
        if d['parent'] == -1:
            Wq = q; Wp = T
        else:
            pq, pp = out[d['parent']]
            Wq = q_mul(pq, q)
            v = mat_vec(q_to_mat(pq), T)
            Wp = (pp[0]+v[0], pp[1]+v[1], pp[2]+v[2])
        out[b] = (Wq, Wp)
    return out


def _axis_rot(axis, ang):
    c = math.cos(ang); s = math.sin(ang)
    if axis == 'x':
        return [[1, 0, 0], [0, c, -s], [0, s, c]]
    if axis == 'y':
        return [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return [[c, -s, 0], [s, c, 0], [0, 0, 1]]


def mat_from_euler(euler_deg, order='xyz'):
    """FBX-style Euler (degrees) -> column-vector matrix. order 'xyz': R = Rz*Ry*Rx."""
    x, y, z = [math.radians(v) for v in euler_deg]
    mats = {'x': _axis_rot('x', x), 'y': _axis_rot('y', y), 'z': _axis_rot('z', z)}
    M = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for ax in order:  # apply x first, then y, then z
        M = mat_mul(mats[ax], M)
    return M
