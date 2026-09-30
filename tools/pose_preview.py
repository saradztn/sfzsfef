"""Render a pose preview: source FBX skeleton vs converted IFP skeleton.

Pure-python PNG writer (no PIL). Rows: source (cyan) / converted CJ (orange).
Columns: sample times across the clip. Projection: game world X right, Z up.
"""
import math
import os
import struct
import sys
import zlib

sys.path.insert(0, '/home/user/sfzsfef/tools')
sys.path.insert(0, '/home/user/sfzsfef/ref')
from rt import (q_mul, q_conj, q_norm, q_from_mat, q_to_mat, mat_vec,
                load_skel)
from fbx_rig import load_rig
import fbx2ifp as conv


def write_png(path, width, height, rows):
    def chunk(tag, data):
        c = struct.pack('>I', len(data)) + tag + data
        c += struct.pack('>I', zlib.crc32(tag + data) & 0xffffffff)
        return c
    raw = b''.join(b'\x00' + bytes(r) for r in rows)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, 6))
    png += chunk(b'IEND', b'')
    open(path, 'wb').write(png)


class Canvas:
    def __init__(self, w, h, bg=(18, 18, 28)):
        self.w, self.h = w, h
        self.px = [bytearray(bg * w) for _ in range(h)]

    def dot(self, x, y, col, r=1):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r + 1:
                    continue
                xx, yy = int(x) + dx, int(y) + dy
                if 0 <= xx < self.w and 0 <= yy < self.h:
                    self.px[yy][xx * 3:xx * 3 + 3] = bytes(col)

    def line(self, x0, y0, x1, y1, col, thick=2):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for i in range(n + 1):
            t = i / n
            self.dot(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, col, thick)


def main():
    import json
    out_png = sys.argv[1] if len(sys.argv) > 1 else '/home/user/sfzsfef/preview.png'

    # ---- file tracks ----
    data = open(conv.OUT, 'rb').read()
    off = 36
    nbones, dsize, flags = struct.unpack_from('<III', data, off + 24)
    o = off + 36
    tracks = {}
    for b in range(nbones):
        ftype, nkf, bid = struct.unpack_from('<IIi', data, o + 24)
        fo = o + 36
        sz = 16 if ftype == 4 else 10
        kfs = []
        for k in range(nkf):
            qx, qy, qz, qw, t = struct.unpack_from('<5h', data, fo)
            px = py = pz = 0.0
            if ftype == 4:
                px, py, pz = [v / 1024 for v in struct.unpack_from('<3h', data, fo + 10)]
            kfs.append((q_norm((qx / 4096, qy / 4096, qz / 4096, qw / 4096)), (px, py, pz)))
            fo += sz
        tracks[bid] = kfs
        o += 36 + nkf * sz

    sk = load_skel(conv.SKEL)
    rig = load_rig(conv.FBX)
    by_name = {}
    for m in rig['models'].values():
        by_name.setdefault(m.name, m)

    # FBX skeleton edges (source, in game axes)
    src_edges = [
        ('Hips', 'Spine1'), ('Spine1', 'Spine2'),
        ('Spine2', 'Spine3'), ('Spine3', 'Spine4'), ('Spine4', 'Neck'),
        ('Neck', 'Head'), ('Head', 'HeadTip'),
        ('Spine4', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'),
        ('LeftArm', 'LeftForeArm'), ('LeftForeArm', 'LeftHand'),
        ('LeftHand', 'LeftFinger2Proximal'),
        ('Spine4', 'RightShoulder'), ('RightShoulder', 'RightArm'),
        ('RightArm', 'RightForeArm'), ('RightForeArm', 'RightHand'),
        ('RightHand', 'RightFinger2Proximal'),
        ('Hips', 'LeftThigh'), ('LeftThigh', 'LeftShin'), ('LeftShin', 'LeftFoot'),
        ('LeftFoot', 'LeftToe'), ('LeftToe', 'LeftToeTip'),
        ('Hips', 'RightThigh'), ('RightThigh', 'RightShin'), ('RightShin', 'RightFoot'),
        ('RightFoot', 'RightToe'), ('RightToe', 'RightToeTip'),
    ]
    # CJ edges (bone ids)
    cj_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (4, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36),
        (4, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26),
        (1, 41), (41, 42), (42, 43), (43, 44),
        (1, 51), (51, 52), (52, 53), (53, 54),
    ]

    order = []
    stack = [b for b in sk if sk[b]['parent'] == -1]
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in sk:
            if sk[c]['parent'] == b:
                stack.append(c)

    def cj_joints(k):
        local_q = {}
        root_T = None
        for bid, kfs in tracks.items():
            kk = k if len(kfs) > 1 else 0
            q, T = kfs[kk]
            local_q[bid] = q
            if bid == 0:
                root_T = T
        out = {}
        for b in order:
            p = sk[b]['parent']
            q = local_q[b]
            T = root_T if (b == 0) else sk[b]['T']
            if p == -1:
                out[b] = (q, T)
            else:
                pq, pp = out[p]
                v = mat_vec(q_to_mat(pq), T)
                out[b] = (q_mul(pq, q), (pp[0] + v[0], pp[1] + v[1], pp[2] + v[2]))
        return {b: out[b][1] for b in out}

    def src_joints(t):
        pts = {}
        for a, b in src_edges:
            for nm in (a, b):
                if nm not in pts:
                    W = by_name[nm].world_mat(t)
                    p = mat_vec(q_to_mat(conv.q_from_mat(conv.A_ROWS_YUP)), (W[0][3] * conv.SCALE,
                                                     W[1][3] * conv.SCALE,
                                                     W[2][3] * conv.SCALE))
                    # relative to hips at t=0
                    pts[nm] = p
        return pts

    # hips reference for centering the source
    W = by_name['Hips'].world_mat(0.0)
    hips0 = mat_vec(q_to_mat(conv.q_from_mat(conv.A_ROWS_YUP)), (W[0][3] * conv.SCALE,
                                         W[1][3] * conv.SCALE,
                                         W[2][3] * conv.SCALE))

    times = [0.0, 4.5, 9.6, 14.5, 19.5, 25.0]
    ncol = len(times)
    cell_w, cell_h = 240, 480
    Wpx = cell_w * ncol
    Hpx = cell_h * 2 + 40
    canvas = Canvas(Wpx, Hpx)
    SCL = 165.0

    def draw(frame_pts, edges, ci, ybase, col, hips_name):
        hp = frame_pts[hips_name]
        for a, b in edges:
            pa, pb = frame_pts[a], frame_pts[b]
            ha = 0.65 * (pa[0] - hp[0]) + 0.76 * (pa[1] - hp[1])
            hb = 0.65 * (pb[0] - hp[0]) + 0.76 * (pb[1] - hp[1])
            x0 = ha * SCL + ci * cell_w + cell_w / 2
            y0 = -(pa[2] - hp[2]) * SCL + ybase
            x1 = hb * SCL + ci * cell_w + cell_w / 2
            y1 = -(pb[2] - hp[2]) * SCL + ybase
            canvas.line(x0, y0, x1, y1, col, 2)

    for ci, t in enumerate(times):
        k = min(833, int(round(t * 30)))
        pts = src_joints(t)
        draw(pts, src_edges, ci, 330, (90, 210, 255), 'Hips')
        jp = cj_joints(k)
        draw(jp, cj_edges, ci, cell_h + 330, (255, 170, 60), 1)

    write_png(out_png, Wpx, Hpx, canvas.px)
    print('wrote', out_png, '%dx%d' % (Wpx, Hpx))


if __name__ == '__main__':
    main()
