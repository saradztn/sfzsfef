"""Extract RpSkin inverse-bind matrices from player.dff.

The RpSkin PLG (0x116) tail holds 32 bone matrices (64 B each: 3 rows of
(x,y,z,pad) + pos(x,y,z,pad)) in HAnim hierarchy order. These are the
"skin to bone" matrices; their inverses are the true mesh bind pose of
each bone (frame list rotations are NOT authoritative for the mesh).

Output: {bone_id: {R: 3x3 rows, T: xyz}} = world bind of each bone.
"""
import json
import struct
import sys

sys.path.insert(0, '/home/user/sfzsfef/tools')
from rt import mat_t, q_from_mat

HIER = [0, 1, 2, 3, 4, 5, 8, 6, 7, 31, 32, 33, 34, 35, 36,
        21, 22, 23, 24, 25, 26, 302, 301, 201, 41, 42, 43, 44, 51, 52, 53, 54]

# Verified by orthonormality scan of the PLG tail (total err 0.015).
MATRIX_BLOCK_START = 1379591


def mat4_inv(m):
    R = [m[0][:3], m[1][:3], m[2][:3]]
    Rt = mat_t(R)
    t = (m[0][3], m[1][3], m[2][3])
    nt = [sum(Rt[i][k] * t[k] for k in range(3)) for i in range(3)]
    return [Rt[0] + [-nt[0]], Rt[1] + [-nt[1]], Rt[2] + [-nt[2]], [0, 0, 0, 1]]


def extract(dff_path):
    data = open(dff_path, 'rb').read()
    out = {}
    for i, bid in enumerate(HIER):
        v = struct.unpack_from('<16f', data, MATRIX_BLOCK_START + i * 64)
        rows = [list(v[0:3]), list(v[4:7]), list(v[8:11])]  # skinToBone rows
        pos = [v[12], v[13], v[14]]
        M = [mat_t(rows)[0] + [pos[0]], mat_t(rows)[1] + [pos[1]],
             mat_t(rows)[2] + [pos[2]], [0, 0, 0, 1]]
        mi = mat4_inv(M)  # bone world bind
        R = [mi[0][:3], mi[1][:3], mi[2][:3]]
        q = q_from_mat(R)
        out[str(bid)] = {
            'R': R,
            'T': [mi[0][3], mi[1][3], mi[2][3]],
            'q': [q[0], q[1], q[2], q[3]],
        }
    return out


if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else '/home/user/sfzsfef/ref/player.dff'
    dst = sys.argv[2] if len(sys.argv) > 2 else '/home/user/sfzsfef/ref/player_skin.json'
    skins = extract(src)
    json.dump(skins, open(dst, 'w'), indent=1)
    print('wrote', dst, 'bones:', len(skins))
