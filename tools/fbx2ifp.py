"""FBX -> GTA SA / MTA IFP (ANP3) converter.

Retarget pipeline (validated against vanilla ped.ifp semantics):
  1. Per source bone: world delta  dW(t) = W_src(t) * W_src_rest^-1  (game axes)
  2. Visual target                V(t)   = dW(t) * M(b)
       M(b) = min-rotation aligning CJ's bind bone aim to the source rest aim
       (pose-matches the source rest pose onto CJ, e.g. T-pose -> T-pose)
  3. Frame world                  FW(t)  = V(t) * skinBindWorld(b)   (RpSkin)
  4. Local quats                  q(b)   = conj(FW(parent)) * FW(b)
  5. Root translation (bone 0, type 4) = Hips world displacement (game axes)
  6. Static ped bones (jaw/brows/breasts/belly) use vanilla default locals.

Output: ANP3 via ref/inu_ifp.py.
"""
import json
import math
import os
import sys

sys.path.insert(0, '/home/user/sfzsfef/tools')
sys.path.insert(0, '/home/user/sfzsfef/ref')
from rt import (q_mul, q_conj, q_norm, q_from_mat, q_to_mat, mat_vec,
                mat_norm_rows, load_skel)
from fbx_rig import load_rig
import inu_ifp

ROOT = '/home/user/sfzsfef'
FBX = os.path.join(ROOT, '35609332-5782-49b7-a043-06e1afcac9d5.fbx')
SKEL = os.path.join(ROOT, 'ref/player_skel.json')
SKIN = os.path.join(ROOT, 'ref/player_skin.json')
OUT = os.path.join(ROOT, 'clip.ifp')

# FBX (Y-up, +Z face, +X left) -> game anim world (Z-up, +Y face, +X right)
A_ROWS = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
A_Q = q_from_mat(A_ROWS)
SCALE = 0.01                          # FBX cm -> game units
FPS = 30.0                            # sample rate (source keys ~30 Hz)

# Newton model -> (SA bone id, SA bone name, aim child model)
BONE_MAP = {
    'Root':       (0, 'Root',        'Hips'),
    'Hips':       (1, ' Pelvis',     'Spine1'),
    'Spine1':     (2, ' Spine',      'Spine2'),
    'Spine2':     (3, ' Spine1',     None),    # folded into SA Spine1
    'Spine3':     (3, ' Spine1',     None),    # folded
    'Spine4':     (3, ' Spine1',     'Neck'),  # last of fold wins
    'Neck':       (4, ' Neck',       'Head'),
    'Head':       (5, ' Head',       'HeadTip'),
    'LeftShoulder':  (31, 'Bip01 L Clavicle', 'LeftArm'),
    'LeftArm':       (32, ' L UpperArm',      'LeftForeArm'),
    'LeftForeArm':   (33, ' L ForeArm',       'LeftHand'),
    'LeftHand':      (34, ' L Hand',          'LeftFinger3Metacarpal'),
    'LeftFinger2Proximal': (35, ' L Finger',  'LeftFinger2Medial'),
    'LeftFinger2Medial':   (36, 'L Finger01', 'LeftFinger2Distal'),
    'RightShoulder': (21, 'Bip01 R Clavicle', 'RightArm'),
    'RightArm':      (22, ' R UpperArm',      'RightForeArm'),
    'RightForeArm':  (23, ' R ForeArm',       'RightHand'),
    'RightHand':     (24, ' R Hand',          'RightFinger3Metacarpal'),
    'RightFinger2Proximal': (25, ' R Finger', 'RightFinger2Medial'),
    'RightFinger2Medial':   (26, 'R Finger01','RightFinger2Distal'),
    'LeftThigh':  (41, ' L Thigh',   'LeftShin'),
    'LeftShin':   (42, ' L Calf',    'LeftFoot'),
    'LeftFoot':   (43, ' L Foot',    'LeftToe'),
    'LeftToe':    (44, ' L Toe0',    'LeftToeTip'),
    'RightThigh': (51, ' R Thigh',   'RightShin'),
    'RightShin':  (52, ' R Calf',    'RightFoot'),
    'RightFoot':  (53, ' R Foot',    'RightToe'),
    'RightToe':   (54, ' R Toe0',    'RightToeTip'),
}

# Vanilla single-keyframe locals for the static ped bones (identical across
# WALK_player / run_player / bomber / woman_idlestance).
STATIC_LOCALS = {
    6:   (0.045, -0.064, 0.577, 0.813),
    7:   (-0.035, 0.05, 0.578, 0.814),
    8:   (0.0, 0.0, 0.823, 0.568),
    201: (0.0, 0.0, 0.695, 0.719),
    301: (-0.051, -0.022, 0.67, 0.74),
    302: (0.111, 0.073, 0.645, 0.752),
}
STATIC_NAMES = {6: 'L Brow', 7: 'R Brow', 8: 'Jaw', 201: 'Belly',
                301: 'R breast', 302: 'L breast'}


def v_norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def min_rot(u, v):
    """Quaternion rotating unit vector u onto unit vector v."""
    u = v_norm(u); v = v_norm(v)
    d = max(-1.0, min(1.0, v_dot(u, v)))
    if d > 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    if d < -1.0 + 1e-9:
        axis = v_cross(u, (1.0, 0.0, 0.0))
        if v_dot(axis, axis) < 1e-6:
            axis = v_cross(u, (0.0, 1.0, 0.0))
        axis = v_norm(axis)
        return (axis[0], axis[1], axis[2], 0.0)
    axis = v_norm(v_cross(u, v))
    ang = math.acos(d)
    s = math.sin(ang / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2.0))


def q_from_world(m3):
    return q_from_mat(mat_norm_rows(m3))


def conj_A(q):
    """Rotate quaternion q from FBX world into game world: A q A^-1."""
    return q_mul(A_Q, q_mul(q, q_conj(A_Q)))


def main():
    anim_name = sys.argv[1] if len(sys.argv) > 1 else 'clip'
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUT

    sk = load_skel(SKEL)
    skin = {int(k): v for k, v in json.load(open(SKIN)).items()}
    skin_q = {b: tuple(v['q']) for b, v in skin.items()}
    skin_T = {b: tuple(v['T']) for b, v in skin.items()}

    rig = load_rig(FBX)
    by_name = {}
    for m in rig['models'].values():
        by_name.setdefault(m.name, m)

    # clip span from curve range
    tmax = -1e18
    for c in rig['curves'].values():
        if c.times:
            tmax = max(tmax, c.times[-1])
    nsamples = int(round(tmax * FPS)) + 1
    times = [k / FPS for k in range(nsamples)]
    print('clip: 0 .. %.4f s, %d samples @ %g Hz' % (tmax, nsamples, FPS))

    # ---- source rest world (static bind) ----
    src_rest = {}
    need = set(BONE_MAP)
    for nm, (_b, _bn, ac) in BONE_MAP.items():
        if ac:
            need.add(ac)
    for nm in need:
        W = by_name[nm].world_mat(0.0, eval_curves=False)
        src_rest[nm] = (q_from_world([W[0][:3], W[1][:3], W[2][:3]]),
                        (W[0][3], W[1][3], W[2][3]))

    # ---- CJ bind aims (skin bind joint positions) ----
    def cj_aim(bid):
        kids = [c for c in sk if sk[c]['parent'] == bid]
        if kids and bid in skin_T and kids[0] in skin_T:
            d = v_sub(skin_T[kids[0]], skin_T[bid])
            if v_dot(d, d) > 1e-8:
                return v_norm(d)
        return v_norm(mat_vec(q_to_mat(skin_q[bid]), (1.0, 0.0, 0.0)))

    # fold: one driver source bone per SA bone (the fold entry with an aim)
    src_of = {}
    for nm, (bid, _bn, aim_child) in BONE_MAP.items():
        prev = src_of.get(bid)
        if prev is None or (BONE_MAP[prev][2] is None and aim_child is not None):
            src_of[bid] = nm

    M = {}
    for bid, nm in src_of.items():
        q_rest, p_rest = src_rest[nm]
        aim_child = BONE_MAP[nm][2]
        if aim_child and aim_child in src_rest:
            d_src = v_sub(src_rest[aim_child][1], p_rest)
            if v_dot(d_src, d_src) < 1e-8:
                d_src = mat_vec(q_to_mat(q_rest), (1.0, 0.0, 0.0))
        else:
            d_src = mat_vec(q_to_mat(q_rest), (1.0, 0.0, 0.0))
        d_src_game = mat_vec(q_to_mat(A_Q), d_src)
        M[bid] = min_rot(cj_aim(bid), d_src_game)
    M[0] = M[1]

    # ---- sample world deltas ----
    delta = {bid: [] for bid in src_of}
    for t in times:
        for bid, nm in src_of.items():
            W = by_name[nm].world_mat(t)
            q_t = q_from_world([W[0][:3], W[1][:3], W[2][:3]])
            delta[bid].append(conj_A(q_mul(q_t, q_conj(src_rest[nm][0]))))

    # ---- root track translation from Hips world displacement ----
    hips = by_name['Hips']
    W0 = hips.world_mat(0.0)
    p0 = (W0[0][3], W0[1][3], W0[2][3])
    root_T = []
    for t in times:
        W = hips.world_mat(t)
        d = v_sub((W[0][3], W[1][3], W[2][3]), p0)
        d_game = mat_vec(q_to_mat(A_Q), d)
        root_T.append(tuple(c * SCALE for c in d_game))

    # ---- frame worlds for animated bones ----
    fws = {bid: [q_mul(q_mul(delta[bid][k], M[bid]), skin_q[bid])
                 for k in range(nsamples)]
           for bid in src_of}

    # ---- hierarchy walk -> locals ----
    order = []
    stack = [b for b in sk if sk[b]['parent'] == -1]
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in sk:
            if sk[c]['parent'] == b:
                stack.append(c)

    locals_by_bone = {}
    for b in order:
        locals_by_bone[b] = []
    for k in range(nsamples):
        fw_chain = {}
        for b in order:
            p = sk[b]['parent']
            if b in fws:
                own = fws[b][k]
                if p == -1:
                    q_loc = own
                else:
                    q_loc = q_mul(q_conj(fw_chain[p]), own)
                fw_chain[b] = own
            else:
                q_loc = q_norm(STATIC_LOCALS.get(b, sk[b]['q']))
                fw_chain[b] = q_loc if p == -1 else q_mul(fw_chain[p], q_loc)
            locals_by_bone[b].append(q_loc)

    # ---- build IFP ----
    ifp = inu_ifp.IFPFile()
    ifp.name = anim_name[:23]
    ifp.source_format = 'ANP3'
    anim = inu_ifp.Animation()
    anim.name = anim_name[:23]

    names = {}
    for nm, (bid, bn, _a) in BONE_MAP.items():
        names.setdefault(bid, bn)
    names.update(STATIC_NAMES)

    all_ids = sorted(locals_by_bone.keys())
    for bid in all_ids:
        bone = inu_ifp.AnimBone()
        bone.name = names[bid][:23]
        bone.bone_id = bid
        if bid in STATIC_LOCALS:
            bone.key_type = inu_ifp.HAS_ROT
            kf = inu_ifp.KeyFrame()
            kf.rotation = q_norm(STATIC_LOCALS[bid])
            kf.time = 0.0
            bone.keyframes = [kf]
        else:
            bone.key_type = (inu_ifp.HAS_ROT | inu_ifp.HAS_TRANS
                             if bid == 0 else inu_ifp.HAS_ROT)
            kfs = []
            prev = None
            for k in range(nsamples):
                q = q_norm(locals_by_bone[bid][k])
                if prev is not None and v_dot(q, prev) < 0:
                    q = tuple(-c for c in q)
                prev = q
                kf = inu_ifp.KeyFrame()
                kf.rotation = q
                kf.time = times[k]
                if bid == 0:
                    kf.translation = root_T[k]
                kfs.append(kf)
            bone.keyframes = kfs
        anim.bones.append(bone)

    ifp.animations.append(anim)
    inu_ifp.write_anp3(out_path, ifp)
    print('wrote %s: %d tracks, %d samples, %d bytes' %
          (out_path, len(anim.bones), nsamples, os.path.getsize(out_path)))
    return out_path


if __name__ == '__main__':
    main()
