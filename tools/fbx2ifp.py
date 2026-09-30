"""FBX -> GTA SA / MTA IFP (ANP3) converter (library + CLI).

Retarget pipeline (validated against vanilla ped.ifp semantics):
  1. Per source bone: world delta  dW(t) = W_src(t) * W_src_rest^-1  (game axes)
  2. Visual target                V(t)   = dW(t) * M(b)
       M(b) = min-rotation aligning CJ's bind bone aim to the source rest aim
       (pose-matches the source rest pose onto CJ, e.g. T-pose -> T-pose)
  3. Frame world                  FW(t)  = V(t) * skinBindWorld(b)   (RpSkin)
  4. Local quats                  q(b)   = conj(FW(parent)) * FW(b)
  5. Root translation (bone 0, type 4) = Hips world displacement (game axes)
  6. Static ped bones (jaw/brows/breasts/belly) use vanilla default locals.

Library entry points:
    convert(fbx_path, skel_path, skin_path, out_path, ...) -> stats dict
CLI:
    python3 tools/fbx2ifp.py [anim_name] [out_path]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ref'))
from rt import (q_mul, q_conj, q_norm, q_from_mat, q_to_mat, mat_vec,
                mat_norm_rows, load_skel)
from fbx_rig import load_rig
import inu_ifp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FBX = os.path.join(ROOT, '35609332-5782-49b7-a043-06e1afcac9d5.fbx')
SKEL = os.path.join(ROOT, 'ref/player_skel.json')
SKIN = os.path.join(ROOT, 'ref/player_skin.json')
OUT = os.path.join(ROOT, 'clip.ifp')

# FBX (Y-up, +Z face, +X left) -> game anim world (Z-up, +Y face, +X right)
A_ROWS_YUP = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
A_ROWS_ZUP = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]   # Blender-style Z-up, +Y face
SCALE = 0.01                          # FBX cm -> game units
FPS = 30.0

# Newton/Mixamo-style model -> (SA bone id, SA bone name, aim child model)
# The aim child is used for rest-pose matching; when it is not itself mapped
# (e.g. HeadTip/ToeTip) the in-direction (parent->joint) is used on both sides.
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

# Mixamo rig (mixamorig:* - prefix stripped by index_models). Mixamo has no
# bone above Hips: 'Root' is a virtual static node (root motion still comes
# from the Hips). Spine1+Spine2 fold into SA Spine1 (Spine2 drives).
BONE_MAP_MIXAMO = {
    'Root':       (0, 'Root',        'Hips'),
    'Hips':       (1, ' Pelvis',     'Spine'),
    'Spine':      (2, ' Spine',      'Spine1'),
    'Spine1':     (3, ' Spine1',     None),    # folded
    'Spine2':     (3, ' Spine1',     'Neck'),
    'Neck':       (4, ' Neck',       'Head'),
    'Head':       (5, ' Head',       'HeadTop_End'),
    'LeftShoulder':  (31, 'Bip01 L Clavicle', 'LeftArm'),
    'LeftArm':       (32, ' L UpperArm',      'LeftForeArm'),
    'LeftForeArm':   (33, ' L ForeArm',       'LeftHand'),
    'LeftHand':      (34, ' L Hand',          'LeftHandMiddle1'),
    'LeftHandIndex1': (35, ' L Finger',       'LeftHandIndex2'),
    'LeftHandIndex2': (36, 'L Finger01',      'LeftHandIndex3'),
    'RightShoulder': (21, 'Bip01 R Clavicle', 'RightArm'),
    'RightArm':      (22, ' R UpperArm',      'RightForeArm'),
    'RightForeArm':  (23, ' R ForeArm',       'RightHand'),
    'RightHand':     (24, ' R Hand',          'RightHandMiddle1'),
    'RightHandIndex1': (25, ' R Finger',      'RightHandIndex2'),
    'RightHandIndex2': (26, 'R Finger01',     'RightHandIndex3'),
    'LeftUpLeg':  (41, ' L Thigh',   'LeftLeg'),
    'LeftLeg':    (42, ' L Calf',    'LeftFoot'),
    'LeftFoot':   (43, ' L Foot',    'LeftToeBase'),
    'LeftToeBase': (44, ' L Toe0',   'LeftToe_End'),
    'RightUpLeg': (51, ' R Thigh',   'RightLeg'),
    'RightLeg':   (52, ' R Calf',    'RightFoot'),
    'RightFoot':  (53, ' R Foot',    'RightToeBase'),
    'RightToeBase': (54, ' R Toe0',  'RightToe_End'),
}

# (label, bone map, name of a virtual static root or None)
PRESETS = [('Newton/Rokoko', BONE_MAP, None),
           ('Mixamo', BONE_MAP_MIXAMO, 'Root')]
# bones that may be absent (they then keep the DFF bind pose)
OPTIONAL_IDS = {25, 26, 35, 36, 44, 54}


class StaticNode:
    """Virtual non-animated node at the origin (Mixamo has no Root bone)."""
    def __init__(self, name):
        self.name = name
        self.parent = None
        self.bind = None
        self.children = []

    def world_mat(self, t, eval_curves=True):
        return [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def resolve_rig(rig, bone_map=None):
    """Pick the bone-map preset that fits the FBX; returns dict with
    label, bone_map (pruned to present bones), by_name, missing (required)."""
    by_name = index_models(rig)
    cands = PRESETS if bone_map is None else [('custom', bone_map, None)]
    best = None
    for label, bm, vroot in cands:
        src_of = build_src_of(bm)
        req = sorted({nm for nm, v in bm.items()
                      if v[0] not in OPTIONAL_IDS and nm != vroot})
        found = [nm for nm in req if nm in by_name]
        score = len(found) / float(len(req))
        if best is None or score > best[0]:
            best = (score, label, bm, vroot, [nm for nm in req if nm not in by_name])
    score, label, bm, vroot, missing = best
    by_name = dict(by_name)
    if vroot and vroot not in by_name:
        by_name[vroot] = StaticNode(vroot)
    pruned = {}
    for nm, (bid, sa, aim) in bm.items():
        if nm in by_name:
            pruned[nm] = (bid, sa, aim if aim in by_name else None)
    return dict(label=label, bone_map=pruned, by_name=by_name,
                missing=missing, score=score)


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


def make_conj_a(a_q):
    def conj_A(q):
        return q_mul(a_q, q_mul(q, q_conj(a_q)))
    return conj_A


def build_src_of(bone_map):
    """One driver source bone per SA bone (fold entries: the one with an aim)."""
    src_of = {}
    for nm, (bid, _bn, aim_child) in bone_map.items():
        prev = src_of.get(bid)
        if prev is None or (bone_map[prev][2] is None and aim_child is not None):
            src_of[bid] = nm
    return src_of


# Root and Pelvis sit (almost) at the same point in SA peds, and the FBX Root
# lies on the floor, so their own links carry no direction. Both take the
# spine direction (Spine -> Spine1) as primary and the hip width
# (L Thigh - R Thigh) as roll reference.
PRIMARY_PAIRS = {0: (2, 3), 1: (2, 3)}
# roll references: bid -> (CJ bone a, CJ bone b, mapped bone A, mapped bone B)
# u = T_cj(a) - T_cj(b)  vs  v = src_rest(src_of[A]) - src_rest(src_of[B]).
# The FBX head has only HeadTip (no roll info): the CJ brow width is matched
# to the body's left-right axis (both rigs face forward in their rest pose).
# Symmetric left-right widths only (a single-side child such as one clavicle
# gives an asymmetric roll: 13.5 deg L/R disagreement measured on the neck).
ROLL_PAIRS = {0: (41, 51, 41, 51), 1: (41, 51, 41, 51), 2: (41, 51, 41, 51),
              3: (32, 22, 32, 22), 4: (32, 22, 32, 22), 5: (6, 7, 41, 51)}
# (clavicle joints coincide with the neck joint in player.dff -> upper arms)


def mapped_kids(bid, sk, src_of, bone_map):
    """Mapped children of bid; the chain continuation (the FBX aim child from
    the bone map) first, independent of the DFF frame order."""
    kids = [c for c in sk if sk[c]['parent'] == bid and c in src_of]
    aim = bone_map[src_of[bid]][2] if bid in src_of else None
    kids.sort(key=lambda c: (0 if src_of[c] == aim else 1, c))
    return kids


def index_models(rig):
    """name -> model, also under the name without a namespace prefix
    ('mixamorig:Hips' -> 'Hips', 'Armature|Hips' -> 'Hips')."""
    by_name = {}
    for m in rig['models'].values():
        by_name.setdefault(m.name, m)
    for m in rig['models'].values():
        short = m.name.replace('|', ':').split(':')[-1]
        by_name.setdefault(short, m)
    return by_name


def _m4_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)]
            for i in range(4)]


def _m4_inv(M):
    n = 4
    A = [list(M[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c:
                f = A[r][c]
                A[r] = [x - f * y for x, y in zip(A[r], A[c])]
    return [row[n:] for row in A]


def _pos3(M):
    return (M[0][3], M[1][3], M[2][3])


def _qrot(M):
    return q_from_world([M[0][:3], M[1][:3], M[2][:3]])


def _vang(a, b):
    a = v_norm(a); b = v_norm(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, v_dot(a, b)))))


def bind_consistent(m, tol=1.0):
    """True if the bind matrix uses the same bone frame as the animation:
    each child's offset seen in the bone's own frame must be identical in
    the bind and in the static pose (old exports, e.g. three.js Samba, have
    leg/arm bind frames flipped by 180 deg)."""
    B = getattr(m, 'bind', None)
    if B is None:
        return False
    S = m.world_mat(0.0, eval_curves=False)
    qB, qS = _qrot(B), _qrot(S)
    for c in getattr(m, 'children', []):
        # leaf end joints (HeadTip, ToeTip...) often carry junk bind matrices
        if getattr(c, 'bind', None) is None or not getattr(c, 'children', None):
            continue
        dB = v_sub(_pos3(c.bind), _pos3(B))
        dS = v_sub(_pos3(c.world_mat(0.0, eval_curves=False)), _pos3(S))
        if v_dot(dB, dB) < 1e-6 or v_dot(dS, dS) < 1e-6:
            continue
        if _vang(mat_vec(q_to_mat(q_conj(qB)), dB),
                 mat_vec(q_to_mat(q_conj(qS)), dS)) > tol:
            return False
    return True


def _mat_from(q, p):
    R = q_to_mat(q)
    return [list(R[0]) + [p[0]], list(R[1]) + [p[1]], list(R[2]) + [p[2]],
            [0.0, 0.0, 0.0, 1.0]]


def rest_world(m):
    """Source rest (bind) world matrix: the FBX BindPose / skin-cluster
    matrix when present (true T-pose), else the parent's bind composed with
    the static local offset, else the static Lcl pose. Mixamo stores an
    animation frame in Lcl (Samba: hips turned 35 deg), so the bind matters.
    A bind whose bone frame disagrees with the animation frame is not used
    as is: the static rotation is swung onto the bind child direction."""
    if getattr(m, 'bind', None) is not None:
        if bind_consistent(m):
            return m.bind
        S = m.world_mat(0.0, eval_curves=False)
        q = _qrot(S)
        pairs = []
        for c in getattr(m, 'children', []):
            if getattr(c, 'bind', None) is None or not getattr(c, 'children', None):
                continue
            dB = v_sub(_pos3(c.bind), _pos3(m.bind))
            dS = v_sub(_pos3(c.world_mat(0.0, eval_curves=False)), _pos3(S))
            if v_dot(dB, dB) > 1e-6 and v_dot(dS, dS) > 1e-6:
                pairs.append((dS, dB))
        if pairs:
            # rigid correction static -> bind: primary = longest child,
            # roll from the most perpendicular other child (hands: fingers)
            pairs.sort(key=lambda p: -v_dot(p[1], p[1]))
            dS0, dB0 = pairs[0]
            R = min_rot(v_norm(dS0), v_norm(dB0))
            sec = None
            best = 0.0
            for dS, dB in pairs[1:]:
                sn = v_norm(v_cross(v_norm(dB0), v_norm(dB)))
                w = math.sqrt(v_dot(sn, sn))
                if w > best:
                    best, sec = w, (dS, dB)
            if sec is not None and best > 0.1:
                R = twist_about(R, v_norm(dB0), sec[0], sec[1])
            q = q_mul(R, q)
        return _mat_from(q, _pos3(m.bind))
    p = m.parent
    while p is not None and getattr(p, 'bind', None) is None:
        p = p.parent
    W = m.world_mat(0.0, eval_curves=False)
    if p is None:
        return W
    Wp = p.world_mat(0.0, eval_curves=False)
    return _m4_mul(rest_world(p), _m4_mul(_m4_inv(Wp), W))


# Folded chains: SA Spine (2) covers several source bones up to the driver
# of SA Spine1 (3). Its per-frame direction is aimed at that source joint
# (the chord of the folded chain) so bends inside the fold are not lost;
# the twist still comes from the source bone.
FOLD_AIM = {2: 3}


def fold_fix(V, bid, t, skin_T, by_name, src_of, a_q):
    """Swing V so the CJ segment bid->FOLD_AIM[bid] points at the source."""
    c = FOLD_AIM.get(bid)
    if c is None or c not in src_of or bid not in src_of:
        return V
    u = v_sub(skin_T[c], skin_T[bid])
    Wb = by_name[src_of[bid]].world_mat(t)
    Wc = by_name[src_of[c]].world_mat(t)
    d = mat_vec(q_to_mat(a_q), v_sub(_pos3(Wc), _pos3(Wb)))
    if v_dot(u, u) < 1e-6 or v_dot(d, d) < 1e-6:
        return V
    cur = v_norm(mat_vec(q_to_mat(V), u))
    return q_norm(q_mul(min_rot(cur, v_norm(d)), V))


def twist_about(q, axis, u2, v2):
    """Rotate q about `axis` so the secondary pair (u2 -> v2) best fits."""
    axis = v_norm(axis)

    def proj(w):
        d = v_dot(w, axis)
        return (w[0] - axis[0] * d, w[1] - axis[1] * d, w[2] - axis[2] * d)

    a = proj(mat_vec(q_to_mat(q), u2))
    b = proj(v2)
    if v_dot(a, a) < 1e-8 or v_dot(b, b) < 1e-8:
        return q
    a = v_norm(a); b = v_norm(b)
    ang = math.atan2(v_dot(v_cross(a, b), axis), max(-1.0, min(1.0, v_dot(a, b))))
    s = math.sin(ang / 2.0)
    qt = (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2.0))
    return q_mul(qt, q)


def build_matchers(sk, skin_q, skin_T, by_name, src_of, bone_map, a_q):
    """M(b): rotation mapping the CJ bind frame onto the source rest frame.

    Primary pair (matched exactly): the first mapped child segment if the bone
    has one, else the in-segment (parent -> joint). The roll about the primary
    axis is resolved by a second mapped child segment when one exists,
    otherwise it is the canonical twist-free min-rotation (validated on the
    Neck->Head pair: 0.003 deg vs the source).
    """
    # source rest world (static bind) for all names we reference
    need = set(src_of.values())
    for nm in list(need):
        ac = bone_map[nm][2]
        if ac:
            need.add(ac)
    for c in sk:
        if c in src_of:
            ac = bone_map[src_of[c]][2]
            if ac:
                need.add(ac)
    src_rest = {}
    for nm in need:
        W = rest_world(by_name[nm])
        src_rest[nm] = (q_from_world([W[0][:3], W[1][:3], W[2][:3]]),
                        (W[0][3], W[1][3], W[2][3]))

    def seg(u, v):
        if v_dot(u, u) > 1e-4 and v_dot(v, v) > 1e-4:
            return v_norm(u), v_norm(v)
        return None

    M = {}
    order = []
    stack = [b for b in sk if sk[b]['parent'] == -1]
    while stack:
        b = stack.pop(0)
        order.append(b)
        stack.extend(c for c in sk if sk[c]['parent'] == b)
    for bid in [b for b in order if b in src_of]:
        nm = src_of[bid]
        q_rest, p_rest = src_rest[nm]
        def to_game(d):
            return mat_vec(q_to_mat(a_q), d)

        kids = mapped_kids(bid, sk, src_of, bone_map)
        primary = secondary = None
        if bid in PRIMARY_PAIRS and all(b in src_of for b in PRIMARY_PAIRS[bid]):
            a0, b0 = PRIMARY_PAIRS[bid]
            primary = seg(v_sub(skin_T[b0], skin_T[a0]),
                          to_game(v_sub(src_rest[src_of[b0]][1],
                                        src_rest[src_of[a0]][1])))
        if primary is None and kids:
            c = kids[0]
            primary = seg(v_sub(skin_T[c], skin_T[bid]),
                          to_game(v_sub(src_rest[src_of[c]][1], p_rest)))
            if primary is None:
                # degenerate link (CJ Pelvis->Spine is ~1 mm): use the child's
                # own first mapped segment as the direction reference
                gks = [g for g in sk if sk[g]['parent'] == c and g in src_of]
                if gks:
                    g = gks[0]
                    primary = seg(v_sub(skin_T[g], skin_T[c]),
                                  to_game(v_sub(src_rest[src_of[g]][1],
                                                src_rest[src_of[c]][1])))
        if primary is None:
            par_cj = sk[bid]['parent']
            par_src = by_name[nm].parent
            u = (v_sub(skin_T[bid], skin_T[par_cj])
                 if par_cj != -1 and bid in skin_T and par_cj in skin_T
                 else mat_vec(q_to_mat(skin_q[bid]), (1.0, 0.0, 0.0)))
            if par_src is not None:
                Wp = rest_world(par_src)
                v = to_game(v_sub(p_rest, (Wp[0][3], Wp[1][3], Wp[2][3])))
            else:
                v = mat_vec(q_to_mat(q_rest), (1.0, 0.0, 0.0))
            primary = seg(u, v)
        if primary is None:
            primary = (v_norm(mat_vec(q_to_mat(skin_q[bid]), (1.0, 0.0, 0.0))),
                       v_norm(mat_vec(q_to_mat(q_rest), (1.0, 0.0, 0.0))))
        rp = ROLL_PAIRS.get(bid)
        if (rp and rp[0] in skin_T and rp[1] in skin_T
                and rp[2] in src_of and rp[3] in src_of):
            secondary = seg(v_sub(skin_T[rp[0]], skin_T[rp[1]]),
                            to_game(v_sub(src_rest[src_of[rp[2]]][1],
                                          src_rest[src_of[rp[3]]][1])))
        if secondary is not None:
            qm = min_rot(primary[0], primary[1])
            qm = twist_about(qm, primary[1], secondary[0], secondary[1])
        else:
            # no roll reference: inherit the parent's correspondence and swing
            # minimally onto the primary pair (independent of skin-space axes)
            par = sk[bid]['parent']
            while par != -1 and par not in M:
                par = sk[par]['parent']
            if par in M:
                q0 = M[par]
                qm = q_mul(min_rot(mat_vec(q_to_mat(q0), primary[0]), primary[1]), q0)
            else:
                qm = min_rot(primary[0], primary[1])
        M[bid] = q_norm(qm)
    return M, src_rest


def convert(fbx_path=FBX, skel_path=SKEL, skin_path=SKIN, out_path=OUT,
            anim_name='clip', bone_map=None, fps=FPS, up_axis='Y',
            skel=None, skin=None, progress=None):
    """Convert one FBX clip to an ANP3 IFP. Returns a stats dict.

    skel/skin: optional pre-loaded dicts (from dff_skel/dff_skin) to avoid
    re-parsing the DFF. progress: callable(str) for status lines.
    """
    log = progress or (lambda s: None)

    if skel is None:
        sk = load_skel(skel_path)
    else:
        sk = skel
    if skin is None:
        skin = {int(k): v for k, v in json.load(open(skin_path)).items()}
    skin_q = {b: tuple(v['q']) for b, v in skin.items()}
    skin_T = {b: tuple(v['T']) for b, v in skin.items()}

    a_rows = A_ROWS_YUP if str(up_axis).upper() == 'Y' else A_ROWS_ZUP
    a_q = q_from_mat(a_rows)
    conj_A = make_conj_a(a_q)

    log('...reading FBX')
    rig = load_rig(fbx_path)
    rr = resolve_rig(rig, bone_map)
    by_name = rr['by_name']
    bone_map = rr['bone_map']
    log('rig: %s' % rr['label'])
    if rr['missing']:
        raise ValueError('FBX bones not found (%s rig): %s' %
                         (rr['label'], ', '.join(rr['missing'])))

    tmax = 0.0
    for c in rig['curves'].values():
        if c.times:
            tmax = max(tmax, c.times[-1])
    nsamples = int(round(tmax * fps)) + 1
    times = [k / fps for k in range(nsamples)]
    log('clip: 0 .. %.3f s, %d samples @ %g Hz' % (tmax, nsamples, fps))

    src_of = build_src_of(bone_map)
    missing = [nm for nm in src_of.values() if nm not in by_name]
    if missing:
        raise ValueError('FBX bones not found: %s' % ', '.join(sorted(set(missing))))

    log('...rest-pose matching')
    M, src_rest = build_matchers(sk, skin_q, skin_T, by_name,
                                 src_of, bone_map, a_q)

    log('...sampling motion')
    delta = {bid: [] for bid in src_of}
    for t in times:
        for bid, nm in src_of.items():
            W = by_name[nm].world_mat(t)
            q_t = q_from_world([W[0][:3], W[1][:3], W[2][:3]])
            delta[bid].append(conj_A(q_mul(q_t, q_conj(src_rest[nm][0]))))

    # root track translation from the pelvis driver's world displacement
    pelvis_src = src_of.get(1, 'Hips')
    hips = by_name[pelvis_src]
    W0 = hips.world_mat(0.0)
    p0 = (W0[0][3], W0[1][3], W0[2][3])
    root_T = []
    for t in times:
        W = hips.world_mat(t)
        d = v_sub((W[0][3], W[1][3], W[2][3]), p0)
        d_game = mat_vec(q_to_mat(a_q), d)
        root_T.append(tuple(c * SCALE for c in d_game))
    max_T = max(math.sqrt(sum(c * c for c in p)) for p in root_T)
    if max_T > 30.0:
        raise ValueError('root motion too large for IFP translation '
                         '(max %.1f units > 32)' % max_T)

    # skin q = bind WORLD rotation W (inverse of skinToBone); the mesh moves
    # by fw*skinToBone = fw*conj(W), so the visual V = fw*conj(W) -> fw = V*W
    fws = {}
    for bid in src_of:
        seq = []
        for k in range(nsamples):
            V = q_mul(delta[bid][k], M[bid])
            if bid in FOLD_AIM:
                V = fold_fix(V, bid, times[k], skin_T, by_name, src_of, a_q)
            seq.append(q_mul(V, skin_q[bid]))
        fws[bid] = seq

    # hierarchy walk -> locals
    order = []
    stack = [b for b in sk if sk[b]['parent'] == -1]
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in sk:
            if sk[c]['parent'] == b:
                stack.append(c)

    locals_by_bone = {b: [] for b in order}
    for k in range(nsamples):
        fw_chain = {}
        for b in order:
            p = sk[b]['parent']
            if b in fws:
                own = fws[b][k]
                q_loc = own if p == -1 else q_mul(q_conj(fw_chain[p]), own)
                fw_chain[b] = own
            else:
                q_loc = q_norm(STATIC_LOCALS.get(b, sk[b]['q']))
                fw_chain[b] = q_loc if p == -1 else q_mul(fw_chain[p], q_loc)
            locals_by_bone[b].append(q_loc)

    log('...writing IFP')
    ifp = inu_ifp.IFPFile()
    ifp.name = anim_name[:23]
    ifp.source_format = 'ANP3'
    anim = inu_ifp.Animation()
    anim.name = anim_name[:23]

    names = {}
    for nm, (bid, bn, _a) in bone_map.items():
        names.setdefault(bid, bn)
    names.update(STATIC_NAMES)

    for bid in sorted(locals_by_bone.keys()):
        bone = inu_ifp.AnimBone()
        bone.name = (names.get(bid, 'Bone%d' % bid))[:23]
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
    stats = {
        'out': out_path,
        'rig': rr['label'],
        'bytes': os.path.getsize(out_path),
        'tracks': len(anim.bones),
        'samples': nsamples,
        'duration': times[-1],
        'root_max': max_T,
    }
    log('wrote %s: %d tracks, %d samples, %d bytes' %
        (out_path, stats['tracks'], nsamples, stats['bytes']))
    return stats


def main():
    anim_name = sys.argv[1] if len(sys.argv) > 1 else 'clip'
    out_path = sys.argv[2] if len(sys.argv) > 2 else OUT
    convert(out_path=out_path, anim_name=anim_name, progress=print)


if __name__ == '__main__':
    main()
