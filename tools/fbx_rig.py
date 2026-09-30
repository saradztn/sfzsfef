"""Extract rig hierarchy + bind + animation curves from the binary FBX.

Builds per-model local TRS evaluators (curves override Lcl values) and the
model tree, then can sample world matrices over the clip.
"""
import math
import struct
import sys

sys.path.insert(0, '/home/user/sfzsfef/tools')
from fbx_parse import parse_fbx
from rt import mat_t, mat_vec, mat_from_euler


def mat_mul(A, B):
    n = len(A)
    return [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

KTIME = 46186158000.0  # FBX time units per second


def dec(s):
    if isinstance(s, (bytes, bytearray)):
        return s.decode('utf-8', 'replace')
    return str(s)


def walk(node, out=None):
    if out is None:
        out = []
    out.append(node)
    for c in node.children:
        walk(c, out)
    return out


def collect(path):
    ver, fieldsz, roots = parse_fbx(path)
    nodes = []
    for r in roots:
        walk(r, nodes)
    return ver, nodes


def props70(node):
    """Properties70 dictionary: name -> list of values."""
    out = {}
    p = node.first('Properties70') or node.first('Properties60')
    if not p:
        return out
    for pr in p.children:
        if pr.props:
            key = dec(pr.props[0])
            out[key] = pr.props[1:]
    return out


def get_vec3(pv, default=(0.0, 0.0, 0.0)):
    if not pv:
        return default
    # typical: [b'Number', b'Number', b'Number', b'Number', flags..., x, y, z]
    nums = [v for v in pv if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) >= 3:
        return tuple(float(v) for v in nums[-3:])
    return default


def get_num(pv, default=0.0):
    if not pv:
        return default
    nums = [v for v in pv if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return float(nums[-1]) if nums else default


class Curve:
    """Evaluated FBX animation curve (times in seconds)."""

    def __init__(self, times, values):
        self.times = times
        self.values = values

    def eval(self, t):
        ts, vs = self.times, self.values
        if not ts:
            return 0.0
        if t <= ts[0]:
            return vs[0]
        if t >= ts[-1]:
            return vs[-1]
        lo, hi = 0, len(ts) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if ts[mid] <= t:
                lo = mid
            else:
                hi = mid
        span = ts[hi] - ts[lo]
        if span <= 0:
            return vs[hi]
        u = (t - ts[lo]) / span
        return vs[lo] * (1.0 - u) + vs[hi] * u


class RigModel:
    def __init__(self, mid, name, cls):
        self.id = mid
        self.name = name
        self.cls = cls
        self.parent = None
        self.children = []
        self.bind = None       # world bind matrix (BindPose / Cluster), rows
        self.lclT = (0.0, 0.0, 0.0)
        self.lclR = (0.0, 0.0, 0.0)
        self.lclS = (1.0, 1.0, 1.0)
        self.rotOff = (0.0, 0.0, 0.0)
        self.rotPiv = (0.0, 0.0, 0.0)
        self.sclOff = (0.0, 0.0, 0.0)
        self.sclPiv = (0.0, 0.0, 0.0)
        self.preRot = (0.0, 0.0, 0.0)
        self.postRot = (0.0, 0.0, 0.0)
        self.rotActive = False
        self.curveT = [None, None, None]
        self.curveR = [None, None, None]
        self.curveS = [None, None, None]

    def local_mat(self, t, eval_curves=True):
        T = list(self.lclT)
        R = list(self.lclR)
        S = list(self.lclS)
        if eval_curves:
            for i in range(3):
                if self.curveT[i] is not None:
                    T[i] = self.curveT[i].eval(t)
                if self.curveR[i] is not None:
                    R[i] = self.curveR[i].eval(t)
                if self.curveS[i] is not None:
                    S[i] = self.curveS[i].eval(t)
        pre = self.preRot if self.rotActive else (0.0, 0.0, 0.0)
        post = self.postRot if self.rotActive else (0.0, 0.0, 0.0)

        def trans(v):
            return [[1, 0, 0, v[0]], [0, 1, 0, v[1]], [0, 0, 1, v[2]], [0, 0, 0, 1]]

        def full(m3):
            return [list(m3[0]) + [0], list(m3[1]) + [0], list(m3[2]) + [0], [0, 0, 0, 1]]

        def scl(v):
            return [[v[0], 0, 0, 0], [0, v[1], 0, 0], [0, 0, v[2], 0], [0, 0, 0, 1]]

        def inv_full(m):
            # only used for pure translation or pure rotation here
            Rt = mat_t([m[0][:3], m[1][:3], m[2][:3]])
            nt = mat_vec(Rt, (m[0][3], m[1][3], m[2][3]))
            return [Rt[0] + [-nt[0]], Rt[1] + [-nt[1]], Rt[2] + [-nt[2]], [0, 0, 0, 1]]

        M = trans(T)
        M = mat_mul(M, trans(self.rotOff))
        M = mat_mul(M, trans(self.rotPiv))
        M = mat_mul(M, full(mat_from_euler(pre)))
        M = mat_mul(M, full(mat_from_euler(R)))
        M = mat_mul(M, inv_full(full(mat_from_euler(post))))
        M = mat_mul(M, inv_full(trans(self.rotPiv)))
        M = mat_mul(M, scl(S))
        M = mat_mul(M, inv_full(trans(self.sclPiv)))
        M = mat_mul(M, trans(self.sclOff))
        return M

    def world_mat(self, t, eval_curves=True):
        chain = []
        m = self
        while m is not None:
            chain.append(m)
            m = m.parent
        W = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        for m in reversed(chain):
            W = mat_mul(W, m.local_mat(t, eval_curves))
        return W


def load_rig(path):
    ver, nodes = collect(path)

    models = {}
    curve_nodes = {}
    curves = {}
    conns_oo = []   # (child_id, parent_id)
    conns_op = []   # (child_id, parent_id, prop)
    stacks = []

    for n in nodes:
        if n.name == 'Model' and n.props:
            mid = n.props[0]
            nm = dec(n.props[1]) if len(n.props) > 1 else dec(n.props[0])
            cls = dec(n.props[2]) if len(n.props) > 2 else ''
            nm = nm.split('\x00')[0].split('\x01')[-1]
            m = RigModel(mid, nm, cls)
            pv = props70(n)
            def pick(*keys):
                for k in keys:
                    if k in pv:
                        return pv[k]
                return None
            m.lclT = get_vec3(pick('Lcl Translation', 'LclTranslation'))
            m.lclR = get_vec3(pick('Lcl Rotation', 'LclRotation'))
            m.lclS = get_vec3(pick('Lcl Scaling', 'LclScaling'), (1, 1, 1))
            m.rotOff = get_vec3(pick('Rotation Offset', 'RotationOffset'))
            m.rotPiv = get_vec3(pick('Rotation Pivot', 'RotationPivot'))
            m.sclOff = get_vec3(pick('Scaling Offset', 'ScalingOffset'))
            m.sclPiv = get_vec3(pick('Scaling Pivot', 'ScalingPivot'))
            m.preRot = get_vec3(pick('PreRotation'))
            m.postRot = get_vec3(pick('PostRotation'))
            m.rotActive = bool(pv.get('RotationActive', [False])[-1]) if pv.get('RotationActive') else False
            models[mid] = m
        elif n.name == 'AnimationCurveNode' and n.props:
            cnid = n.props[0]
            pv = props70(n)
            vals = [get_num(pv.get('d|X')), get_num(pv.get('d|Y')), get_num(pv.get('d|Z'))]
            curve_nodes[cnid] = {'values': vals, 'owner': None, 'target': None}
        elif n.name == 'AnimationCurve' and n.props:
            cid = n.props[0]
            kv = None
            kt = None
            for c in n.children:
                if c.name == 'KeyValueFloat':
                    kv = c.props[0] if c.props and isinstance(c.props[0], list) else c.props
                elif c.name == 'KeyValueDouble':
                    kv = c.props[0] if c.props and isinstance(c.props[0], list) else c.props
                elif c.name == 'KeyTime':
                    kt = c.props[0] if c.props and isinstance(c.props[0], list) else c.props
            if kt is not None and kv is not None:
                times = [tt / KTIME for tt in kt]
                curves[cid] = Curve(times, [float(v) for v in kv])
        elif n.name == 'AnimationStack' and n.props:
            stacks.append(n.props[0])
        elif n.name == 'Connections':
            for c in n.children:
                if not c.props:
                    continue
                kind = dec(c.props[0])
                if kind == 'OO':
                    conns_oo.append((c.props[1], c.props[2]))
                elif kind == 'OP':
                    conns_op.append((c.props[1], c.props[2], dec(c.props[3])))

    # parent links among models
    for child_id, parent_id in conns_oo:
        if child_id in models and parent_id in models:
            models[child_id].parent = models[parent_id]
            models[parent_id].children.append(models[child_id])

    # curve node -> model target (OP with prop LclTranslation/LclRotation/LclScaling)
    for cid, pid, prop in conns_op:
        if cid in curve_nodes and pid in models:
            curve_nodes[cid]['owner'] = pid
            curve_nodes[cid]['target'] = prop
    # curves -> curve nodes on d|X/Y/Z
    for cid, pid, prop in conns_op:
        if cid in curves and pid in curve_nodes:
            slot = {'d|X': 0, 'd|Y': 1, 'd|Z': 2}.get(prop)
            if slot is not None:
                m = models.get(curve_nodes[pid]['owner'])
                tgt = curve_nodes[pid]['target']
                if m is not None:
                    arr = {'LclTranslation': m.curveT, 'LclRotation': m.curveR,
                           'LclScaling': m.curveS,
                           'Lcl Translation': m.curveT, 'Lcl Rotation': m.curveR,
                           'Lcl Scaling': m.curveS}.get(tgt)
                    if arr is not None:
                        arr[slot] = curves[cid]
                        # curve overrides the static Lcl value at that slot:
                        # keep static value = curve value at its first key (bind)
                        if curves[cid].times:
                            v0 = curves[cid].values[0]
                            cur = list(arr is m.curveT and m.lclT or arr is m.curveR and m.lclR or m.lclS)
                            cur[slot] = v0
                            if arr is m.curveT:
                                m.lclT = tuple(cur)
                            elif arr is m.curveR:
                                m.lclR = tuple(cur)
                            else:
                                m.lclS = tuple(cur)

    # clip span from the first AnimationStack
    span = None
    for n in nodes:
        if n.name == 'AnimationStack' and n.props:
            pv = props70(n)
            st = get_num(pv.get('LocalStart'), 0)
            sp = get_num(pv.get('LocalStop'), 0)
            if sp > st:
                span = (st / KTIME, sp / KTIME)
            break

    gs = {}
    for n in nodes:
        if n.name == 'GlobalSettings':
            gs = props70(n)
            break

    # true bind pose (T-pose): BindPose PoseNode matrices, else skin-cluster
    # TransformLink. FBX stores 16 doubles row-vector style (axes in rows,
    # translation in 12..14) -> transpose to column-vector rows.
    def _mat(vals):
        v = list(vals)
        return [[v[0], v[4], v[8], v[12]], [v[1], v[5], v[9], v[13]],
                [v[2], v[6], v[10], v[14]], [0.0, 0.0, 0.0, 1.0]]
    bind = {}
    for n in nodes:
        if n.name == 'Pose' and len(n.props) > 2 and dec(n.props[2]) == 'BindPose':
            for pn in n.children:
                if pn.name != 'PoseNode':
                    continue
                nid = mat = None
                for c in pn.children:
                    if c.name == 'Node' and c.props:
                        nid = c.props[0]
                    elif c.name == 'Matrix' and c.props and len(c.props[0]) == 16:
                        mat = c.props[0]
                if nid in models and mat is not None:
                    bind.setdefault(nid, _mat(mat))
    cluster_link = {}
    for n in nodes:
        if n.name == 'Deformer' and len(n.props) > 2 and dec(n.props[2]) == 'Cluster':
            tl = [c for c in n.children if c.name == 'TransformLink' and c.props]
            if tl and len(tl[0].props[0]) == 16:
                cluster_link[n.props[0]] = _mat(tl[0].props[0])
    for child, par in conns_oo:
        if child in cluster_link and par in models:
            bind.setdefault(par, cluster_link[child])
        elif par in cluster_link and child in models:
            bind.setdefault(child, cluster_link[par])
    for mid, M in bind.items():
        models[mid].bind = M

    return {
        'models': models, 'curve_nodes': curve_nodes, 'curves': curves,
        'conns_oo': conns_oo, 'span': span, 'gs': gs,
    }


def roots_of(models):
    return [m for m in models.values() if m.parent is None]


if __name__ == '__main__':
    rig = load_rig(sys.argv[1] if len(sys.argv) > 1 else
                   '/home/user/sfzsfef/35609332-5782-49b7-a043-06e1afcac9d5.fbx')
    models = rig['models']
    print('models:', len(models))
    print('curves:', len(rig['curves']), 'curve nodes:', len(rig['curve_nodes']))
    print('clip span (s):', rig['span'])
    gs = rig['gs']
    for k in ['UpAxis', 'UpAxisSign', 'FrontAxis', 'FrontAxisSign', 'CoordAxis', 'CoordAxisSign', 'UnitScaleFactor']:
        if k in gs:
            print(' ', k, gs[k][-1] if isinstance(gs[k], list) else gs[k])
    print('roots:')
    for r in roots_of(models):
        print('  ', r.id, repr(r.name), r.cls)
    # print hierarchy of one root
    def dump(m, d=0):
        anim = []
        for arr, nm in ((m.curveT, 'T'), (m.curveR, 'R'), (m.curveS, 'S')):
            if any(arr):
                anim.append(nm)
        print('  ' * d + '%-14s T=%s R=%s S=%s %s' % (
            m.name, tuple(round(v, 3) for v in m.lclT),
            tuple(round(v, 2) for v in m.lclR), tuple(round(v, 3) for v in m.lclS),
            ('anim:' + ','.join(anim)) if anim else ''))
        for c in m.children:
            dump(c, d + 1)

    for r in roots_of(models):
        dump(r)
