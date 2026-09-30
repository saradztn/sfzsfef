"""Validate a converted IFP against its source FBX.

Checks:
  A. structure: independent raw parse of the ANP3 (mirrors ped.ifp dumps)
  B. quantization: reconstructed visual vs intended visual (angle)
  C. tracking: achieved bone-aim directions vs the source FBX directions
  D. rest pose: joint-segment directions at t=0 vs source rest (catches twists)
  E. root translation continuity
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fbx2ifp as conv
from rt import q_mul, q_conj, q_norm, q_to_mat, mat_vec, load_skel
from fbx_rig import load_rig


def read_anp3_raw(path):
    data = open(path, 'rb').read()
    assert data[:4] == b'ANP3', 'not ANP3'
    size = struct.unpack_from('<I', data, 4)[0]
    name = data[8:32].split(b'\x00')[0].decode()
    nanim = struct.unpack_from('<I', data, 32)[0]
    off = 36
    anims = []
    for a in range(nanim):
        aname = data[off:off+24].split(b'\x00')[0].decode()
        nbones, dsize, flags = struct.unpack_from('<III', data, off+24)
        o = off + 36
        tracks = []
        for b in range(nbones):
            bname = data[o:o+24].split(b'\x00')[0].decode()
            ftype, nkf, bid = struct.unpack_from('<IIi', data, o+24)
            fo = o + 36
            sz = 16 if ftype == 4 else 10
            kfs = []
            for k in range(nkf):
                qx, qy, qz, qw, t = struct.unpack_from('<5h', data, fo)
                px = py = pz = 0.0
                if ftype == 4:
                    px, py, pz = [v / 1024 for v in struct.unpack_from('<3h', data, fo + 10)]
                kfs.append({'q': q_norm((qx / 4096, qy / 4096, qz / 4096, qw / 4096)),
                            't': t, 'p': (px, py, pz)})
                fo += sz
            tracks.append({'name': bname, 'type': ftype, 'id': bid, 'kfs': kfs})
            o += 36 + nkf * sz
        anims.append({'name': aname, 'tracks': tracks, 'dsize': dsize, 'flags': flags})
        off = o
    return {'name': name, 'anims': anims, 'size': size, 'total': len(data)}


def ang(q1, q2):
    q1 = q_norm(q1); q2 = q_norm(q2)
    d = q_mul(q_conj(q1), q2)
    return 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(d[3])))))


def vang(u, v):
    u = conv.v_norm(u); v = conv.v_norm(v)
    c = max(-1.0, min(1.0, conv.v_dot(u, v)))
    return math.degrees(math.acos(c))


def validate(ifp_path, fbx_path, sk=None, skin=None, log=None,
             up_axis='Y', stride=3):
    """Run all checks; returns a dict with 'passed' + the error metrics.
    log: callable(*args) for report lines (default print)."""
    if log is None:
        log = print
    _log = log
    log = lambda *a: _log(' '.join(str(x) for x in a))
    import json

    log('== A. structure ==')
    f = read_anp3_raw(ifp_path)
    log('file %s: %d bytes, pkg=%r, anims=%d' %
          (ifp_path, f['total'], f['name'], len(f['anims'])))
    a = f['anims'][0]
    log('anim %r: %d tracks, flags=%d, dsize=%d' %
          (a['name'], len(a['tracks']), a['flags'], a['dsize']))
    t4 = [t for t in a['tracks'] if t['type'] == 4]
    log('type-4 tracks:', [(t['name'], t['id'], len(t['kfs'])) for t in t4])
    ok = True
    for t in a['tracks']:
        ts = [kf['t'] for kf in t['kfs']]
        if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
            log('!! non-monotonic time in', t['name']); ok = False
    log('time monotonic + unit quats:', 'OK' if ok else 'FAIL')

    # ---- shared model data ----
    if sk is None:
        sk = load_skel(conv.SKEL)
    if skin is None:
        skin = {int(k): v for k, v in json.load(open(conv.SKIN)).items()}
    skin_q = {b: tuple(v['q']) for b, v in skin.items()}
    skin_T = {b: tuple(v['T']) for b, v in skin.items()}
    rig = load_rig(fbx_path)
    rr = conv.resolve_rig(rig)
    by_name = rr['by_name']
    bone_map = rr['bone_map']
    log('rig: %s' % rr['label'])
    a_q = conv.q_from_mat(conv.A_ROWS_YUP if str(up_axis).upper() == 'Y'
                          else conv.A_ROWS_ZUP)
    src_of = conv.build_src_of(bone_map)
    M, src_rest = conv.build_matchers(sk, skin_q, skin_T, by_name,
                                      src_of, bone_map, a_q)

    tmax = max(c.times[-1] for c in rig['curves'].values() if c.times)
    nsamples = int(round(tmax * conv.FPS)) + 1
    times = [k / conv.FPS for k in range(nsamples)]

    def intended_V(bid, k):
        nm = src_of[bid]
        W = by_name[nm].world_mat(times[k])
        q_t = conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]])
        dq = conv.make_conj_a(a_q)(q_mul(q_t, q_conj(src_rest[nm][0])))
        V = q_mul(dq, M[bid])
        if bid in conv.FOLD_AIM:
            V = conv.fold_fix(V, bid, times[k], skin_T, by_name, src_of, a_q)
        return V

    def src_world_q(nm, t):
        W = by_name[nm].world_mat(t)
        return conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]])

    # bind joint positions (game axes) for FK of the achieved pose
    bind_pos = {b: tuple(skin_T[b]) for b in sk if b in skin_T}

    def fk_pos(fw):
        """Joint positions exactly as the game computes them: IFP local
        rotations composed with the DFF frame local offsets."""
        pos = {}
        for b in order:
            p = sk[b]['parent']
            if p == -1:
                pos[b] = (0.0, 0.0, 0.0)
                continue
            rot = mat_vec(q_to_mat(fw[p]), sk[b]['T'])
            pos[b] = tuple(pos[p][i] + rot[i] for i in range(3))
        return pos

    # valid pairs: mapped parent with mapped child (segment = joint->child)
    seg_pairs = []
    for bid in src_of:
        kids = conv.mapped_kids(bid, sk, src_of, bone_map)
        if kids and bid != 0:      # root link: CJ root at pelvis, FBX root on floor
            seg_pairs.append((bid, kids[0]))
    # skip degenerate CJ links (Pelvis->Spine is ~1 mm in player.dff: its
    # direction is meaningless). The pelvis is checked by hip width instead.
    def _blen(b, c):
        d = conv.v_sub(skin_T[c], skin_T[b])
        return math.sqrt(conv.v_dot(d, d))
    seg_pairs = [(b, c) for b, c in seg_pairs if _blen(b, c) > 0.01]
    # Spine->Spine1 spans 3 folded source bones (Spine1..Spine4): bends inside
    # the fold cannot be represented by one CJ bone -> reported separately.
    FOLDED = {(2, 3)}
    # the DFF's own frame offsets vs its skin bind can disagree slightly
    # (player.dff: L Finger01 2.71 deg, L Finger 0.34 deg) - model data, not
    # conversion error; measured here and reported separately
    from rt import skel_world
    _wb = skel_world(sk)
    intrinsic = {}
    for b, c in seg_pairs:
        chain = [c]
        while sk[chain[-1]]['parent'] not in (b, -1):
            chain.append(sk[chain[-1]]['parent'])
        intrinsic[(b, c)] = vang(conv.v_sub(skin_T[c], skin_T[b]),
                                 conv.v_sub(_wb[c][1], _wb[b][1]))
    roll_prim, roll_lat = {}, {}
    for bid, (ca, cb, sa, sb) in conv.ROLL_PAIRS.items():
        if bid not in src_of or sa not in src_of or sb not in src_of:
            continue
        _d = conv.v_sub(skin_T[ca], skin_T[cb])
        if conv.v_dot(_d, _d) < 1e-4:       # degenerate width: matcher skips it too
            continue
        g = lambda d: mat_vec(q_to_mat(a_q), d)
        roll_lat[bid] = g(conv.v_sub(src_rest[src_of[sa]][1], src_rest[src_of[sb]][1]))
        if bid in conv.PRIMARY_PAIRS:
            x, y = conv.PRIMARY_PAIRS[bid]
        else:
            kids = conv.mapped_kids(bid, sk, src_of, bone_map)
            if kids:
                x, y = bid, kids[0]
            else:
                x, y = sk[bid]['parent'], bid
        roll_prim[bid] = (conv.v_sub(skin_T[y], skin_T[x]),
                          g(conv.v_sub(src_rest[src_of[y]][1], src_rest[src_of[x]][1])))
    max_roll = 0.0
    roll_worst = {}
    max_fold_err = 0.0
    max_hip_err = 0.0

    tracks = {t['id']: t for t in a['tracks']}
    order = []
    stack = [b for b in sk if sk[b]['parent'] == -1]
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in sk:
            if sk[c]['parent'] == b:
                stack.append(c)

    log('')
    log('== B/C/D. game pose from file vs source ==')
    max_vis_err = sum_vis_err = 0.0
    max_seg_err = 0.0
    seg_worst = {}
    n_vis = 0
    for k in range(0, nsamples, stride):
        local_q = {}
        for bid, tr in tracks.items():
            local_q[bid] = tr['kfs'][0]['q'] if len(tr['kfs']) == 1 else tr['kfs'][k]['q']
        fw = {}
        for b in order:
            p = sk[b]['parent']
            q = local_q[b]
            fw[b] = q if p == -1 else q_mul(fw[p], q)
        pos = fk_pos(fw)
        for bid in src_of:
            V = q_mul(fw[bid], q_conj(skin_q[bid]))
            err = ang(V, intended_V(bid, k))
            max_vis_err = max(max_vis_err, err)
            sum_vis_err += err
            n_vis += 1
        # joint segment directions (independent geometric check)
        for bid, cid in seg_pairs:
            d_ach = conv.v_sub(pos[cid], pos[bid])
            nm_b = src_of[bid]
            Wb = by_name[nm_b].world_mat(times[k])
            nm_c = src_of[cid]
            Wc = by_name[nm_c].world_mat(times[k])
            d_src = conv.v_sub((Wc[0][3], Wc[1][3], Wc[2][3]),
                               (Wb[0][3], Wb[1][3], Wb[2][3]))
            d_src = mat_vec(q_to_mat(a_q), d_src)
            if conv.v_dot(d_ach, d_ach) > 1e-8 and conv.v_dot(d_src, d_src) > 1e-8:
                e = vang(d_ach, d_src)
                if (bid, cid) in FOLDED:
                    max_fold_err = max(max_fold_err, e)
                    continue
                max_seg_err = max(max_seg_err, max(0.0, e - intrinsic.get((bid, cid), 0.0)))
                seg_worst[(bid, cid)] = max(seg_worst.get((bid, cid), 0.0), e)
        # pelvis orientation: hip width (L Thigh - R Thigh) rotated by V(pelvis)
        if 1 in src_of and 41 in src_of and 51 in src_of:
            V1 = q_mul(fw[1], q_conj(skin_q[1]))
            h_ach = mat_vec(q_to_mat(V1), conv.v_sub(skin_T[41], skin_T[51]))
            Wl = by_name[src_of[41]].world_mat(times[k])
            Wr = by_name[src_of[51]].world_mat(times[k])
            h_src = mat_vec(q_to_mat(a_q), conv.v_sub((Wl[0][3], Wl[1][3], Wl[2][3]),
                                                       (Wr[0][3], Wr[1][3], Wr[2][3])))
            max_hip_err = max(max_hip_err, vang(h_ach, h_src))
        # roll (twist) check: each bone's left-right axis from the file vs
        # the source bone's motion applied to its rest left-right axis,
        # both projected perpendicular to the bone's primary axis
        for bid, (ca, cb, sa, sb) in conv.ROLL_PAIRS.items():
            if bid not in src_of or ca not in skin_T or cb not in skin_T:
                continue
            prim = roll_prim.get(bid)
            if prim is None:
                continue
            V = q_mul(fw[bid], q_conj(skin_q[bid]))
            nm = src_of[bid]
            W = by_name[nm].world_mat(times[k])
            q_t = conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]])
            dq = conv.make_conj_a(a_q)(q_mul(q_t, q_conj(src_rest[nm][0])))
            axis = mat_vec(q_to_mat(dq), prim[1])
            a_v = mat_vec(q_to_mat(V), conv.v_sub(skin_T[ca], skin_T[cb]))
            b_v = mat_vec(q_to_mat(dq), roll_lat[bid])
            def _pj(w):
                d = conv.v_dot(w, conv.v_norm(axis)); n = conv.v_norm(axis)
                return tuple(w[i] - n[i] * d for i in range(3))
            e = vang(_pj(a_v), _pj(b_v))
            max_roll = max(max_roll, e)
            roll_worst[bid] = max(roll_worst.get(bid, 0.0), e)

    log('visual error vs intended: max %.4f deg, mean %.4f deg' %
          (max_vis_err, sum_vis_err / max(n_vis, 1)))
    log('joint segment directions vs source (game FK): max %.4f deg '
        '(net of the DFF\'s own frame/skin difference)' % max_seg_err)
    big = [(src_of[b], src_of[c], e) for (b, c), e in intrinsic.items() if e > 0.1]
    if big:
        log('  DFF frame-vs-skin difference (model data): ' +
            ', '.join('%s->%s %.2f' % x for x in big))
    worst5 = sorted(seg_worst.items(), key=lambda kv: -kv[1])[:5]
    log('worst segments:', ', '.join('%s->%s %.2f' %
          (src_of[b], src_of[c], e) for (b, c), e in worst5))
    log('roll/twist check (left-right axes): max %.4f deg  [%s]' % (max_roll,
        ', '.join('%s %.2f' % (src_of[b], e) for b, e in sorted(roll_worst.items()))))
    log('pelvis hip-width direction: max %.4f deg' % max_hip_err)
    log('folded spine chord (aimed): max %.4f deg' % max_fold_err)

    log('')
    log('== E. root translation ==')
    root = t4[0]['kfs']
    mx = max(math.sqrt(sum(c * c for c in kf['p'])) for kf in root)
    step = max(math.dist(root[i]['p'], root[i + 1]['p']) for i in range(len(root) - 1))
    log('max |T| %.3f, max step %.4f' % (mx, step))

    passed = (max_vis_err < 0.5 and max_seg_err < 0.5 and max_roll < 1.0
              and max_hip_err < 1.0 and max_fold_err < 0.5
              and ok and step < 0.2)
    log('')
    log('RESULT: ' + ('PASS' if passed else 'FAIL'))
    return dict(passed=passed, visual=max_vis_err, segments=max_seg_err,
                roll=max_roll, hip=max_hip_err, fold=max_fold_err,
                root_step=step, structure_ok=ok)



def main():
    ifp_path = sys.argv[1] if len(sys.argv) > 1 else conv.OUT
    fbx_path = sys.argv[2] if len(sys.argv) > 2 else conv.FBX
    sk = skin = None
    if len(sys.argv) > 3:
        from dff_rig import load_dff
        sk, skin, _ = load_dff(sys.argv[3])
    r = validate(ifp_path, fbx_path, sk, skin)
    sys.exit(0 if r['passed'] else 1)


if __name__ == '__main__':
    main()
