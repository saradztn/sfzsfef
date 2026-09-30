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


def main():
    import json
    ifp_path = sys.argv[1] if len(sys.argv) > 1 else conv.OUT
    fbx_path = sys.argv[2] if len(sys.argv) > 2 else conv.FBX

    print('== A. structure ==')
    f = read_anp3_raw(ifp_path)
    print('file %s: %d bytes, pkg=%r, anims=%d' %
          (ifp_path, f['total'], f['name'], len(f['anims'])))
    a = f['anims'][0]
    print('anim %r: %d tracks, flags=%d, dsize=%d' %
          (a['name'], len(a['tracks']), a['flags'], a['dsize']))
    t4 = [t for t in a['tracks'] if t['type'] == 4]
    print('type-4 tracks:', [(t['name'], t['id'], len(t['kfs'])) for t in t4])
    ok = True
    for t in a['tracks']:
        ts = [kf['t'] for kf in t['kfs']]
        if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
            print('!! non-monotonic time in', t['name']); ok = False
    print('time monotonic + unit quats:', 'OK' if ok else 'FAIL')

    # ---- shared model data ----
    sk = load_skel(conv.SKEL)
    skin = {int(k): v for k, v in json.load(open(conv.SKIN)).items()}
    skin_q = {b: tuple(v['q']) for b, v in skin.items()}
    skin_T = {b: tuple(v['T']) for b, v in skin.items()}
    rig = load_rig(fbx_path)
    by_name = {}
    for m in rig['models'].values():
        by_name.setdefault(m.name, m)
    a_q = conv.q_from_mat(conv.A_ROWS_YUP)
    src_of = conv.build_src_of(conv.BONE_MAP)
    M, src_rest = conv.build_matchers(sk, skin_q, skin_T, by_name,
                                      src_of, conv.BONE_MAP, a_q)

    tmax = max(c.times[-1] for c in rig['curves'].values() if c.times)
    nsamples = int(round(tmax * conv.FPS)) + 1
    times = [k / conv.FPS for k in range(nsamples)]

    def intended_V(bid, k):
        nm = src_of[bid]
        W = by_name[nm].world_mat(times[k])
        q_t = conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]])
        dq = conv.make_conj_a(a_q)(q_mul(q_t, q_conj(src_rest[nm][0])))
        return q_mul(dq, M[bid])

    def src_world_q(nm, t):
        W = by_name[nm].world_mat(t)
        return conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]])

    # bind joint positions (game axes) for FK of the achieved pose
    bind_pos = {b: tuple(skin_T[b]) for b in sk if b in skin_T}

    def fk_pos(fw):
        """Achieved joint positions through the HAnim bind offsets, rotated
        by each parent bone's visual V (posed offset = V * bind offset)."""
        pos = {}
        for b in order:
            p = sk[b]['parent']
            if p == -1:
                pos[b] = (0.0, 0.0, 0.0)
                continue
            owner = p
            Vp = q_mul(fw[owner], skin_q[owner])      # fw = V*conj(skinToBone)
            off = tuple(bind_pos.get(b, (0, 0, 0))[i] - bind_pos.get(p, (0, 0, 0))[i]
                        for i in range(3))
            rot = mat_vec(q_to_mat(Vp), off)
            pos[b] = tuple(pos[p][i] + rot[i] for i in range(3))
        return pos

    # valid pairs: mapped parent with mapped child (segment = joint->child)
    seg_pairs = [(bid, [c for c in sk if sk[c]['parent'] == bid][0])
                 for bid in src_of
                 if [c for c in sk if sk[c]['parent'] == bid]
                 and [c for c in sk if sk[c]['parent'] == bid][0] in src_of]

    tracks = {t['id']: t for t in a['tracks']}
    order = []
    stack = [b for b in sk if sk[b]['parent'] == -1]
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in sk:
            if sk[c]['parent'] == b:
                stack.append(c)

    print()
    print('== B/C/D. game pose from file vs source ==')
    max_vis_err = sum_vis_err = 0.0
    max_seg_err = 0.0
    max_head_roll = 0.0
    seg_worst = {}
    n_vis = 0
    for k in range(0, nsamples, 3):
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
            V = q_mul(fw[bid], skin_q[bid])
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
                max_seg_err = max(max_seg_err, e)
                seg_worst[(bid, cid)] = max(seg_worst.get((bid, cid), 0.0), e)
        # head-roll regression check (Neck->Head pair, convention-clean)
        if 5 in src_of and 4 in src_of:
            V5 = q_mul(fw[5], skin_q[5])
            V4 = q_mul(fw[4], skin_q[4])
            rel_ach = q_mul(q_conj(V4), V5)
            W5 = by_name[src_of[5]].world_mat(times[k])
            W4 = by_name[src_of[4]].world_mat(times[k])
            q5 = conv.q_from_world([W5[0][:3], W5[1][:3], W5[2][:3]])
            q4 = conv.q_from_world([W4[0][:3], W4[1][:3], W4[2][:3]])
            rel_src = q_mul(a_q, q_mul(q_mul(q_conj(q4), q5), q_conj(a_q)))
            max_head_roll = max(max_head_roll, ang(rel_ach, rel_src))

    print('visual error vs intended: max %.4f deg, mean %.4f deg' %
          (max_vis_err, sum_vis_err / max(n_vis, 1)))
    print('joint segment directions vs source (FK positions): max %.4f deg'
          % max_seg_err)
    worst5 = sorted(seg_worst.items(), key=lambda kv: -kv[1])[:5]
    print('worst segments:', ', '.join('%s->%s %.2f' %
          (src_of[b], src_of[c], e) for (b, c), e in worst5))
    print('head-roll check (Neck->Head rel): max %.4f deg' % max_head_roll)

    print()
    print('== E. root translation ==')
    root = t4[0]['kfs']
    mx = max(math.sqrt(sum(c * c for c in kf['p'])) for kf in root)
    step = max(math.dist(root[i]['p'], root[i + 1]['p']) for i in range(len(root) - 1))
    print('max |T| %.3f, max step %.4f' % (mx, step))

    passed = (max_vis_err < 0.5 and max_seg_err < 0.5 and max_head_roll < 1.0
              and ok and step < 0.2)
    print()
    print('RESULT:', 'PASS' if passed else 'FAIL')


if __name__ == '__main__':
    main()
