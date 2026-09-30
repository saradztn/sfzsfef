"""Validate the converted IFP against the source FBX motion.

Checks:
  A. structure: independent raw parse of the ANP3 (mirrors ped.ifp dumps)
  B. quantization: intended local quats vs file quats (angle error)
  C. pose reconstruction: apply the file quats game-style to player.dff and
     measure the achieved visual vs the intended visual (angle)
  D. tracking: achieved bone-aim directions vs the source FBX directions
  E. root translation continuity
"""
import math
import os
import struct
import sys

sys.path.insert(0, '/home/user/sfzsfef/tools')
sys.path.insert(0, '/home/user/sfzsfef/ref')
from rt import (q_mul, q_conj, q_norm, q_from_mat, q_to_mat, mat_vec,
                mat_norm_rows, load_skel)
from fbx_rig import load_rig
import fbx2ifp as conv

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
                kfs.append({'q': (qx / 4096, qy / 4096, qz / 4096, qw / 4096),
                            't': t, 'p': (px, py, pz)})
                fo += sz
            tracks.append({'name': bname, 'type': ftype, 'id': bid, 'kfs': kfs})
            o += 36 + nkf * sz
        anims.append({'name': aname, 'tracks': tracks, 'dsize': dsize, 'flags': flags})
        off = o
    return {'name': name, 'anims': anims, 'size': size, 'total': len(data)}


def ang(q1, q2):
    # game normalises keyframe quats on load; compare as rotations
    q1 = q_norm(q1); q2 = q_norm(q2)
    d = q_mul(q_conj(q1), q2)
    return 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(d[3])))))


def main():
    import json
    ifp_path = sys.argv[1] if len(sys.argv) > 1 else conv.OUT

    print('== A. structure ==')
    f = read_anp3_raw(ifp_path)
    print('file %s: %d bytes, pkg=%r, anims=%d' %
          (ifp_path, f['total'], f['name'], len(f['anims'])))
    a = f['anims'][0]
    print('anim %r: %d tracks, flags=%d, dsize=%d' %
          (a['name'], len(a['tracks']), a['flags'], a['dsize']))
    t4 = [t for t in a['tracks'] if t['type'] == 4]
    print('type-4 tracks:', [(t['name'], t['id'], len(t['kfs'])) for t in t4])
    ids = sorted(t['id'] for t in a['tracks'])
    print('bone ids:', ids)
    ok = True
    for t in a['tracks']:
        ts = [kf['t'] for kf in t['kfs']]
        if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
            print('!! non-monotonic time in', t['name']); ok = False
        for kf in t['kfs']:
            n = math.sqrt(sum(c * c for c in kf['q']))
            if abs(n - 1) > 0.01:
                print('!! non-unit quat', t['name'], kf['q']); ok = False
    print('time monotonic + unit quats:', 'OK' if ok else 'FAIL')

    # ---- recompute intended locals ----
    sk = load_skel(conv.SKEL)
    skin = {int(k): v for k, v in json.load(open(conv.SKIN)).items()}
    skin_q = {b: tuple(v['q']) for b, v in skin.items()}
    skin_T = {b: tuple(v['T']) for b, v in skin.items()}

    rig = load_rig(conv.FBX)
    by_name = {}
    for m in rig['models'].values():
        by_name.setdefault(m.name, m)
    tmax = max(c.times[-1] for c in rig['curves'].values() if c.times)
    nsamples = int(round(tmax * conv.FPS)) + 1
    times = [k / conv.FPS for k in range(nsamples)]

    src_rest = {}
    need = set(conv.BONE_MAP)
    for nm, (_b, _bn, ac) in conv.BONE_MAP.items():
        if ac:
            need.add(ac)
    for nm in need:
        W = by_name[nm].world_mat(0.0, eval_curves=False)
        src_rest[nm] = (conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]]),
                        (W[0][3], W[1][3], W[2][3]))

    def cj_aim(bid):
        kids = [c for c in sk if sk[c]['parent'] == bid]
        if kids and bid in skin_T and kids[0] in skin_T:
            d = conv.v_sub(skin_T[kids[0]], skin_T[bid])
            if conv.v_dot(d, d) > 1e-8:
                return conv.v_norm(d)
        return conv.v_norm(mat_vec(q_to_mat(skin_q[bid]), (1.0, 0.0, 0.0)))

    src_of = {}
    for nm, (bid, _bn, aim_child) in conv.BONE_MAP.items():
        prev = src_of.get(bid)
        if prev is None or (conv.BONE_MAP[prev][2] is None and aim_child is not None):
            src_of[bid] = nm
    M = {}
    for bid, nm in src_of.items():
        q_rest, p_rest = src_rest[nm]
        aim_child = conv.BONE_MAP[nm][2]
        if aim_child and aim_child in src_rest:
            d_src = conv.v_sub(src_rest[aim_child][1], p_rest)
            if conv.v_dot(d_src, d_src) < 1e-8:
                d_src = mat_vec(q_to_mat(q_rest), (1.0, 0.0, 0.0))
        else:
            d_src = mat_vec(q_to_mat(q_rest), (1.0, 0.0, 0.0))
        M[bid] = conv.min_rot(cj_aim(bid), mat_vec(q_to_mat(conv.A_Q), d_src))
    M[0] = M[1]

    # intended V per sample per bone
    def intended_V(bid, k):
        t = times[k]
        W = by_name[src_of[bid]].world_mat(t)
        q_t = conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]])
        dq = conv.conj_A(q_mul(q_t, q_conj(src_rest[src_of[bid]][0])))
        return q_mul(dq, M[bid])

    # ---- B+C: apply the FILE's quantized locals game-style and compare ----
    print()
    print('== B/C. quantized file -> game pose -> visual vs intended ==')
    tracks = {t['id']: t for t in a['tracks']}
    order = []
    stack = [b for b in sk if sk[b]['parent'] == -1]
    while stack:
        b = stack.pop(0)
        order.append(b)
        for c in sk:
            if sk[c]['parent'] == b:
                stack.append(c)

    max_vis_err = 0.0
    sum_vis_err = 0.0
    n_vis = 0
    max_dir_err = 0.0
    worst = None
    for k in range(0, nsamples, 3):  # every 3rd sample for speed
        # locals from file (nearest key: times match exactly at 30 Hz)
        local_q = {}
        for bid, tr in tracks.items():
            if len(tr['kfs']) == 1:
                local_q[bid] = q_norm(tr['kfs'][0]['q'])
            else:
                local_q[bid] = q_norm(tr['kfs'][k]['q'])
        # frame worlds
        fw = {}
        for b in order:
            p = sk[b]['parent']
            q = local_q[b]
            fw[b] = q if p == -1 else q_mul(fw[p], q)
        # visual
        for bid in src_of:
            if bid == 0:
                continue  # root has no meaningful bone aim (zero-length link)
            V = q_mul(fw[bid], q_conj(skin_q[bid]))  # fw * skinToBone
            err = ang(V, intended_V(bid, k))
            max_vis_err = max(max_vis_err, err)
            sum_vis_err += err
            n_vis += 1
            if err > 0.5 and (worst is None or err > worst[1]):
                worst = (bid, err, k)
            # tracking: achieved aim vs source aim
            d_ach = mat_vec(q_to_mat(V), cj_aim(bid))
            W = by_name[src_of[bid]].world_mat(times[k])
            q_rest = src_rest[src_of[bid]][0]
            d_src = mat_vec(q_to_mat(conv.conj_A(q_mul(
                conv.q_from_world([W[0][:3], W[1][:3], W[2][:3]]),
                q_conj(q_rest)))), mat_vec(q_to_mat(M[bid]), cj_aim(bid)))
            # simpler: source direction in game axes
            aim_child = conv.BONE_MAP[src_of[bid]][2]
            if aim_child:
                p_c = by_name[aim_child].world_mat(times[k])
                p_b = W
                dsrc = conv.v_norm(conv.v_sub(
                    (p_c[0][3], p_c[1][3], p_c[2][3]),
                    (p_b[0][3], p_b[1][3], p_b[2][3])))
                dsrc_g = conv.v_norm(mat_vec(q_to_mat(conv.A_Q), dsrc))
                c = max(-1.0, min(1.0, conv.v_dot(d_ach, dsrc_g)))
                aerr = math.degrees(math.acos(c))
                max_dir_err = max(max_dir_err, aerr)

    print('visual error vs intended: max %.4f deg, mean %.4f deg (%d samples)' %
          (max_vis_err, sum_vis_err / max(n_vis, 1), n_vis))
    print('bone-aim tracking vs source: max %.4f deg' % max_dir_err)
    if worst:
        print('worst visual: bone %d err %.3f at k=%d' % worst)

    # ---- E: root translation ----
    print()
    print('== E. root translation ==')
    root = tracks[0]
    p0 = root['kfs'][0]['p']
    pl = root['kfs'][-1]['p']
    mx = max(math.sqrt(sum(c * c for c in kf['p'])) for kf in root['kfs'])
    print('root T: start %s end %s max|T| %.3f' %
          (tuple(round(c, 3) for c in p0), tuple(round(c, 3) for c in pl), mx))
    step = max(math.dist(root['kfs'][i]['p'], root['kfs'][i + 1]['p'])
               for i in range(len(root['kfs']) - 1))
    print('max step between keys: %.4f (smooth if small)' % step)

    print()
    print('RESULT:', 'PASS' if max_vis_err < 0.5 and max_dir_err < 1.0 else 'CHECK')


if __name__ == '__main__':
    main()
