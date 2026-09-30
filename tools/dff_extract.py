"""GTA SA DFF -> player skeleton + skin data (definitive extractor).

Verified layout (player.dff, 32-bone CJ skin):
  HAnim PLG 0x11e: u32 numBones, u32 flags, u32 nodeIDs[n], u32 numFrames,
                   per frame: f32[9] LOCAL rot, f32[3] LOCAL pos
  Skin PLG 0x116:  u8 numBones, u8 numUsedBones, u16 flags,
                   u8 usedBones[numUsedBones], u8 indices[4*nv],
                   f32 weights[4*nv], matrix[numBones] (64 B: skinToBone,
                   rotation only), pad.  nv = (chunk - 4 - numUsed - 64*n - pad)/20

Outputs:
  --skel OUT.json : list of {idx, name, bone, rot, pos, parent, world_q}
                    with WORLD transforms (composed from the local frames)
  --skin OUT.json : {bone: {q: skinToBone quat, T: WORLD joint position}}
"""
import json
import struct
import sys

from dff_skel import chunks, mat_to_quat, qmul, qrot

# CJ hierarchy (nodeID -> parent nodeID), from the HAnim hash families
PARENTS = {
    0: -1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 8: 5, 6: 5, 7: 5,
    31: 2, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35,
    41: 2, 42: 41, 43: 42, 44: 43,
    21: 2, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25,
    51: 2, 52: 51, 53: 52, 54: 53,
    201: 1, 301: 1, 302: 1,
}
NAMES = {
    0: 'Root', 1: 'Pelvis', 2: 'Spine', 3: 'Spine1', 4: 'Neck', 5: 'Head',
    8: 'Jaw', 6: 'L Brow', 7: 'R Brow',
    31: 'Bip01 L Clavicle', 32: 'L UpperArm', 33: 'L ForeArm', 34: 'L Hand',
    35: 'L Finger', 36: 'L Finger01',
    41: 'L Thigh', 42: 'L Calf', 43: 'L Foot', 44: 'L Toe0',
    21: 'Bip01 R Clavicle', 22: 'R UpperArm', 23: 'R ForeArm', 24: 'R Hand',
    25: 'R Finger', 26: 'R Finger01',
    51: 'R Thigh', 52: 'R Calf', 53: 'R Foot', 54: 'R Toe0',
    201: 'Belly', 301: 'R Breast', 302: 'L Breast',
}


def find_hanim(data):
    for typ, sz, s, e in chunks(data, 12, len(data)):
        if typ != 0x01:
            continue
        for t2, s2, a2, b2 in chunks(data, s, e):
            if t2 == 0x11E:
                return a2, b2
    raise ValueError('no HAnim 0x11e')


def find_skin(data):
    best = None
    for typ, sz, s, e in chunks(data, 12, len(data)):
        if typ != 0x01:
            continue
        for t2, s2, a2, b2 in chunks(data, s, e):
            if t2 == 0x116:
                n = data[a2]
                nu = data[a2 + 1]
                hdr = 4 + nu
                body = (b2 - a2) - hdr - 64 * n
                for pad in range(0, 64):
                    if (body - pad) >= 0 and (body - pad) % 20 == 0:
                        nv = (body - pad) // 20
                        rec = (a2, b2, n, nu, nv, a2 + hdr + 20 * nv)
                        if best is None or nv > best[4]:
                            best = rec
                        break
    if best is None:
        raise ValueError('no skin 0x116')
    return best


def extract(data):
    a, e = find_hanim(data)
    num_bones, flags = struct.unpack_from('<II', data, a)
    node_ids = list(struct.unpack_from('<%dI' % num_bones, data, a + 8))
    num_frames = struct.unpack_from('<I', data, a + 8 + 4 * num_bones)[0]
    off = a + 12 + 4 * num_bones
    raw = []
    for i in range(num_frames):
        m = list(struct.unpack_from('<12f', data, off))
        off += 48
        raw.append((m[0:9], m[9:12]))
    assert num_frames == num_bones

    # compose world transforms along the CJ hierarchy
    world = {}
    frames = []
    for nid in node_ids:
        i = node_ids.index(nid)
        rm, rt = raw[i]
        ql = mat_to_quat(rm)
        p = PARENTS.get(nid, -1)
        if p in world:
            pq, pt = world[p]
            q = qmul(pq, ql)
            r = qrot(pq, rt)
            t = (pt[0] + r[0], pt[1] + r[1], pt[2] + r[2])
        else:
            q, t = ql, tuple(rt)
        world[nid] = (q, t)
        x, y, z, w = q
        rm_w = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
        frames.append(dict(idx=nid, name=NAMES.get(nid), bone=nid,
                           rot=[rm_w[0:3], rm_w[3:6], rm_w[6:9]],
                           pos=list(t), parent=p, world_q=list(q)))

    a2, b2, n, nu, nv, moff = find_skin(data)
    skin = {}
    for bi in range(n):
        m = list(struct.unpack_from('<16f', data, moff + 64 * bi))
        r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]
        q = mat_to_quat(r)                       # skinToBone rotation
        nid = node_ids[bi] if bi < len(node_ids) else bi
        t = world[nid][1] if nid in world else (0.0, 0.0, 0.0)
        skin[nid] = {'q': list(q), 'T': list(t)}
    return frames, skin, dict(num_bones=n, num_used=nu, num_verts=nv)


if __name__ == '__main__':
    args = sys.argv[1:]
    dff = args[0]
    data = open(dff, 'rb').read()
    frames, skin, meta = extract(data)
    print('bones=%d used=%d verts=%d' % (meta['num_bones'], meta['num_used'],
                                         meta['num_verts']))
    # ground-truth spot checks (player.dff, verified independently)
    gt = {1: ((0, 0, 1.048), (-0.002, -0.124, 0.118, 0.985)),
          32: ((-0.18, 0.03, 1.51), None),
          34: (None, (-0.73, 0.111, 0.665, 0.12)),
          41: ((0.11, 0, 0.95), (0.476, 0.523, 0.476, 0.523))}
    for b, (tp, tq) in gt.items():
        got_t = tuple(round(x, 3) for x in skin[b]['T'])
        got_q = tuple(round(x, 3) for x in skin[b]['q'])
        print('bone %-3d T=%s (want %s)  q=%s (want %s)' %
              (b, got_t, tp, got_q, tq))
    if len(args) > 2 and args[1] == '--skel':
        with open(args[2], 'w') as f:
            json.dump(frames, f)
        print('wrote', args[2])
    if len(args) > 4 and args[3] == '--skin':
        with open(args[4], 'w') as f:
            json.dump(skin, f)
        print('wrote', args[4])
