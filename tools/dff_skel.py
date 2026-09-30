"""GTA SA DFF -> HAnim skeleton extractor (RW chunk walk, raw 12-byte headers).

Outputs (stdout or --out):
  frames: list of {idx, name, bone, rot, pos, parent} with WORLD transforms
  (composed from the HAnim local frames).

HAnim PLG (0x11e) layout:
  u32 numBones, u32 flags, u32 nodeIDs[numBones], u32 numFrames,
  per frame: f32[9] rot (LOCAL), f32[3] pos (LOCAL)
Skin PLG (0x116) nodeIDs (matrix order) are included under 'skin_node_ids'.
"""
import json
import struct
import sys

BONE_NAMES = {
    0: "Root", 1: "Pelvis", 2: "Spine", 3: "L UpperArm", 4: "L ForeArm",
    5: "L Hand", 6: "L Finger", 7: "L Finger01", 8: "L Finger02",
    9: "L Toe0", 10: "L Toe01", 11: "L Toe02", 12: "L Foot",
    13: "L Calf", 14: "L Thigh", 15: "L Clavicle", 16: "Neck",
    17: "Head", 18: "Jaw", 19: "L Brow", 20: "L Eye lid",
    21: "L Eye", 22: "R Brow", 23: "R Eye lid", 24: "R Eye",
    25: "R UpperArm", 26: "R ForeArm", 27: "R Hand", 28: "R Finger",
    29: "R Finger01", 30: "R Finger02", 31: "R Toe0", 32: "R Toe01",
    33: "R Toe02", 34: "R Foot", 35: "R Calf", 36: "R Thigh",
    37: "R Clavicle", 38: "Belly", 39: "R Breast", 40: "L Breast",
}


def chunks(data, off, end):
    end = min(end, len(data))
    while off + 12 <= end:
        typ, sz, _ = struct.unpack_from('<III', data, off)
        yield typ, sz, off + 12, off + 12 + sz
        off += 12 + sz


def mat_to_quat(m):
    tr = m[0] + m[4] + m[8]
    if tr > 0:
        S = (tr + 1.0) ** 0.5 * 2
        return ((m[7] - m[5]) / S, (m[2] - m[6]) / S,
                (m[3] - m[1]) / S, 0.25 * S)
    if m[0] > m[4] and m[0] > m[8]:
        S = (1.0 + m[0] - m[4] - m[8]) ** 0.5 * 2
        return (0.25 * S, (m[1] + m[3]) / S,
                (m[2] + m[6]) / S, (m[7] - m[5]) / S)
    if m[4] > m[8]:
        S = (1.0 + m[4] - m[0] - m[8]) ** 0.5 * 2
        return ((m[1] + m[3]) / S, 0.25 * S,
                (m[5] + m[7]) / S, (m[2] - m[6]) / S)
    S = (1.0 + m[8] - m[0] - m[4]) ** 0.5 * 2
    return ((m[2] + m[6]) / S, (m[5] + m[7]) / S, 0.25 * S,
            (m[3] - m[1]) / S)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def qrot(q, v):
    x, y, z, w = q
    tx = 2 * (y * v[2] - z * v[1])
    ty = 2 * (z * v[0] - x * v[2])
    tz = 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + (y * tz - z * ty),
            v[1] + w * ty + (z * tx - x * tz),
            v[2] + w * tz + (x * ty - y * tx))


def parse_skin_ids(data, e):
    num_bones, flags = struct.unpack_from('<II', data, e)
    ids = list(struct.unpack_from('<%dI' % num_bones, data, e + 8))
    return {'num_bones': num_bones, 'flags': flags, 'node_ids': ids}


def extract_skeleton(data):
    root_end = len(data)
    hits = []
    for typ, sz, s, e in chunks(data, 12, root_end):
        for t2, s2, a2, b2 in chunks(data, s, e):
            if t2 == 0x11E:
                hits.append((a2, b2))
    if not hits:
        raise ValueError('no HAnim 0x11e found')
    a, e = hits[0]
    num_bones, flags = struct.unpack_from('<II', data, a)
    node_ids = list(struct.unpack_from('<%dI' % num_bones, data, a + 8))
    num_frames = struct.unpack_from('<I', data, a + 8 + 4 * num_bones)[0]
    off = a + 12 + 4 * num_bones
    raw = []
    for i in range(num_frames):
        m = list(struct.unpack_from('<12f', data, off))
        off += 48
        raw.append((m[0:9], m[9:12]))
    parent = {nid: (-1 if nid == 0 else (nid & ~15) | (nid & 15) - 1)
              for nid in node_ids}
    # compose world transforms
    world = {}
    frames = []
    remaining = set(node_ids)
    order = []
    while remaining:
        progressed = False
        for nid in list(remaining):
            p = parent[nid]
            if p not in node_ids or p in world:
                rm, rt = raw[node_ids.index(nid)]
                if p in world:
                    pq, pt = world[p]
                    q = qmul(pq, mat_to_quat(rm))
                    t = (pt[0] + qrot(pq, rt)[0],
                         pt[1] + qrot(pq, rt)[1],
                         pt[2] + qrot(pq, rt)[2])
                else:
                    q, t = mat_to_quat(rm), tuple(rt)
                world[nid] = (q, t)
                order.append(nid)
                remaining.discard(nid)
                progressed = True
        if not progressed:
            for nid in list(remaining):
                rm, rt = raw[node_ids.index(nid)]
                world[nid] = (mat_to_quat(rm), tuple(rt))
                order.append(nid)
                remaining.discard(nid)
    for nid in node_ids:
        q, t = world[nid]
        rm = raw[node_ids.index(nid)][0]
        # store the WORLD rotation matrix (load_skel reads 'rot' as world)
        x, y, z, w = q
        rm_w = [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w),
                2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w),
                2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
        frames.append(dict(idx=nid, name=BONE_NAMES.get(nid), bone=nid,
                           rot=[rm_w[0:3], rm_w[3:6], rm_w[6:9]],
                           pos=list(t), parent=parent[nid],
                           world_q=list(q)))
    skin_ids = None
    for typ, sz, s, e2 in chunks(data, 12, root_end):
        for t2, s2, a2, b2 in chunks(data, s, e2):
            if t2 == 0x116 and b2 - a2 > 200000:
                try:
                    skin_ids = parse_skin_ids(data, a2)
                except Exception:
                    pass
    return {'frames': frames, 'skin_node_ids': skin_ids}


if __name__ == '__main__':
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    data = open(path, 'rb').read()
    sk = extract_skeleton(data)
    if out:
        with open(out, 'w') as f:
            json.dump(sk['frames'], f)   # load_skel-compatible list
        print('wrote', out, '(%d frames)' % len(sk['frames']))
    else:
        print(json.dumps(sk, indent=1)[:2000])
