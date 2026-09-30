"""GTA SA DFF RpSkin-PLG (0x116) parser -> per-bone skin data.

Solved layout of the 0x116 payload (verified on player.dff):
  u8 numBones, u8 numUsedBones, u16 flags, u8 usedBones[numUsedBones],
  u8 indices[4*numVerts], f32 weights[4*numVerts],
  matrix[numBones] (64 B each: 4x4 skinToBone, rotation only),
  ~12 B pad.
numVerts is not stored: infer as (chunk - 4 - numUsedBones - 64*numBones - pad)/20.

Output JSON: {bone: {"q": skinToBone quat, "T": WORLD joint position}}.
The world positions come from the HAnim hierarchy (pass the dff_skel frames).
"""
import struct
import sys
from dff_skel import chunks, mat_to_quat


def read_skin_matrices(data, a, e):
    num_bones = data[a]
    num_used = data[a + 1]
    flags = struct.unpack_from('<H', data, a + 2)[0]
    hdr = 4 + num_used
    body = (e - a) - hdr - 64 * num_bones
    # body = 20*numVerts + pad, pad is small (< 64)
    num_verts = None
    for pad in range(0, 64):
        if (body - pad) >= 0 and (body - pad) % 20 == 0:
            num_verts = (body - pad) // 20
            break
    if num_verts is None:
        raise ValueError('cannot infer numVerts')
    off = a + hdr + 20 * num_verts
    if off + 64 * num_bones > e:
        raise ValueError('matrix block overflows chunk')
    mats = []
    for b in range(num_bones):
        m = list(struct.unpack_from('<16f', data, off))
        mats.append(m)
        off += 64
    return dict(num_bones=num_bones, num_used=num_used, flags=flags,
                num_verts=num_verts, matrices=mats)


def extract_skin(data, frames_by_id=None):
    root_end = len(data)
    best = None
    for typ, sz, s, e in chunks(data, 12, root_end):
        for t2, s2, a2, b2 in chunks(data, s, e):
            if t2 == 0x116:
                try:
                    rec = read_skin_matrices(data, a2, b2)
                except Exception:
                    continue
                if best is None or rec['num_verts'] > best['num_verts']:
                    rec['off'] = a2
                    best = rec
    if best is None:
        raise ValueError('no usable 0x116 skin PLG found')
    skin = {}
    for b, m in enumerate(best['matrices']):
        r = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]
        q = mat_to_quat(r)            # skinToBone rotation
        t = (m[3], m[7], m[11])       # zero in GTA skins; world T from HAnim
        if frames_by_id and b in frames_by_id:
            t = tuple(frames_by_id[b]['pos'])
        skin[b] = {'q': list(q), 'T': list(t)}
    return skin, best


if __name__ == '__main__':
    import json
    dff_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'player_skin.json'
    data = open(dff_path, 'rb').read()
    frames = None
    skel_out = out_path.replace('skin', 'skel')
    try:
        sk = json.load(open(skel_out))
        frames = {f['idx']: f for f in (sk['frames'] if isinstance(sk, dict) else sk)}
    except Exception:
        pass
    skin, meta = extract_skin(data, frames)
    with open(out_path, 'w') as f:
        json.dump(skin, f)
    print('wrote', out_path, 'bones=%d numVerts=%d flags=%d' %
          (meta['num_bones'], meta['num_verts'], meta['flags']))
