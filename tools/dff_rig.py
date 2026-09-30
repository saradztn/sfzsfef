"""Load the skeleton + skin bind of ANY GTA SA ped DFF (generic).

    sk, skin, info = load_dff(path)
      sk   : skeleton dict (rt.skel_from_frames) keyed by SA bone id
      skin : {bone_id: {'R','T','q'}} world bind of each bone from the RpSkin
             PLG (0x116) skinToBone matrices (authoritative for the mesh)

The matrix block is located at the tail of the 0x116 chunk by testing every
tail padding 0..63 B and keeping the one whose numBones 3x3 blocks are the
most orthonormal (works for player.dff: pad=12, err~0.015, and custom peds).
"""
import struct

from dff_skel import extract as extract_frames
from rt import mat_t, q_from_mat, skel_from_frames


def _find_skin_chunks(data):
    ver = struct.unpack_from('<I', data, 8)[0]
    out = []
    for off in range(0, len(data) - 12):
        t, s, v = struct.unpack_from('<III', data, off)
        if t == 0x116 and v == ver and 64 < s <= len(data) - off - 12:
            out.append((off + 12, off + 12 + s))
    return out


def _ortho_err(v):
    """Scale-invariant orthogonality error of the 3x3 part (custom peds are
    often uniformly scaled, e.g. s=1.048)."""
    import math
    if not all(math.isfinite(x) for x in v):
        return 99.0
    R = [v[0:3], v[4:7], v[8:11]]
    L = [math.sqrt(sum(x * x for x in r)) for r in R]
    if min(L) < 1e-4 or max(L) > 1e4:
        return 99.0
    m = sum(L) / 3.0
    e = sum(abs(l - m) for l in L) / m
    for i in range(3):
        for j in range(i + 1, 3):
            e += abs(sum(R[i][k] * R[j][k] for k in range(3))) / (L[i] * L[j])
    e += (abs(v[3]) + abs(v[7]) + abs(v[11])) / m
    return e


def _matrix_block(data, a, b, n):
    best = None
    for pad in range(0, 64):
        start = b - pad - 64 * n
        if start < a:
            break
        err = 0.0
        for i in range(n):
            err += _ortho_err(struct.unpack_from('<16f', data, start + 64 * i))
            if best is not None and err >= best[0]:
                break
        if best is None or err < best[0]:
            best = (err, start, pad)
    return best


def _inv(v):
    """skinToBone (row-stored, possibly uniformly scaled) -> world bind."""
    import math
    rows = [list(v[0:3]), list(v[4:7]), list(v[8:11])]
    sc = sum(math.sqrt(sum(x * x for x in r)) for r in rows) / 3.0
    Rw = [[x / sc for x in r] for r in rows]
    pos = [v[12], v[13], v[14]]
    t = [-sum(Rw[i][k] * pos[k] for k in range(3)) / sc for i in range(3)]
    return Rw, t, sc


def load_dff(path):
    frames = extract_frames(path)
    hier = next((f['hier'] for f in frames if f.get('hier')), None)
    if not hier:
        raise ValueError('DFF has no HAnim hierarchy (not a skinned ped model)')
    data = open(path, 'rb').read()
    n = len(hier)
    cands = _find_skin_chunks(data)
    if not cands:
        raise ValueError('DFF has no RpSkin (0x116) data')
    best = None
    for a, b in cands:
        r = _matrix_block(data, a, b, n)
        if r and (best is None or r[0] < best[0]):
            best = r
    err, start, pad = best
    if err / n > 0.05:
        raise ValueError('could not locate skin bind matrices (err %.3f)' % err)
    skin = {}
    scales = []
    for i, bid in enumerate(hier):
        v = struct.unpack_from('<16f', data, start + 64 * i)
        R, T, sc = _inv(v)
        scales.append(sc)
        q = q_from_mat(R)
        skin[bid] = {'R': R, 'T': T, 'q': list(q)}
    sk = skel_from_frames(frames)
    missing = [b for b in hier if b not in sk]
    info = dict(bones=n, hier=hier, matrix_offset=start, pad=pad,
                ortho_err=err, scale=sum(scales) / len(scales),
                missing_in_frames=missing)
    return sk, skin, info


if __name__ == '__main__':
    import sys
    sk, skin, info = load_dff(sys.argv[1])
    print(info)
