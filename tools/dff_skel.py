"""DFF frame-list + HAnim extractor (RW 3.x), tolerant of size-flag quirks."""
import struct, sys, json, math

def parse_chunks(data, pos, end):
    out = []
    while pos + 12 <= end:
        cid, size, ver = struct.unpack_from('<III', data, pos)
        cstart = pos + 12
        cend = cstart + size
        if cend > end:
            size2 = size & 0x00FFFFFF
            if cstart + size2 <= end:
                size, cend = size2, cstart + size2
            else:
                break
        out.append((cid, ver, cstart, cend))
        pos = cend
    return out

def extract(path):
    data = open(path, 'rb').read()
    top = parse_chunks(data, 0, len(data))
    clump = next(c for c in top if c[0] == 0x10)
    children = parse_chunks(data, clump[2], clump[3])
    fl = next(c for c in children if c[0] == 0x0E)
    subs = parse_chunks(data, fl[2], fl[3])
    fs = next(c for c in subs if c[0] == 0x01)
    count, = struct.unpack_from('<I', data, fs[2])
    rec = (fs[3] - fs[2] - 4) // count
    frames = []
    p = fs[2] + 4
    for i in range(count):
        raw = data[p:p + rec]
        f = struct.unpack_from('<12f', raw, 0)
        parent, = struct.unpack_from('<i', raw, 48)
        frames.append({'idx': i, 'rot': [list(f[0:3]), list(f[3:6]), list(f[6:9])],
                       'pos': list(f[9:12]), 'parent': parent, 'name': None, 'bone': None})
        p += rec
    exts = [c for c in subs if c[0] == 0x03]
    ei = 0
    for es, ee in ((c[2], c[3]) for c in exts):
        nm = None; bone = None; hier = None
        for c2, v2, s2, e2 in parse_chunks(data, es, ee):
            if c2 == 0x253F2FE:
                ln, = struct.unpack_from('<I', data, s2)
                raw = data[s2 + 4:s2 + 4 + ln]
                nm = raw.split(b'\x00')[0].decode('latin1', 'replace').strip('\r\n')
            elif c2 == 0x11E:
                ver_, boneid, numnodes = struct.unpack_from('<III', data, s2)
                bone = boneid
                if numnodes > 0:
                    nodes = []
                    pp = s2 + 20
                    for k in range(numnodes):
                        nid = struct.unpack_from('<I', data, pp)[0]
                        nodes.append(nid)
                        pp += 12
                    hier = nodes
        # empty extension (size 0) still belongs to clump-root frame
        if ei < len(frames):
            frames[ei]['name'] = nm
            frames[ei]['bone'] = bone
            if hier:
                frames[ei]['hier'] = hier
            ei += 1
    return frames

def mat_normalize(rows):
    out = []
    for r in rows:
        n = math.sqrt(sum(v * v for v in r)) or 1.0
        out.append([v / n for v in r])
    return out

if __name__ == '__main__':
    frames = extract(sys.argv[1])
    print('frames:', len(frames))
    for f in frames:
        r = f['rot']
        sc = math.sqrt(sum(r[0][j] ** 2 for j in range(3)))
        h = f.get('hier')
        print('%2d par=%3d bone=%-6s name=%-22r pos=(%8.5f %8.5f %8.5f) scl=%.6f' % (
            f['idx'], f['parent'], str(f['bone']), f['name'], f['pos'][0], f['pos'][1], f['pos'][2], sc))
        print('     R=[%9.6f %9.6f %9.6f | %9.6f %9.6f %9.6f | %9.6f %9.6f %9.6f]' % tuple(r[0] + r[1] + r[2]))
        if h:
            print('     HIERARCHY (%d nodes): %s' % (len(h), h))
    if len(sys.argv) > 2:
        json.dump(frames, open(sys.argv[2], 'w'), indent=1)
        print('json saved:', sys.argv[2])
