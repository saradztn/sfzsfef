"""Binary FBX parser (handles both 4-byte and 8-byte node header dialects)."""
import struct, zlib

class Node:
    __slots__ = ('name','props','children')
    def __init__(self, name, props):
        self.name = name; self.props = props; self.children = []
    def find(self, name):
        return [c for c in self.children if c.name == name]
    def first(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None
    def __repr__(self):
        return '<Node %s p%d c%d>' % (self.name, len(self.props), len(self.children))

def parse_props(data):
    props = []
    i = 0
    n = len(data)
    while i < n:
        t = data[i:i+1]; i += 1
        if t == b'Y':
            props.append(struct.unpack_from('<h', data, i)[0]); i += 2
        elif t == b'C':
            props.append(bool(data[i])); i += 1
        elif t == b'I':
            props.append(struct.unpack_from('<i', data, i)[0]); i += 4
        elif t == b'F':
            props.append(struct.unpack_from('<f', data, i)[0]); i += 4
        elif t == b'D':
            props.append(struct.unpack_from('<d', data, i)[0]); i += 8
        elif t == b'L':
            props.append(struct.unpack_from('<q', data, i)[0]); i += 8
        elif t in (b'S', b'R'):
            ln = struct.unpack_from('<I', data, i)[0]; i += 4
            props.append(data[i:i+ln]); i += ln
        elif t in (b'f', b'd', b'l', b'i', b'b'):
            alen, enc, clen = struct.unpack_from('<III', data, i); i += 12
            raw = data[i:i+clen]; i += clen
            if enc == 1:
                raw = zlib.decompress(raw)
            fmt = {b'f':'f', b'd':'d', b'l':'q', b'i':'i', b'b':'b'}[t]
            sz = struct.calcsize('<'+fmt)
            arr = list(struct.unpack('<%d%s' % (alen, fmt), raw[:alen*sz]))
            props.append(arr)
        else:
            raise ValueError('bad prop type %r at %d (blob len %d)' % (t, i-1, n))
    return props

def parse_nodes(data, pos, end, fieldsz, nullsz):
    nodes = []
    while pos < end:
        if fieldsz == 8:
            end_off, nprop, plen, nlen = struct.unpack_from('<QQQ B', data, pos)
        else:
            end_off, nprop, plen, nlen = struct.unpack_from('<III B', data, pos)
        if end_off == 0 and nprop == 0 and plen == 0 and nlen == 0:
            return nodes, pos + nullsz
        name = data[pos+fieldsz*3+1:pos+fieldsz*3+1+nlen]
        pstart = pos + fieldsz*3 + 1 + nlen
        props = parse_props(data[pstart:pstart+plen]) if plen else []
        cstart = pstart + plen
        if end_off > cstart:
            children, _ = parse_nodes(data, cstart, end_off, fieldsz, nullsz)
        else:
            children = []
        nd = Node(name.decode('utf-8','replace'), props)
        nd.children = children
        nodes.append(nd)
        if end_off <= pos:
            raise ValueError('bad EndOffset %d at %d' % (end_off, pos))
        pos = end_off
    return nodes, pos

def parse_fbx(path):
    with open(path,'rb') as f:
        data = f.read()
    assert data[:20] == b'Kaydara FBX Binary  ', 'not binary fbx'
    assert data[20:23] == b'\x00\x1a\x00'
    ver = struct.unpack_from('<I', data, 23)[0]
    # auto-detect header dialect
    last_err = None
    for fieldsz, nullsz in ((8,25),(8,13),(4,13),(2,7)):
        try:
            roots, pos = parse_nodes(data, 27, len(data), fieldsz, nullsz)
            # remainder should only be the file footer
            if pos == len(data) or (len(data) - pos) <= 256:
                return ver, fieldsz, roots
        except Exception as e:
            last_err = e
            continue
    raise ValueError('parse failed: %r' % last_err)

if __name__ == '__main__':
    import sys
    from collections import Counter
    ver, fieldsz, roots = parse_fbx(sys.argv[1])
    print('FBX version:', ver, 'header fields:', fieldsz)
    for r in roots:
        print('ROOT', r.name, 'children:', len(r.children))
    objs = next((r for r in roots if r.name == 'Objects'), None)
    conns = next((r for r in roots if r.name == 'Connections'), None)
    if objs:
        c = Counter(ch.name for ch in objs.children)
        print('Objects:', dict(c))
        for ch in objs.children:
            if ch.name in ('Model','AnimationStack','AnimationLayer','NodeAttribute','Pose'):
                nm = ch.props[1].decode('utf8','replace') if len(ch.props)>1 else '?'
                cls = ch.props[2].decode('utf8','replace') if len(ch.props)>2 else ''
                print(' ', ch.name, repr(nm), cls)
    if conns:
        print('Connections:', len(conns.children))
