import struct, sys, zlib
from collections import Counter

def parse_props(data):
    """Return list of python values for one property list blob."""
    props = []
    i = 0
    n = len(data)
    while i < n:
        t = data[i:i+1]; i += 1
        if t in (b'Y',):
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
            raise ValueError('bad prop type %r at %d' % (t, i-1))
    return props

class Node:
    __slots__ = ('name','props','children')
    def __init__(self, name, props):
        self.name = name; self.props = props; self.children = []
    def find(self, name):
        return [c for c in self.children if c.name == name]
    def first(self, name):
        for c in self.children:
            if c.name == name: return c
        return None

def parse_nodes(data, pos, end, ver):
    nodes = []
    while pos < end:
        if ver >= 7500:
            end_off, nprop, plen, nlen = struct.unpack_from('<III B', data, pos)
            hdr = 13
        else:
            end_off, nprop, plen, nlen = struct.unpack_from('<HHH B', data, pos)
            hdr = 7
        if end_off == 0 and nprop == 0 and plen == 0 and nlen == 0:
            return nodes, pos + hdr
        name = data[pos+hdr:pos+hdr+nlen]
        pstart = pos + hdr + nlen
        props = parse_props(data[pstart:pstart+plen])
        cstart = pstart + plen
        if end_off > cstart:
            children, _ = parse_nodes(data, cstart, end_off, ver)
        else:
            children = []
        nd = Node(name.decode('utf-8','replace'), props)
        nd.children = children
        nodes.append(nd)
        pos = end_off
    return nodes, pos

def parse_fbx(path):
    with open(path,'rb') as f:
        data = f.read()
    assert data[:20] == b'Kaydara FBX Binary  ', 'not binary fbx'
    assert data[20:23] == b'\x00\x1a\x00'
    ver = struct.unpack_from('<I', data, 23)[0]
    roots, _ = parse_nodes(data, 27, len(data), ver)
    return ver, roots

if __name__ == '__main__':
    ver, roots = parse_fbx(sys.argv[1])
    print('FBX version:', ver)
    for r in roots:
        print('ROOT', r.name, 'props:', len(r.props), 'children:', len(r.children))
    objs = next((r for r in roots if r.name == 'Objects'), None)
    conns = next((r for r in roots if r.name == 'Connections'), None)
    if objs:
        c = Counter(ch.name for ch in objs.children)
        print('Objects:', dict(c))
        for ch in objs.children:
            if ch.name in ('Model','AnimationStack','AnimationLayer','NodeAttribute'):
                nm = ch.props[1].decode('utf8','replace') if len(ch.props)>1 else '?'
                cls = ch.props[2].decode('utf8','replace') if len(ch.props)>2 else ''
                print(' ', ch.name, repr(nm), cls)
    if conns:
        print('Connections:', len(conns.children))
