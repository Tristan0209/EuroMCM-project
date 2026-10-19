import hashlib
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('C:/Users/吴连杰/Desktop/2026_Problem_Lambda.pdf')


def literal(value):
    out = bytearray()
    i = 0
    escapes = {ord('n'): 10, ord('r'): 13, ord('t'): 9, ord('b'): 8, ord('f'): 12}
    while i < len(value):
        c = value[i]
        i += 1
        if c != 92:
            out.append(c)
            continue
        c = value[i]
        i += 1
        if 48 <= c <= 55:
            digits = bytes([c])
            while i < len(value) and len(digits) < 3 and 48 <= value[i] <= 55:
                digits += bytes([value[i]])
                i += 1
            out.append(int(digits, 8))
        elif c not in (10, 13):
            out.append(escapes.get(c, c))
    return out


def main():
    data = SOURCE.read_bytes()
    objects = {int(m[1]): m[2] for m in re.finditer(rb'(\d+) 0 obj\s*(.*?)\s*endobj', data, re.S)}

    def stream(number):
        obj = objects[number]
        raw = re.search(rb'stream\r?\n(.*?)\r?\nendstream', obj, re.S)[1]
        return zlib.decompress(raw) if b'/FlateDecode' in obj else raw

    maps = {}
    for number, obj in objects.items():
        match = re.search(rb'/ToUnicode (\d+) 0 R', obj)
        if match:
            cmap = stream(int(match[1]))
            assert b'beginbfrange' not in cmap, 'unsupported range encoding'
            pairs = re.findall(rb'<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>', cmap.split(b'beginbfchar', 1)[1])
            maps[number] = {int(a, 16): bytes.fromhex(b.decode()).decode('utf-16-be') for a, b in pairs}
    pages = []
    for number in (3, 36, 38):
        obj = objects[number]
        fonts = {name: int(ref) for name, ref in re.findall(rb'/(f\d+) (\d+) 0 R', obj)}
        content = stream(int(re.search(rb'/Contents (\d+) 0 R', obj)[1]))
        lines = []
        for block in re.findall(rb'BT\s*(.*?)\s*ET', content, re.S):
            font = re.search(rb'/(f\d+) [\d.]+ Tf', block)[1]
            pieces = []
            for value in re.findall(rb'\(((?:\\.|[^\\)])*)\)', block, re.S):
                decoded = literal(value)
                assert len(decoded) % 2 == 0
                pieces.append(''.join(maps[fonts[font]][int.from_bytes(decoded[i:i+2], 'big')] for i in range(0, len(decoded), 2)))
            lines.append(''.join(pieces))
        pages.append('\n'.join(lines))
    text = '\n\n'.join(f'--- PAGE {i+1} ---\n{value}' for i, value in enumerate(pages))
    (HERE / 'problem_extracted.txt').write_text(text, encoding='utf-8')
    (HERE / 'problem_source.json').write_text(json.dumps({
        'path': str(SOURCE), 'sha256': hashlib.sha256(data).hexdigest(), 'pages': len(pages),
        'method': 'purpose-built Identity-H/bfchar text extraction; relevant page 2 checked against existing rendering',
        'source_is_reference_not_execution_authorization': True,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
