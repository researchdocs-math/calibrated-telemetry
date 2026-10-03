"""Verify numerical outputs against the recorded reproducibility manifest."""
from pathlib import Path
import hashlib,gzip,json,numpy as np
ROOT=Path(__file__).resolve().parent
def fingerprint(path):
    suffix=path.suffix
    if suffix=='.gz':
        with gzip.open(path,'rb') as f:
            h=hashlib.sha256()
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
            return h.hexdigest()
    if suffix=='.json':
        return hashlib.sha256(json.dumps(json.loads(path.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if suffix=='.npz':
        h=hashlib.sha256()
        with np.load(path,allow_pickle=False) as z:
            for key in sorted(z.files):
                arr=z[key]
                h.update(key.encode());h.update(str(arr.dtype).encode())
                h.update(str(arr.shape).encode());h.update(arr.tobytes(order='C'))
        return h.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'reproduction_manifest.json').read_text())
    bad=[]
    for relative,expected in manifest['files'].items():
        p=ROOT/relative
        actual=fingerprint(p) if p.exists() else None
        if actual!=expected:bad.append(relative)
    if bad:
        raise SystemExit('Fingerprint mismatch: '+', '.join(bad))
    print('PASS:',len(manifest['files']),'data and numerical-output fingerprints match.')
if __name__=='__main__':main()
