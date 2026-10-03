"""Download checksum-verified source measurements and rebuild processed arrays."""
from pathlib import Path
import hashlib,sys,subprocess,urllib.request
ROOT=Path(__file__).resolve().parent
SOURCES=[
('beijing.zip','https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip','b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4'),
('intel.txt.gz','https://db.csail.mit.edu/labdata/data.txt.gz','d99288c8f406ca6604d359ceaa0d8adfffa79e7095061a1e27dc4399f48c7225')]
def main():
    raw=ROOT/'data/raw';raw.mkdir(parents=True,exist_ok=True)
    for name,url,expected in SOURCES:
        p=raw/name
        if not p.exists():
            req=urllib.request.Request(url,headers={'User-Agent':'CumulativeTelemetryResearch/1.0'})
            with urllib.request.urlopen(req,timeout=120) as src,p.open('wb') as out:
                while True:
                    b=src.read(1048576)
                    if not b:break
                    out.write(b)
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        if actual!=expected:raise SystemExit('Source checksum mismatch: '+name)
        print('Verified',name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'src/prepare_data.py'),'--raw',str(raw)],cwd=ROOT,check=True)
if __name__=='__main__':main()
