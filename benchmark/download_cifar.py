"""Parallel HTTP range download with official torchvision archive checksum."""
import concurrent.futures
import hashlib
import urllib.request
import time
import subprocess
from .data import DATA

URL='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
SIZE=170498071
MD5='c58f30108f718f92721af3b95e74349a'


def part(i):
    start=SIZE*i//12;end=SIZE*(i+1)//12-1;parent=DATA/'downloads'/'cifar_parts_v3';parent.mkdir(exist_ok=True);path=parent/f'range{i}'
    for retry in range(8):
        current=path.stat().st_size if path.exists() else 0
        if current==end-start+1:return path
        try:
            # Native Windows curl handles this host's TLS chain more reliably than urllib.
            chunkpath=path.parent/(path.name+'.chunk'); headers=path.parent/(path.name+'.headers')
            process=subprocess.run(['curl.exe','--fail','-sS','-L','--retry','3','--connect-timeout','30','--max-time','900','--range',f'{start+current}-{end}','-D',str(headers),'-o',str(chunkpath),URL],capture_output=True,text=True)
            if process.returncode:raise IOError(process.stderr)
            if not chunkpath.exists() or chunkpath.stat().st_size!=end-start+1-current:raise IOError('Range response size mismatch')
            with path.open('ab') as f, chunkpath.open('rb') as src:
                while chunk:=src.read(1024*1024):f.write(chunk)
            print('range complete',i,flush=True);return path
        except Exception as exc:print('range retry',i,retry,repr(exc),flush=True);time.sleep(2)
    raise RuntimeError(f'Range {i} failed repeatedly')


if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:paths=list(pool.map(part,range(12)))
    target=DATA/'downloads'/'cifar_verified.tar.gz'
    with target.open('wb') as f:
        for p in paths:
            with p.open('rb') as src:
                while chunk:=src.read(1024*1024):f.write(chunk)
    with target.open('rb') as f:actual=hashlib.file_digest(f,'md5').hexdigest()
    assert actual==MD5,(actual,MD5)
    print('Verified',target,flush=True)
