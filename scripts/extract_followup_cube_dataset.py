"""Verify official compressed SHA256 while safely streaming the Cube archive."""
import hashlib,io,os,shutil,sys,tarfile,time
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import zstandard as zstd
from zj_wam.io import resolve,write_json
SOURCE=resolve('data/downloads/lewm-cube-expert-http/cube_single_expert.tar.zst')
DEST=resolve('data/datasets/lewm-cube');DEST.mkdir(parents=True,exist_ok=True)
EXPECTED_SIZE=46184624478
EXPECTED_SHA='3725d6a01abd492164441ef0a27e588f52b94a118fab56b96987b1a34a6c2600'
assert SOURCE.stat().st_size==EXPECTED_SIZE
class HashReader(io.RawIOBase):
    def __init__(self,f):self.f=f;self.digest=hashlib.sha256();self.count=0
    def readable(self):return True
    def read(self,n=-1):
        value=self.f.read(n);self.digest.update(value);self.count+=len(value);return value
    def readinto(self,b):
        value=self.read(len(b));b[:len(value)]=value;return len(value)
start=time.perf_counter();records=[];pending=[]
with SOURCE.open('rb',buffering=8*1024*1024) as f:
    hashed=HashReader(f)
    with zstd.ZstdDecompressor().stream_reader(hashed,closefd=False) as z:
        with tarfile.open(fileobj=z,mode='r|') as archive:
            for member in archive:
                name=PurePosixPath(member.name)
                assert not name.is_absolute() and '..' not in name.parts and member.isfile(),(member.name,member.type)
                target=DEST.joinpath(*name.parts).resolve();assert target.is_relative_to(DEST.resolve());assert target.suffix=='.h5',member.name
                partial=target.with_suffix(target.suffix+'.partial');assert not target.exists() and not partial.exists(),target
                assert member.size<shutil.disk_usage(DEST).free-10*1024**3
                target.parent.mkdir(parents=True,exist_ok=True);copied=0;last=0
                with archive.extractfile(member) as source,partial.open('xb') as output:
                    while True:
                        chunk=source.read(8*1024*1024)
                        if not chunk:break
                        output.write(chunk);copied+=len(chunk)
                        if copied-last>=10*1024**3:print(member.name,copied,'/',member.size,flush=True);last=copied
                    output.flush();os.fsync(output.fileno())
                assert copied==member.size;records.append({'member':member.name,'bytes':copied,'target':str(target)});pending.append((partial,target))
        while z.read(8*1024*1024):pass
    while hashed.read(8*1024*1024):pass
    assert hashed.count==EXPECTED_SIZE and hashed.digest.hexdigest()==EXPECTED_SHA,(hashed.count,hashed.digest.hexdigest())
for partial,target in pending:partial.rename(target)
write_json(resolve('artifacts/followup_v1/protocol/cube_archive_audit.json'),{'repo':'quentinll/lewm-cube','revision':'02a19a67a0dc8c9d6215f89c19e0a597691e152a','archive':str(SOURCE),'compressed_bytes':EXPECTED_SIZE,'compressed_sha256':EXPECTED_SHA,'official_lfs_hash_match':True,'members':records,'seconds':time.perf_counter()-start,'safe_extraction':'Only regular .h5 members; relative traversal-free paths; no overwrite or links; size/disk checks; compressedSHA256 verified before final filenames exposed.'})
print('COMPLETE',records,'seconds',time.perf_counter()-start,flush=True)
