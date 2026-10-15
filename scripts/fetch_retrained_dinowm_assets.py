"""Acquire public third-party frozen weights; do not unpickle or execute them."""
import hashlib,json,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/followup_v1/protocol/dinowm_retrained_inventory'


def getjson(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'research-asset-audit'}),timeout=60) as r:return json.load(r)


def acquire(env):
    repo='fafraob/point-cloud-'+env;meta=getjson('https://huggingface.co/api/models/'+repo);revision=meta['sha'];listing=getjson(f'https://huggingface.co/api/models/{repo}/tree/{revision}/dino-wm?recursive=true&expand=false')
    dest=ROOT/'data/checkpoints/dinowm-retrained-fafraob'/env;dest.mkdir(parents=True,exist_ok=True);files=[]
    for row in listing:
        if row['type']!='file':continue
        relative=Path(row['path']).relative_to('dino-wm');path=dest/relative;path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://huggingface.co/{repo}/resolve/{revision}/{row["path"]}'
        if not path.exists():
            temp=path.with_suffix(path.suffix+'.partial')
            with urllib.request.urlopen(url,timeout=60) as response,temp.open('wb') as f:
                while True:
                    block=response.read(1024*1024)
                    if not block:break
                    f.write(block)
            assert temp.stat().st_size==row['size'];temp.replace(path)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert path.stat().st_size==row['size']
        if 'lfs' in row:assert digest==row['lfs']['oid']
        files.append({'path':str(path.relative_to(ROOT)),'url':url,'bytes':path.stat().st_size,'sha256':digest,'matches_published_lfs':digest==row.get('lfs',{}).get('oid') if 'lfs' in row else None})
    record={'repository':repo,'revision':revision,'files':files,'metadata':meta,'acquired_unix':time.time(),'provenance':'Third-party retraining per officialissue93comment5886831457, not original LeWM baseline weights. No loading or evaluation yet.'}
    (OUT/(env+'_acquired.json')).write_text(json.dumps(record,indent=2)+'\n');print('ACQUIRED',env,len(files),'files',flush=True)


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(acquire,['pusht','ogb-cube']))
    commit=getjson('https://api.github.com/repos/fafraob/point-lewm/commits/main')['sha'];tree=getjson(f'https://api.github.com/repos/fafraob/point-lewm/git/trees/{commit}?recursive=1')
    (OUT/'source_tree.json').write_text(json.dumps({'revision':commit,**tree},indent=2)+'\n')
    for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md']:
        url=f'https://raw.githubusercontent.com/fafraob/point-lewm/{commit}/{name}'
        with urllib.request.urlopen(url,timeout=60) as r:body=r.read()
        (OUT/name).write_bytes(body)
    print('SOURCE_METADATA',commit,flush=True)
if __name__=='__main__':main()
