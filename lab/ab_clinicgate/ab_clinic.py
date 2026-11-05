import re,subprocess,os,sys,json,time
from pathlib import Path
sys.path.insert(0,'/tmp/cl/m/scripts')
from matrix import strip_ansi
os.chdir('/tmp/cl/m2')
env0=dict(os.environ)
for k in ("MATSURI_CAMP_SEARCH_N","MATSURI_NORMAL_SEARCH_N","MATSURI_SEARCH_N","MATSURI_SEARCH_RADICAL","MATSURI_SEARCH_TOPK","MATSURI_CLINIC_DEFER_CAMP"): env0.pop(k,None)
envD=dict(env0); envD["MATSURI_CLINIC_DEFER_CAMP"]="1"

BASE_BIN='/tmp/matsuri_base_bin'
NEW_BIN='/tmp/td_clinic/release/matsuri'
# champion base with route=500 (restore)
BASE=Path('/tmp/cl/m2/lab/config/ga-r2.toml').read_text()
assert 'route = 500' in BASE
Path('data/matsuri_config.toml').write_text(BASE)

# sanity: new binary without env == base binary on 2 seeds
for s in (61444,61445):
    Path('ai_config.toml').write_text(f"[sim]\nseed={s}\ncards=[303174,303114,303004,303224,303204,303234]\numa=100101\n")
    a=subprocess.run([BASE_BIN,'matsuri'],capture_output=True,text=True,errors='replace',env=env0)
    b=subprocess.run([NEW_BIN,'matsuri'],capture_output=True,text=True,errors='replace',env=env0)
    sa=re.search(r'评分\s+(\d+)',strip_ansi(a.stdout)); sb=re.search(r'评分\s+(\d+)',strip_ansi(b.stdout))
    assert sa and sb and sa.group(1)==sb.group(1), (s, sa and sa.group(1), sb and sb.group(1))
print('sanity OK: new bin == base bin without env',flush=True)

ACT=re.compile(r'演目Lv \[([0-9,\s]+)\]（上限')
out={'off':[],'defer':[]}
t0=time.time()
for seed in range(61444,61494):
    Path('ai_config.toml').write_text(f"[sim]\nseed={seed}\ncards=[303174,303114,303004,303224,303204,303234]\numa=100101\n")
    for tag,binary,env in (('off',BASE_BIN,env0),('defer',NEW_BIN,envD)):
        p=subprocess.run([binary,'matsuri'],capture_output=True,text=True,errors='replace',timeout=600,env=env)
        o,e=strip_ansi(p.stdout),strip_ansi(p.stderr)
        row={'seed':seed}
        m=re.search(r'评分\s+(\d+)\s+(\S+)',o)
        if m: row['score'],row['rank']=int(m.group(1)),m.group(2)
        m=re.search(r'^PT\s+(\d+)',o,re.M)
        if m: row['pt']=int(m.group(1))
        m2=re.findall(r'速 (\d+) 耐 (\d+) 力 (\d+) 根 (\d+) 智 (\d+)',o)
        if m2: row['attrs']=[int(x) for x in m2[-1]]
        m=ACT.search(e)
        if m: row['act_levels']=[int(x) for x in m.group(1).replace(' ','').split(',') if x]
        row['clinic_fires']=len(re.findall(r'守门·治病优先',e))
        out[tag].append(row)
    print(seed,'done',round(time.time()-t0,1),'s',flush=True)
json.dump(out,open('/tmp/clinic_raw.json','w'))
print('ALL DONE',round(time.time()-t0,1),'s')
