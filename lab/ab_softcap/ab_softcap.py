import re,subprocess,os,sys,json,time
from pathlib import Path
sys.path.insert(0,'/tmp/cl/m/scripts')
from matrix import strip_ansi
os.chdir('/tmp/cl/m')
env0=dict(os.environ)
for k in ("MATSURI_CAMP_SEARCH_N","MATSURI_NORMAL_SEARCH_N","MATSURI_SEARCH_N"): env0.pop(k,None)

BASE=Path('data/matsuri_config.toml').read_text()
assert 'status_soft_reserve' not in BASE

def with_soft(txt, factor):
    lines = txt.splitlines()
    idx = max(i for i,l in enumerate(lines) if l.strip()=='[trainer]')
    add = ['status_soft_reserve = true', f'reserve_status_factor = {factor}', 'final_bonus = 0']
    lines[idx+1:idx+1] = add
    return '\n'.join(lines)+'\n'

ARMS={
 'off':  BASE,
 'r40':  with_soft(BASE,'40.0'),
 'r20':  with_soft(BASE,'20.0'),
 'r60':  with_soft(BASE,'60.0'),
}
# appended keys land in the LAST toml table -> must be [trainer]; verify
import tomllib
for tag,txt in ARMS.items():
    if tag=='off': continue
    d=tomllib.loads(txt)
    assert d['trainer'].get('status_soft_reserve') is True and d['trainer'].get('reserve_status_factor')==float(tag[1:]), (tag, d['trainer'].get('status_soft_reserve'), d['trainer'].get('reserve_status_factor'))
print('arm configs verified (keys in [trainer])',flush=True)

ACT=re.compile(r'演目Lv \[([0-9,\s]+)\]（上限')
out={t:[] for t in ARMS}
SEEDS=range(61444,61469)  # 25 seeds x 4 arms = 100 games
t0=time.time()
for seed in SEEDS:
    Path('ai_config.toml').write_text(f"[sim]\nseed={seed}\ncards=[303174,303114,303004,303224,303204,303234]\numa=100101\n")
    for tag,txt in ARMS.items():
        Path('data/matsuri_config.toml').write_text(txt)
        p=subprocess.run(['target/release/matsuri','matsuri'],capture_output=True,text=True,errors='replace',timeout=600,env=env0)
        o,e=strip_ansi(p.stdout),strip_ansi(p.stderr)
        row={'seed':seed}
        m=re.search(r'评分\s+(\d+)\s+(\S+)',o)
        if m: row['score'],row['rank']=int(m.group(1)),m.group(2)
        m=re.search(r'^PT\s+(\d+)',o,re.M)
        if m: row['pt']=int(m.group(1))
        m2=re.findall(r'速 (\d+) 耐 (\d+) 力 (\d+) 根 (\d+) 智 (\d+)',o)
        if m2:
            v=[int(x) for x in m2[-1]]
            row['attrs']=v
        m=ACT.search(e)
        if m: row['act_levels']=[int(x) for x in m.group(1).replace(' ','').split(',') if x]
        out[tag].append(row)
    print(seed,'done',round(time.time()-t0,1),'s',flush=True)
Path('data/matsuri_config.toml').write_text(BASE)
json.dump(out,open('/tmp/softcap_raw.json','w'))
print('ALL DONE',round(time.time()-t0,1),'s')
