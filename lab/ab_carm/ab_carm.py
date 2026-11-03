import re,subprocess,os,sys,json,time
from pathlib import Path
sys.path.insert(0,'/tmp/cl/m/scripts')
from matrix import strip_ansi
os.chdir('/tmp/cl/m')
env0=dict(os.environ)
for k in ("MATSURI_CAMP_SEARCH_N","MATSURI_NORMAL_SEARCH_N","MATSURI_SEARCH_N","MATSURI_SEARCH_RADICAL","MATSURI_SEARCH_TOPK"): env0.pop(k,None)
envS=dict(env0); envS.update({"MATSURI_SEARCH_N":"4096","MATSURI_SEARCH_TOPK":"4","MATSURI_SEARCH_RADICAL":"1.4"})

BASE=Path('data/matsuri_config.toml').read_text()
assert 'route = 0.0' in BASE  # champion base already set to route=0 for this run
R500=BASE.replace('route = 0.0','route = 500.0')

ARMS=[
 ('A0', R500, env0),   # 先验, 无搜索
 ('B1', R500, envS),   # 先验 + 4096/rad1.4 (复现 §8.3)
 ('C0', BASE, env0),   # 无先验, 无搜索
 ('C1', BASE, envS),   # 无先验 + 4096/rad1.4 (补跑作废的 C 档)
]
ACT=re.compile(r'演目Lv \[([0-9,\s]+)\]（上限')
out={t:[] for t,_,_ in ARMS}
t0=time.time()
for seed in range(62700,62720):
    Path('ai_config.toml').write_text(f"[sim]\nseed={seed}\ncards=[303174,303114,303004,303224,303204,303234]\numa=100101\n")
    for tag,txt,env in ARMS:
        Path('data/matsuri_config.toml').write_text(txt)
        p=subprocess.run(['target/release/matsuri','matsuri'],capture_output=True,text=True,errors='replace',timeout=900,env=env)
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
        out[tag].append(row)
    print(seed,'done',round(time.time()-t0,1),'s',flush=True)
Path('data/matsuri_config.toml').write_text(BASE)
json.dump(out,open('/tmp/carm_raw.json','w'))
print('ALL DONE',round(time.time()-t0,1),'s')
