#!/usr/bin/env python3
"""骏大祭「减半」验证分析：把模拟器非减半评分换算回实机面板评分，做配对比较。

模拟器五维是**非减半**累积（raw），实机面板 = 1200+(raw-1200)/2（raw>1200）。
评分表 five_status_final_score 是 raw 索引的凸表：
  sim评分  = skill + PT*pt_rate + Σ table[raw_axis]
  real评分 = skill + PT*pt_rate + Σ table[panel_axis]   （实机口径，换算回来）
skill 用 sim评分 反解（PT/技能不受属性减半影响，两口径一致）。

用法：analyze_halve.py base.csv arm1.csv [arm2.csv ...]
以第一个 CSV 为基线，逐臂配对（同 seed）算 Δ 与 t。
"""
from __future__ import annotations
import csv, json, math, os, sys
from pathlib import Path

C = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'constants.json')))
T = C['five_status_final_score']
PT_RATE = C['pt_score_rate']
AXES = ('speed', 'stamina', 'power', 'guts', 'wisdom')


def panel(raw: float) -> int:
    return int(round(raw if raw <= 1200 else 1200 + (raw - 1200) / 2))


def tbl(v: int) -> float:
    return float(T[min(max(v, 0), len(T) - 1)])


def load(csv_path: str) -> dict[int, dict]:
    """seed -> {sim, real, pt, raw[5], panel[5], five_raw, five_pan}"""
    out: dict[int, dict] = {}
    with open(csv_path, newline='', encoding='utf8') as fh:
        for r in csv.DictReader(fh):
            if not str(r.get('score', '')).isdigit():
                continue
            seed = int(r['seed'])
            sim = int(r['score'])
            pt = int(r['pt']) if str(r.get('pt', '')).lstrip('-').isdigit() else 0
            raw = [int(r[a]) for a in AXES if str(r.get(a, '')).lstrip('-').isdigit()]
            if len(raw) != 5:
                continue
            sum_raw = sum(tbl(v) for v in raw)
            skill = sim - pt * PT_RATE - sum_raw          # 反解技能分
            pan = [panel(v) for v in raw]
            sum_pan = sum(tbl(v) for v in pan)
            real = skill + pt * PT_RATE + sum_pan
            out[seed] = dict(sim=sim, real=real, pt=pt, raw=raw, panel=pan,
                             five_raw=sum(raw), five_pan=sum(pan), skill=skill)
    return out


def paired(base: dict, arm: dict, key: str):
    seeds = sorted(set(base) & set(arm))
    d = [arm[s][key] - base[s][key] for s in seeds]
    n = len(d)
    if n < 2:
        return dict(n=n)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    t = mean / (sd / math.sqrt(n)) if sd > 0 else float('inf')
    wins = sum(1 for x in d if x > 0)
    return dict(n=n, mean=round(mean, 1), sd=round(sd, 1), t=round(t, 2),
                wins=f"{wins}/{n}", min=min(d), max=max(d))


def mean_of(d: dict, key: str):
    vals = [v[key] for v in d.values()]
    return round(sum(vals) / len(vals), 1) if vals else None


def main() -> int:
    paths = sys.argv[1:]
    if len(paths) < 2:
        print(__doc__); return 2
    arms = [load(p) for p in paths]
    names = [Path(p).parent.name for p in paths]
    base = arms[0]
    print(f"基线: {names[0]}  ({len(base)} 局)")
    print(f"  sim评分 mean {mean_of(base,'sim')}   real(面板)评分 mean {mean_of(base,'real')}"
          f"   五维raw {mean_of(base,'five_raw')}  五维panel {mean_of(base,'five_pan')}  PT {mean_of(base,'pt')}")
    for name, arm in zip(names[1:], arms[1:]):
        print(f"\n臂 {name}  ({len(arm)} 局, 配对 {len(set(base)&set(arm))})")
        print(f"  sim评分 mean {mean_of(arm,'sim')}   real(面板)评分 mean {mean_of(arm,'real')}"
              f"   五维raw {mean_of(arm,'five_raw')}  五维panel {mean_of(arm,'five_pan')}  PT {mean_of(arm,'pt')}")
        for key, label in (('sim', 'sim非减半评分'), ('real', 'real面板评分'),
                           ('five_pan', '五维panel合计'), ('pt', 'PT')):
            st = paired(base, arm, key)
            if st.get('n', 0) >= 2:
                print(f"    Δ{label:12s} mean {st['mean']:+9.1f}  t={st['t']:+.2f}  "
                      f"胜 {st['wins']}  区间[{st['min']:+.0f},{st['max']:+.0f}]")
    # 五维分列（real 面板口径）均值
    print("\n五维分列（面板 panel 均值）:")
    for name, arm in zip(names, arms):
        cols = []
        for i, a in enumerate(AXES):
            vals = [v['panel'][i] for v in arm.values()]
            cols.append(f"{a[:2]}{round(sum(vals)/len(vals))}" if vals else f"{a[:2]}-")
        print(f"  {name:24s} " + "  ".join(cols))
    return 0


if __name__ == '__main__':
    sys.exit(main())
