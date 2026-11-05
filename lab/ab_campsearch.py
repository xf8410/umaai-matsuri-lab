#!/usr/bin/env python3
"""攻略优化点 3（夏合宿 4 轮组合前瞻）· CRN A/B 验证

合宿前瞻由环境变量 MATSURI_CAMP_SEARCH_N 开关（0=逐回合贪心，>0=FlatSearch MC rollout 到终局），
不动配置文件 → 无还原风险。同一批种子跑多臂，CRN 配对 → 配对 t 检验。

采集：育成评分 / 档次 / 五维分列+合计 / PT / 演目Lv（速耐力根智）/ 合宿前瞻触发回合数。
合宿窗口 = T12-15 / T36-39 / T60-63（共 12 回合，第一窗在 T12 → 改动会级联全局）。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from matrix import strip_ansi, five_status  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "target/release/matsuri"
OVERLAY = REPO / "ai_config.toml"

ACT = ["速", "耐", "力", "根", "智"]
SCORE_RE = re.compile(r"^评分\s+(\d+)\s+(\S+)")
PT_RE = re.compile(r"^PT\s+(\d+)")
ACTLV_RE = re.compile(r"演目Lv \[([0-9,\s]+)\]（上限 (\d+)）")
CAMP_RE = re.compile(r"【搜索】T\d+ 合宿前瞻")
DECIDE_RE = re.compile(r"【决策】T\d+ 动作 (\S+)")


def run_one(seed: int, cards: str, uma: str, camp_n: int, topk: int) -> dict:
    OVERLAY.write_text(
        f"[sim]\nseed = {seed}\ncards = [{cards}]\numa = {uma}\n", encoding="utf8")
    env = {"MATSURI_CAMP_SEARCH_N": str(camp_n), "MATSURI_CAMP_SEARCH_TOPK": str(topk)}
    p = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                       text=True, errors="replace", timeout=600,
                       env={**__import__("os").environ, **env})
    out, err = strip_ansi(p.stdout), strip_ansi(p.stderr)
    row: dict = {"seed": seed, "ok": 0}
    for ln in out.splitlines():
        if m := SCORE_RE.match(ln.strip()):
            row["score"], row["rank"] = int(m.group(1)), m.group(2)
        elif m := PT_RE.match(ln.strip()):
            row["pt"] = int(m.group(1))
    st = five_status(out)
    if st:
        for k, v in zip(("speed", "stamina", "power", "guts", "wisdom"), st):
            row[k] = v
        row["five_sum"] = sum(st)
    if m := ACTLV_RE.search(err):
        row["act_levels"] = [int(x) for x in m.group(1).replace(" ", "").split(",") if x]
    row["camp_fired"] = len(CAMP_RE.findall(err))
    row["decisions"] = DECIDE_RE.findall(err)
    row["ok"] = int("score" in row and st is not None and "act_levels" in row)
    return row


def paired(ctrl, treat, name):
    c, t = np.array(ctrl, float), np.array(treat, float)
    d = t - c
    tstat, p = stats.ttest_rel(t, c)
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return (f"  {name:<10} 对照 {c.mean():8.1f} → 处理 {t.mean():8.1f}  "
            f"Δ{d.mean():+7.1f} ± {d.std(ddof=1)/np.sqrt(len(d)):5.1f}  "
            f"t={tstat:+6.2f} p={p:.4f} {star}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--arms", default="0,48,192", help="MATSURI_CAMP_SEARCH_N 各臂")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cards", default="303174, 303114, 303004, 303224, 303204, 303234")
    ap.add_argument("--uma", default="100101")
    ap.add_argument("--out", default="lab/ab_campsearch")
    args = ap.parse_args()

    if int(args.uma) in (106301, 106302):
        print("禁用马，拒跑", file=sys.stderr); return 2
    arms = [int(x) for x in args.arms.split(",")]
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    if len(seeds) > 100:
        print("每臂 >100 局，违反硬上限", file=sys.stderr); return 2

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    results: dict[int, list[dict]] = {}
    try:
        for n in arms:
            print(f"\n=== 臂 camp_search_n={n} · {len(seeds)} 局 ===", file=sys.stderr)
            rows = []
            for i, s in enumerate(seeds):
                rows.append(run_one(s, args.cards, args.uma, n, args.topk))
                if (i + 1) % 10 == 0:
                    print(f"  {i+1}/{len(seeds)}", file=sys.stderr)
            results[n] = rows
            ok = [r for r in rows if r["ok"]]
            sc = np.array([r["score"] for r in ok], float)
            print(f"  解析 {len(ok)}/{len(rows)}  评分 mean {sc.mean():.0f} "
                  f"[{sc.min():.0f},{sc.max():.0f}]", file=sys.stderr)
    finally:
        OVERLAY.unlink(missing_ok=True)

    ctrl = arms[0]
    cc = [r for r in results[ctrl] if r["ok"]]
    lines = ["# 攻略点3 合宿前瞻 camp_search · CRN A/B\n",
             f"对照臂 = camp_search_n={ctrl}（贪心）；每臂 {len(cc)} 局，"
             f"seeds {seeds[0]}-{seeds[-1]}；topk={args.topk}；uma={args.uma}\n"]

    def col(rows, k):
        return [r[k] for r in rows if k in r]

    lines.append(f"\n## 对照臂（贪心 camp_search_n=0）基线")
    for k, lbl in [("score", "育成评分"), ("five_sum", "五维合计"), ("pt", "PT")]:
        v = np.array(col(cc, k), float)
        lines.append(f"- {lbl}: {v.mean():.1f} ± {v.std(ddof=1):.1f}  "
                     f"[{v.min():.0f},{v.max():.0f}]")
    ranks = Counter(str(r.get("rank", "?")) for r in cc)
    lines.append(f"- 档次分布: {dict(ranks.most_common())}")
    for j, a in enumerate(ACT):
        v = np.array([r["act_levels"][j] for r in cc if "act_levels" in r], float)
        lines.append(f"- 演目Lv {a}: {v.mean():.2f}")
    for j, a in enumerate(ACT):
        v = np.array(col(cc, ("speed", "stamina", "power", "guts", "wisdom")[j]), float)
        lines.append(f"- 属性 {a}: {v.mean():.0f}")

    for n in arms[1:]:
        tt = [r for r in results[n] if r["ok"]]
        idx = {r["seed"]: r for r in tt}
        pairs = [(c, idx[c["seed"]]) for c in cc if c["seed"] in idx]
        # 决策改动量：每局有多少回合动作不同
        changed = [sum(a != b for a, b in zip(c["decisions"], t["decisions"]))
                   for c, t in pairs]
        rank_up = sum(1 for c, t in pairs
                      if str(t.get("rank", "")) != str(c.get("rank", "")))
        lines.append(f"\n## 处理臂 camp_search_n={n} vs 贪心（配对 n={len(pairs)}）")
        lines.append(f"  平均每局改动决策回合数: {np.mean(changed):.1f}；档次变化局数: {rank_up}")
        for k, lbl in [("score", "育成评分"), ("five_sum", "五维合计"), ("pt", "PT"),
                       ("speed", "速"), ("stamina", "耐"), ("power", "力"),
                       ("guts", "根"), ("wisdom", "智")]:
            lines.append(paired([c[k] for c, _ in pairs], [t[k] for _, t in pairs], lbl))
        tr = Counter(str(t.get("rank", "?")) for _, t in pairs)
        lines.append(f"  处理臂档次分布: {dict(tr.most_common())}")

    report = "\n".join(lines)
    (out / "REPORT.md").write_text(report + "\n", encoding="utf8")
    slim = {str(k): [{kk: vv for kk, vv in r.items() if kk != "decisions"}
                     for r in v] for k, v in results.items()}
    (out / "raw.json").write_text(json.dumps(slim, ensure_ascii=False), encoding="utf8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
