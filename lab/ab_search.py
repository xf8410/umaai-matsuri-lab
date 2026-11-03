#!/usr/bin/env python3
"""攻略优化点 1/3（前瞻搜索）· CRN A/B 验证（通用：合宿 / 常规 / 两者）

前瞻搜索由环境变量开关（0=逐回合贪心，>0=FlatSearch MC rollout 到终局），不动配置文件。
--mode camp   → MATSURI_CAMP_SEARCH_N（点3：合宿训练回合）
--mode normal → MATSURI_NORMAL_SEARCH_N（点1：常规回合稽古洗牌前瞻）
--mode both   → 两者同开
同一批种子跑多臂，CRN 配对 → 配对 t 检验。
"""
from __future__ import annotations

import argparse
import json
import os
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
SEARCH_RE = re.compile(r"【搜索】T\d+ (合宿前瞻|常规前瞻)")
DECIDE_RE = re.compile(r"【决策】T\d+ 动作 (\S+)")

MODE_ENV = {
    "camp": ("MATSURI_CAMP_SEARCH_N", "MATSURI_CAMP_SEARCH_TOPK"),
    "normal": ("MATSURI_NORMAL_SEARCH_N", "MATSURI_NORMAL_SEARCH_TOPK"),
}


def run_one(seed, cards, uma, n, topk, mode):
    OVERLAY.write_text(
        f"[sim]\nseed = {seed}\ncards = [{cards}]\numa = {uma}\n", encoding="utf8")
    env = dict(os.environ)
    if mode == "both":
        env.update({"MATSURI_CAMP_SEARCH_N": str(n), "MATSURI_CAMP_SEARCH_TOPK": str(topk),
                    "MATSURI_NORMAL_SEARCH_N": str(n), "MATSURI_NORMAL_SEARCH_TOPK": str(topk)})
    else:
        nvar, tvar = MODE_ENV[mode]
        env.update({nvar: str(n), tvar: str(topk)})
    p = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                       text=True, errors="replace", timeout=900, env=env)
    out, err = strip_ansi(p.stdout), strip_ansi(p.stderr)
    row = {"seed": seed, "ok": 0}
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
    row["search_fired"] = len(SEARCH_RE.findall(err))
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
    ap.add_argument("--mode", default="normal", choices=["camp", "normal", "both"])
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--arms", default="0,64,192")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--cards", default="303174, 303114, 303004, 303224, 303204, 303234")
    ap.add_argument("--uma", default="100101")
    ap.add_argument("--out", default="lab/ab_search")
    args = ap.parse_args()

    if int(args.uma) in (106301, 106302):
        print("禁用马，拒跑", file=sys.stderr); return 2
    arms = [int(x) for x in args.arms.split(",")]
    seeds = list(range(args.start_seed, args.start_seed + args.seeds))
    if len(seeds) > 100:
        print("每臂 >100 局，违反硬上限", file=sys.stderr); return 2

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    try:
        for n in arms:
            print(f"\n=== 臂 {args.mode} search_n={n} · {len(seeds)} 局 ===", file=sys.stderr)
            rows = []
            for i, s in enumerate(seeds):
                rows.append(run_one(s, args.cards, args.uma, n, args.topk, args.mode))
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
    lines = [f"# 攻略点前瞻搜索 · CRN A/B（mode={args.mode}）\n",
             f"对照臂 = search_n={ctrl}（贪心）；每臂 {len(cc)} 局，"
             f"seeds {seeds[0]}-{seeds[-1]}；topk={args.topk}；uma={args.uma}\n"]

    def col(rows, k):
        return [r[k] for r in rows if k in r]

    lines.append("\n## 对照臂（贪心）基线")
    for k, lbl in [("score", "育成评分"), ("five_sum", "五维合计"), ("pt", "PT")]:
        v = np.array(col(cc, k), float)
        lines.append(f"- {lbl}: {v.mean():.1f} ± {v.std(ddof=1):.1f}  [{v.min():.0f},{v.max():.0f}]")
    lines.append(f"- 档次分布: {dict(Counter(str(r.get('rank','?')) for r in cc).most_common())}")
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
        changed = [sum(a != b for a, b in zip(c["decisions"], t["decisions"])) for c, t in pairs]
        rank_chg = sum(1 for c, t in pairs if str(t.get("rank", "")) != str(c.get("rank", "")))
        fired = np.mean([t["search_fired"] for _, t in pairs])
        lines.append(f"\n## 处理臂 search_n={n} vs 贪心（配对 n={len(pairs)}）")
        lines.append(f"  平均改动决策回合数 {np.mean(changed):.1f}；档次变化 {rank_chg} 局；前瞻触发 {fired:.1f} 回合/局")
        for k, lbl in [("score", "育成评分"), ("five_sum", "五维合计"), ("pt", "PT"),
                       ("speed", "速"), ("stamina", "耐"), ("power", "力"),
                       ("guts", "根"), ("wisdom", "智")]:
            lines.append(paired([c[k] for c, _ in pairs], [t[k] for _, t in pairs], lbl))
        tl = np.array([t["act_levels"] for _, t in pairs if "act_levels" in t], float)
        if len(tl):
            lines.append("  处理臂演目Lv " + " ".join(f"{a}{tl[:,j].mean():.2f}" for j, a in enumerate(ACT)))
        lines.append(f"  处理臂档次分布: {dict(Counter(str(t.get('rank','?')) for _,t in pairs).most_common())}")

    report = "\n".join(lines)
    (out / "REPORT.md").write_text(report + "\n", encoding="utf8")
    slim = {str(k): [{kk: vv for kk, vv in r.items() if kk != "decisions"} for r in v]
            for k, v in results.items()}
    (out / "raw.json").write_text(json.dumps(slim, ensure_ascii=False), encoding="utf8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
