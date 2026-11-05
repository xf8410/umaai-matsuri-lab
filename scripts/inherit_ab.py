#!/usr/bin/env python3
"""继承补接移植 A/B：机制 × 种马因子 四臂配对 CRN，共 100 局（≤100 硬上限）。

上游 3bb8cb2 给事件 4012（T30）/4013（T54）加了特殊结算（五维 += 蓝因子×6 +
round(剧本因子×scale)，PT += round(extra[PT]×(0.5+0.5×scale))，上限 += floor(蓝因子×5.33)）；
本仓此前这两个事件结算成「无变化」，`extra_count` 完全失效。本脚本把两个变量拆开记账：

  pre-labfac  移植前二进制 + 本仓因子 blue[15,3,0,0,0] extra[0,30,0,0,30,30]（= 现生产口径）
  pre-upfac   移植前二进制 + 上游因子 blue[18,0,0,0,0] extra[30,30,60,80,80,100]
  post-labfac 移植后二进制 + 本仓因子   → 纯机制增益
  post-upfac  移植后二进制 + 上游因子   → 机制 + 因子（对齐上游 81553 口径）

用法：python3 scripts/inherit_ab.py [--seeds 61444-61468] [--smoke]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POST = REPO / "target/release/matsuri"
PRE = Path("/tmp/matsuri-pre-port")
BASE = REPO / "lab/config/ga-r2.toml"
UMA = "100101"
CARDS = [303174, 303114, 303004, 303224, 303204, 303234]
FACTORS = {
    "labfac": ([15, 3, 0, 0, 0], [0, 30, 0, 0, 30, 30]),
    "upfac": ([18, 0, 0, 0, 0], [30, 30, 60, 80, 80, 100]),
}
ARMS = [("pre-labfac", PRE, "labfac"), ("pre-upfac", PRE, "upfac"),
        ("post-labfac", POST, "labfac"), ("post-upfac", POST, "upfac")]

SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
RANK_RE = re.compile(r"^评分\s+\d+\s+(\S+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")
INH_RE = re.compile(r"【继承】T(\d+) #\d+")


def play(binary: Path, fac: str, seed: int) -> dict:
    blue, extra = FACTORS[fac]
    (REPO / "matsuri_config.toml").write_text(BASE.read_text(encoding="utf8"), encoding="utf8")
    (REPO / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {UMA}\ncards = [{', '.join(map(str, CARDS))}]\n"
        f"blue_count = [{', '.join(map(str, blue))}]\n"
        f"extra_count = [{', '.join(map(str, extra))}]\n", encoding="utf8")
    env = {**os.environ, "MATSURI_NORMAL_SEARCH_N": "128", "MATSURI_NORMAL_SEARCH_TOPK": "4"}
    t0 = time.time()
    try:
        p = subprocess.run([str(binary), "matsuri"], cwd=REPO, capture_output=True,
                           text=True, errors="replace", timeout=900, env=env)
    except subprocess.TimeoutExpired:
        print(f"  seed {seed} 超时 900s，重试一次", flush=True)
        t0 = time.time()
        p = subprocess.run([str(binary), "matsuri"], cwd=REPO, capture_output=True,
                           text=True, errors="replace", timeout=1800, env=env)
    out, err = p.stdout, p.stderr
    rec = {"seed": seed, "secs": round(time.time() - t0, 1)}
    m = SCORE_RE.search(out)
    if not m:
        rec["score"] = 0
        rec["err"] = (out[-300:] + err[-300:]).strip()
        return rec
    five, lv, rank = FIVE_RE.search(out), LV_RE.search(err), RANK_RE.search(out)
    rec["score"] = int(m.group(1))
    rec["rank"] = rank.group(1) if rank else ""
    rec["pt"] = int(PT_RE.search(out).group(1)) if PT_RE.search(out) else 0
    rec["five"] = [int(x) for x in five.groups()] if five else []
    rec["lv"] = [int(x) for x in lv.group(1).split(",")] if lv else []
    # 继承结算是否真的落地（移植后应看到 T30/T54 两条【继承】）
    rec["inherit_turns"] = [int(x) for x in INH_RE.findall(err)]
    return rec


def mean(xs):
    return round(sum(xs) / len(xs), 1) if xs else 0.0


def paired(a, b):
    d = [y["score"] - x["score"] for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    sd = (sum((x - mu) ** 2 for x in d) / max(n - 1, 1)) ** 0.5
    return {"delta": round(mu, 1), "t": round(mu / (sd / n ** 0.5), 2) if sd > 0 else 0.0,
            "wins": f"{sum(1 for x in d if x > 0)}/{n}"}


# 四臂（机制 × 因子 2×2）时的可读名；其余组合退化成 "a→b"
PAIR_LABELS = {
    ("pre-labfac", "post-labfac"): "机制增益(pre-labfac→post-labfac)",
    ("pre-labfac", "pre-upfac"): "因子增益(pre-labfac→pre-upfac)",
    ("pre-labfac", "post-upfac"): "机制+因子(pre-labfac→post-upfac)",
    ("post-labfac", "post-upfac"): "机制在上游因子下(post-upfac−post-labfac)",
}


def all_pairs(runs: dict, tags: list[str]) -> dict:
    """按 ARMS 实际存在的臂生成全部两两配对，不写死臂名。

    此前把四个臂名硬编码在 paired 调用里，换 ARMS（例如只跑 labfac/upfac 两臂）
    会在跑完 40 局之后 KeyError 崩掉、丢掉整份汇总。
    """
    out = {}
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            if a not in runs or b not in runs:
                continue
            out[PAIR_LABELS.get((a, b), f"{a}→{b}")] = paired(runs[a], runs[b])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="61444-61468")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/tmp/inherit_ab_results.json")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if args.smoke:
        seeds = seeds[:1]
    for tag, binary, _ in ARMS:
        if not binary.exists():
            print(f"缺少二进制 {tag}: {binary}", file=sys.stderr)
            return 2

    runs = {tag: [] for tag, _, _ in ARMS}
    total = len(seeds) * len(ARMS)
    done = 0
    for seed in seeds:
        for tag, binary, fac in ARMS:
            rec = play(binary, fac, seed)
            runs[tag].append(rec)
            done += 1
            print(f"[{tag}] {done}/{total} seed={seed} score={rec['score']} "
                  f"pt={rec.get('pt')} rank={rec.get('rank')} "
                  f"inherit={rec.get('inherit_turns')} {rec['secs']}s", flush=True)

    summary = []
    for tag, _, fac in ARMS:
        rs = [r for r in runs[tag] if r["score"] > 0]
        summary.append({"tag": tag, "factors": fac, "n": len(rs),
                        "inherit_ok": sum(1 for r in rs if len(r.get("inherit_turns", [])) == 2),
                        "score_mean": mean([r["score"] for r in rs]),
                        "score_min": min((r["score"] for r in rs), default=0),
                        "score_max": max((r["score"] for r in rs), default=0),
                        "pt_mean": mean([r["pt"] for r in rs]),
                        "five_per_axis": [mean([r["five"][i] for r in rs if len(r["five"]) == 5])
                                          for i in range(5)],
                        "five_total_mean": mean([sum(r["five"]) for r in rs if r["five"]]),
                        "act_lv_mean": [mean([r["lv"][i] for r in rs if len(r["lv"]) == 5])
                                        for i in range(5)]})
    out = {"summary": summary,
           "paired": all_pairs(runs, [tag for tag, _, _ in ARMS]),
           "runs": runs}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({"summary": summary, "paired": out["paired"]}, ensure_ascii=False, indent=1))
    (REPO / "matsuri_config.toml").unlink(missing_ok=True)
    (REPO / "ai_config.toml").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
