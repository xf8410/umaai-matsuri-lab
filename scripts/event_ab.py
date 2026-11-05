#!/usr/bin/env python3
"""事件选项前瞻搜索的档位 A/B（上游 21b278c 第四路，本仓 c7b1ac9 自持移植）。

为什么要单独量：事件决策点 **166–184 个/局**，比常规训练点（~65/局）多 2.5 倍，
是四路搜索里最贵的一面（RAYON=1 实测 49s → 74s@N=32 → 95s@N=128）。
而 `flags_value` 符号修复（3cf379d）已经证明事件选项的静态排序此前是**系统性反向**的，
所以这一路既可能收益最大、也可能纯烧钱——只能实测。

臂 = `MATSURI_EVENT_SEARCH_N` 档（0 = 关，走静态期望值打分）；其余全固定为生产档
（常规 N=128 / topk=4 / radical=0）+ ga-r2 冠军层 + 生产卡组/马。同种子 CRN 配对。

用法：python3 scripts/event_ab.py --seeds 73641-73660 --ns 0,32,128
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
BIN = REPO / "target/release/matsuri"
BASE = REPO / "lab/config/ga-r2.toml"
UMA = "100101"
CARDS = [303174, 303114, 303004, 303224, 303204, 303234]

SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
RANK_RE = re.compile(r"^评分\s+\d+\s+(\S+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")
EV_RE = re.compile(r"【搜索】T\d+ 事件#")


def play(binary: Path, cfg: Path, seed: int, event_n: int, normal_n: int,
         topk: int) -> dict:
    (REPO / "matsuri_config.toml").write_text(cfg.read_text(encoding="utf8"), encoding="utf8")
    (REPO / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {UMA}\ncards = [{', '.join(map(str, CARDS))}]\n"
        f"[search]\nradical_factor_max = 0.0\n", encoding="utf8")
    env = {**os.environ, "MATSURI_NORMAL_SEARCH_N": str(normal_n),
           "MATSURI_NORMAL_SEARCH_TOPK": str(topk),
           "MATSURI_EVENT_SEARCH_N": str(event_n)}
    t0 = time.time()
    try:
        p = subprocess.run([str(binary), "matsuri"], cwd=REPO, capture_output=True,
                           text=True, errors="replace", timeout=7200, env=env)
    except subprocess.TimeoutExpired:
        return {"seed": seed, "score": 0, "err": "timeout 7200s",
                "secs": round(time.time() - t0, 1)}
    out, err = p.stdout, p.stderr
    rec = {"seed": seed, "secs": round(time.time() - t0, 1),
           "event_points": len(EV_RE.findall(err))}
    m = SCORE_RE.search(out)
    if not m:
        return {**rec, "score": 0, "err": (out[-300:] + err[-300:]).strip()}
    five, lv, rank = FIVE_RE.search(out), LV_RE.search(err), RANK_RE.search(out)
    rec["score"] = int(m.group(1))
    rec["rank"] = rank.group(1) if rank else ""
    rec["pt"] = int(PT_RE.search(out).group(1)) if PT_RE.search(out) else 0
    rec["five"] = [int(x) for x in five.groups()] if five else []
    rec["lv"] = [int(x) for x in lv.group(1).split(",")] if lv else []
    return rec


def mean(xs):
    return round(sum(xs) / len(xs), 1) if xs else 0.0


def paired(a, b):
    d = [y["score"] - x["score"] for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    sd = (sum((x - mu) ** 2 for x in d) / max(n - 1, 1)) ** 0.5
    return {"delta": round(mu, 1), "t": round(mu / (sd / n ** 0.5), 2) if sd > 0 else 0.0,
            "wins": f"{sum(1 for x in d if x > 0)}/{n}",
            "secs_mean": round(mean([r["secs"] for r in b]), 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="73641-73660")
    ap.add_argument("--ns", default="0,32,128", help="MATSURI_EVENT_SEARCH_N 档，逗号分隔")
    ap.add_argument("--normal-n", type=int, default=128)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--binary", default=str(BIN))
    ap.add_argument("--config", default=str(BASE))
    ap.add_argument("--out", default="/tmp/event_ab.json")
    args = ap.parse_args()
    binary, cfg = Path(args.binary), Path(args.config)
    ns = [int(x) for x in args.ns.split(",")]
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    runs = {f"ev{n}": [] for n in ns}
    total = len(seeds) * len(ns)
    done = 0
    for seed in seeds:
        for n in ns:
            rec = play(binary, cfg, seed, n, args.normal_n, args.topk)
            runs[f"ev{n}"].append(rec)
            done += 1
            print(f"[ev{n}] {done}/{total} seed={seed} score={rec['score']} "
                  f"pt={rec.get('pt')} rank={rec.get('rank')} "
                  f"事件搜索点={rec.get('event_points')} {rec['secs']}s", flush=True)

    summary = []
    for n in ns:
        rs = [r for r in runs[f"ev{n}"] if r["score"] > 0]
        s = sorted(r["score"] for r in rs)
        summary.append({"tag": f"ev{n}", "event_search_n": n, "n_games": len(rs),
                        "secs_mean": mean([r["secs"] for r in rs]),
                        "event_points_mean": mean([r.get("event_points", 0) for r in rs]),
                        "score_mean": mean([r["score"] for r in rs]),
                        "score_min": s[0] if s else 0, "score_max": s[-1] if s else 0,
                        "worst5_mean": mean(s[:5]), "best5_mean": mean(s[-5:]),
                        "pt_mean": mean([r["pt"] for r in rs]),
                        "five_per_axis": [mean([r["five"][i] for r in rs if len(r["five"]) == 5])
                                          for i in range(5)],
                        "act_lv_mean": [mean([r["lv"][i] for r in rs if len(r["lv"]) == 5])
                                        for i in range(5)]})
    base = runs[f"ev{ns[0]}"]
    out = {"binary": str(binary), "config": str(cfg), "normal_n": args.normal_n,
           "topk": args.topk, "seeds": [seeds[0], seeds[-1]], "summary": summary,
           "paired_vs_first": {f"ev{n}": paired(base, runs[f"ev{n}"]) for n in ns[1:]},
           "runs": runs}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({"summary": summary, "paired_vs_first": out["paired_vs_first"]},
                     ensure_ascii=False, indent=1))
    (REPO / "matsuri_config.toml").unlink(missing_ok=True)
    (REPO / "ai_config.toml").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
