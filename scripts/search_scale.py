#!/usr/bin/env python3
"""常规回合前瞻搜索的 N 缩放实测：128（生产档）→ 1024 → 8192。

成本口径（trainer.rs:107）：每局 ~65 个常规决策点 × topk 个候选 × 每个候选 search_n 次
rollout 到终局，所以耗时对 search_n 是**线性**的：128→8192 = 64×，单局 ~13s → ~830s。

配对 CRN，全新随机种子块（避开 lab 已用的 61444–63699），其余全用 ga-r2 冠军层 +
生产卡组/马（100101 + 参考卡组）。用法：python3 scripts/search_scale.py [--seeds 70001-70006]
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
# (标签, search_n, topk, radical_factor_max)；由 --ns / --radicals 笛卡尔积生成
ARMS: list[tuple[str, int, int, float]] = []

SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
RANK_RE = re.compile(r"^评分\s+\d+\s+(\S+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")


def play(binary: Path, seed: int, n: int, topk: int, radical: float) -> dict:
    (REPO / "matsuri_config.toml").write_text(BASE.read_text(encoding="utf8"), encoding="utf8")
    (REPO / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {UMA}\ncards = [{', '.join(map(str, CARDS))}]\n"
        f"[search]\nradical_factor_max = {radical}\n", encoding="utf8")
    env = {**os.environ, "MATSURI_NORMAL_SEARCH_N": str(n),
           "MATSURI_NORMAL_SEARCH_TOPK": str(topk)}
    t0 = time.time()
    try:
        p = subprocess.run([str(binary), "matsuri"], cwd=REPO, capture_output=True,
                           text=True, errors="replace", timeout=7200, env=env)
    except subprocess.TimeoutExpired:
        return {"seed": seed, "score": 0, "err": "timeout 7200s",
                "secs": round(time.time() - t0, 1)}
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
    ap.add_argument("--seeds", default="70001-70006")
    ap.add_argument("--out", default="/tmp/search_scale_results.json")
    ap.add_argument("--binary", default=str(BIN))
    ap.add_argument("--ns", default="128,1024,8192", help="search_n 档，逗号分隔")
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--radicals", default="50.0", help="radical_factor_max 档，逗号分隔")
    args = ap.parse_args()
    binary = Path(args.binary)
    global ARMS
    ARMS = [(f"n{n}-rad{r:g}", n, args.topk, r)
            for n in (int(x) for x in args.ns.split(","))
            for r in (float(x) for x in args.radicals.split(","))]
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    runs = {tag: [] for tag, _, _, _ in ARMS}
    total = len(seeds) * len(ARMS)
    done = 0
    for seed in seeds:
        for tag, n, topk, rad in ARMS:
            rec = play(binary, seed, n, topk, rad)
            runs[tag].append(rec)
            done += 1
            print(f"[{tag}] {done}/{total} seed={seed} score={rec['score']} "
                  f"pt={rec.get('pt')} lv={rec.get('lv')} {rec['secs']}s", flush=True)

    summary = []
    for tag, n, topk, rad in ARMS:
        rs = [r for r in runs[tag] if r["score"] > 0]
        summary.append({"tag": tag, "search_n": n, "topk": topk, "radical": rad,
                        "n_games": len(rs),
                        "secs_mean": mean([r["secs"] for r in rs]),
                        "score_mean": mean([r["score"] for r in rs]),
                        "score_min": min((r["score"] for r in rs), default=0),
                        "score_max": max((r["score"] for r in rs), default=0),
                        "pt_mean": mean([r["pt"] for r in rs]),
                        "five_per_axis": [mean([r["five"][i] for r in rs if len(r["five"]) == 5])
                                          for i in range(5)],
                        "act_lv_mean": [mean([r["lv"][i] for r in rs if len(r["lv"]) == 5])
                                        for i in range(5)]})
    base = runs[ARMS[0][0]]
    out = {"summary": summary,
           "paired_vs_first": {tag: paired(base, runs[tag]) for tag, _, _, _ in ARMS[1:]},
           "runs": runs}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({"summary": summary, "paired_vs_first": out["paired_vs_first"]},
                     ensure_ascii=False, indent=1))
    (REPO / "matsuri_config.toml").unlink(missing_ok=True)
    (REPO / "ai_config.toml").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
