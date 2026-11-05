#!/usr/bin/env python3
"""演目 gauge 权重探针：weights.gauge 27.9217（ga-r2/上游默认）是否偏低。

三臂配对 CRN，共 99 局（≤100 硬上限）：gauge ∈ {27.9217, 36, 45}，其余全用 ga-r2 冠军层，
马 100101 + 参考卡组（生产默认），MATSURI_NORMAL_SEARCH_N=128 / TOPK=4。
GA 搜索区间是 (0, 60)（scripts/ga.py:59），27.9217 不是撞边界收敛，所以「是否偏低」只能实测。

用法：python3 scripts/gauge_probe.py [--seeds 61444-61476] [--smoke]
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
ARMS = [("g27.92", 27.9217), ("g36", 36.0), ("g45", 45.0)]

SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
RANK_RE = re.compile(r"^评分\s+\d+\s+(\S+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")


def overlay(gauge: float) -> str:
    text = BASE.read_text(encoding="utf8")
    new, n = re.subn(r"^gauge\s*=\s*[\d.]+", f"gauge = {gauge:.4f}", text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("ga-r2.toml 里找不到 gauge 行")
    return new


def play(cfg_text: str, seed: int) -> dict:
    (REPO / "matsuri_config.toml").write_text(cfg_text, encoding="utf8")
    (REPO / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {UMA}\ncards = [{', '.join(map(str, CARDS))}]\n",
        encoding="utf8")
    env = {**os.environ,
           "MATSURI_NORMAL_SEARCH_N": "128", "MATSURI_NORMAL_SEARCH_TOPK": "4"}
    t0 = time.time()
    try:
        p = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                           text=True, errors="replace", timeout=900, env=env)
    except subprocess.TimeoutExpired:
        print(f"  seed {seed} 超时 900s，重试一次", flush=True)
        t0 = time.time()
        p = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                           text=True, errors="replace", timeout=1800, env=env)
    out, err = p.stdout, p.stderr
    rec = {"seed": seed, "secs": round(time.time() - t0, 1)}
    m = SCORE_RE.search(out)
    if not m:
        rec["score"] = 0
        rec["err"] = (out[-300:] + err[-300:]).strip()
        return rec
    five = FIVE_RE.search(out)
    lv = LV_RE.search(err)
    rank = RANK_RE.search(out)
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
    var = sum((x - mu) ** 2 for x in d) / max(n - 1, 1)
    sd = var ** 0.5
    t = mu / (sd / n ** 0.5) if sd > 0 else 0.0
    return {"delta": round(mu, 1), "t": round(t, 2),
            "wins": f"{sum(1 for x in d if x > 0)}/{n}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="61444-61476")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/tmp/gauge_probe_results.json")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if args.smoke:
        seeds = seeds[:1]

    cfgs = [(tag, overlay(g)) for tag, g in ARMS]
    runs = {tag: [] for tag, _ in cfgs}
    total = len(seeds) * len(cfgs)
    done = 0
    for seed in seeds:
        for tag, cfg in cfgs:
            rec = play(cfg, seed)
            runs[tag].append(rec)
            done += 1
            print(f"[{tag}] {done}/{total} seed={seed} score={rec['score']} "
                  f"pt={rec.get('pt')} lv={rec.get('lv')} {rec['secs']}s", flush=True)

    summary = []
    for tag, _ in cfgs:
        rs = [r for r in runs[tag] if r["score"] > 0]
        summary.append({
            "tag": tag, "gauge": dict(ARMS)[tag], "n": len(rs),
            "score_mean": mean([r["score"] for r in rs]),
            "score_min": min(r["score"] for r in rs),
            "score_max": max(r["score"] for r in rs),
            "pt_mean": mean([r["pt"] for r in rs]),
            "five_total_mean": mean([sum(r["five"]) for r in rs if r["five"]]),
            "five_per_axis": [mean([r["five"][i] for r in rs if len(r["five"]) == 5])
                              for i in range(5)],
            "act_lv_mean": [mean([r["lv"][i] for r in rs if len(r["lv"]) == 5])
                            for i in range(5)],
        })
    base = runs[ARMS[0][0]]
    out = {"summary": summary,
           "paired_vs_baseline": {tag: paired(base, runs[tag])
                                  for tag, _ in cfgs[1:]},
           "runs": runs}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({"summary": summary,
                      "paired_vs_baseline": out["paired_vs_baseline"]},
                     ensure_ascii=False, indent=1))
    (REPO / "matsuri_config.toml").unlink(missing_ok=True)
    (REPO / "ai_config.toml").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
