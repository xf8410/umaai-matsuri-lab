#!/usr/bin/env python3
"""卡根流派实验（用户 2026-11-04 指定策略）：两臂配对 CRN，共 100 局。

臂 A（root-flow）：lab/config/root-flow.toml + 用户钉位路线 + 卡根卡组 + 泳装青云天空 102003
臂 B（ga-r2 基线）：lab/config/ga-r2.toml + Trainer 自选（advice.route=500 先验）+ 同卡组同马

用户钉位路线（MATSURI_ASSIGN_PIN，buff id 串按演目序 速/耐/力/根/智）：
  窗口0/1（T2/T11）  401-201-301-501-101   速助力 耐得意率 力感叹号 根体力 智羁绊
  窗口2/3（T23/T35） 401-301-601-501-701   year2 转成后：力特化 智钻研（第二次选择不变故沿用）
  窗口4/5（T47/T59） 801-601-1001-901-701  year3 转成后：速协力 耐特化 力兼修 根不屈 智钻研
  （401+801 硬约束可能把协力换到担当最多演目 → 钉位不命中则回落 Trainer，逐局记录。）

用法：python3 scripts/root_flow.py [--seeds 61444-61493] [--smoke]
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
UMA = "102003"  # 【渚のきまぐれオランジュ】セイウンスカイ（泳装青云天空）
# 卡根卡组：速乐透心 / 耐樱花千代王 / 力摩耶重炮 / 根微光飞驹(30315,根+智 gain,hintLv5) /
# 智爱慕律动 / 友吾妻(SSR 锁定)。替换掉 ga-r2 参考卡组里的第二张速卡（待兼诗歌剧 303174）。
CARDS = [303114, 303004, 303224, 303154, 303204, 303234]
PIN = ",".join([
    "0:401-201-301-501-101",
    "1:401-201-301-501-101",
    "2:401-301-601-501-701",
    "3:401-301-601-501-701",
    "4:801-601-1001-901-701",
    "5:801-601-1001-901-701",
])

SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")
PIN_RE = re.compile(r"【教诲】T(\d+) 钉位 窗口(\d+)")
ASSIGN_RE = re.compile(r"【教诲】T(\d+) 分配 slots=\[([0-9, ]+)\] fused=\[([0-9, ]+)\]")
FUSION_RE = re.compile(r"【教诲】T(\d+) 融合 演目(\d+): (\d+)→(\d+)")
SWAP_RE = re.compile(r"【教诲】T(\d+) 401\+801 硬约束 交换演目 (\d+)↔(\d+)")


def play(overlay: Path, seed: int, pin: str) -> dict:
    (REPO / "matsuri_config.toml").write_text(overlay.read_text(encoding="utf8"), encoding="utf8")
    (REPO / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {UMA}\ncards = [{', '.join(map(str, CARDS))}]\n",
        encoding="utf8")
    env = {**os.environ,
           "MATSURI_NORMAL_SEARCH_N": "128", "MATSURI_NORMAL_SEARCH_TOPK": "4"}
    if pin:
        env["MATSURI_ASSIGN_PIN"] = pin
    t0 = time.time()
    try:
        proc = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                              text=True, errors="replace", timeout=900, env=env)
    except subprocess.TimeoutExpired:
        # 偶发 CPU 争用卡死（实测同种子重跑 ~2 分钟即完成）：重试一次
        print(f"  seed {seed} 超时 900s，重试一次", flush=True)
        t0 = time.time()
        proc = subprocess.run([str(BIN), "matsuri"], cwd=REPO, capture_output=True,
                              text=True, errors="replace", timeout=1800, env=env)
    out, err = proc.stdout, proc.stderr
    m = SCORE_RE.search(out)
    rec = {"seed": seed, "secs": round(time.time() - t0, 1)}
    if not m:
        rec["score"] = 0
        rec["err"] = (out[-300:] + err[-300:]).strip()
        return rec
    five = FIVE_RE.search(out)
    lv = LV_RE.search(err)
    rec["score"] = int(m.group(1))
    rec["pt"] = int(PT_RE.search(out).group(1)) if PT_RE.search(out) else 0
    rec["five"] = [int(x) for x in five.groups()] if five else []
    rec["lv"] = [int(x) for x in lv.group(1).split(",")] if lv else []
    # 教诲记录：钉位命中窗口 / 每窗口落地 slots+fused / 融合 / 硬约束交换
    rec["pin_hits"] = sorted({int(g[1]) for g in PIN_RE.findall(err)})
    rec["assigns"] = [{"t": int(t), "slots": [int(x) for x in s.split(",")],
                       "fused": [int(x) for x in f.split(",")]}
                      for t, s, f in ASSIGN_RE.findall(err)]
    rec["fusions"] = [{"t": int(t), "act": int(a), "base": int(b), "target": int(g)}
                      for t, a, b, g in FUSION_RE.findall(err)]
    rec["swaps801"] = [{"t": int(t), "a": int(x), "b": int(y)} for t, x, y in SWAP_RE.findall(err)]
    return rec


def paired(a: list[int], b: list[int]):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / n
    t = mean / ((var ** 0.5) / (n ** 0.5)) if var else 0.0
    return mean, t, sum(1 for x in d if x > 0), n


def summarize(tag: str, recs: list[dict]) -> dict:
    ok = [r for r in recs if r["score"] > 0]
    if not ok:
        return {"tag": tag, "n": 0}
    scores = [r["score"] for r in ok]
    pts = [r["pt"] for r in ok]
    fives = [r["five"] for r in ok if r["five"]]
    lvs = [r["lv"] for r in ok if r["lv"]]
    per_axis = [round(sum(f[i] for f in fives) / len(fives), 1) for i in range(5)]
    per_lv = [round(sum(l[i] for l in lvs) / len(lvs), 2) for i in range(5)]
    return {
        "tag": tag, "n": len(ok),
        "score_mean": round(sum(scores) / len(scores), 1),
        "score_min": min(scores), "score_max": max(scores),
        "pt_mean": round(sum(pts) / len(pts), 1),
        "five_total_mean": round(sum(sum(f) for f in fives) / len(fives), 1),
        "five_per_axis": per_axis, "act_lv_mean": per_lv,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="61444-61493")
    ap.add_argument("--smoke", action="store_true", help="只跑每臂前 2 个种子")
    ap.add_argument("--out", default="/tmp/root_flow_results.json")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if args.smoke:
        seeds = seeds[:2]

    arms = [("root-flow", REPO / "lab/config/root-flow.toml", PIN),
            ("ga-r2", REPO / "lab/config/ga-r2.toml", "")]
    results = {}
    try:
        for tag, overlay, pin in arms:
            recs = []
            for i, seed in enumerate(seeds):
                rec = play(overlay, seed, pin)
                recs.append(rec)
                print(f"[{tag}] {i + 1}/{len(seeds)} seed={seed} score={rec['score']} "
                      f"pins={rec.get('pin_hits')} {rec['secs']}s", flush=True)
            results[tag] = recs
    finally:
        for f in ("matsuri_config.toml", "ai_config.toml"):
            (REPO / f).unlink(missing_ok=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf8")
    sums = [summarize(tag, results[tag]) for tag, _, _ in arms]
    a = [r["score"] for r in results["root-flow"] if r["score"] > 0]
    b = [r["score"] for r in results["ga-r2"] if r["score"] > 0]
    report = {"summary": sums}
    if len(a) == len(b) and a:
        mean, t, wins, n = paired(a, b)
        report["paired"] = {"delta": round(mean, 1), "t": round(t, 2),
                            "wins": f"{wins}/{n}"}
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
