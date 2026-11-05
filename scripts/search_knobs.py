#!/usr/bin/env python3
"""`[search]` 段旋钮单变量扫描（维度扩容第一步：34 → 41 候选基因）。

背景（`lab/RESULTS.md` §9.4）：`radical_factor_max = 0` 单变量就值 +2621.8（t=4.11、12/12、
零成本），而 `[search]` 段的其余旋钮**从未进过 GA 基因组、也从未扫过**。本脚本按
「一次只动一个旋钮」配对 CRN 扫，其余全用 ga-r2 冠军层 + 生产卡组/马（100101 + 参考卡组）。

成本口径：搜索成本 = 决策点数 × topk × search_n，所以只有 topk / search_n 臂会改耗时，
UCB / cpuct / stdev / pt_favor 臂是**同耗时**的（跟 rad0 一样属于免费档）。

用法：python3 scripts/search_knobs.py --seeds 73517-73536 --arms base,g16,noucb
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from search_scale import (  # noqa: E402
    CARDS, UMA, mean, paired, SCORE_RE, PT_RE, FIVE_RE, LV_RE, RANK_RE,
)

BASE_CFG = REPO / "lab/config/ga-r2.toml"

# 臂 = (标签, `[search]` 段覆盖行, 环境变量覆盖, `matsuri_config.toml` 文本覆盖)
# 空 dict = 沿用默认层（= base 臂）。
# 第四个元素是 {正则: 替换} 形式的**配置文本**覆盖，用于改 `[trainer]` 段的旋钮——
# 生产档的 `pt_favor_rate` 活在 `lab/config/ga-r2.toml` 的 `[trainer]` 里，
# 而不是 `[search]` 段（上游 21b278c 起 `[search].pt_favor_rate` 只作为剧本搜索档的
# 覆盖来源；早期 `pt1`/`pt3` 两臂写在 `[search]` 段里，实测是**死臂**，故改成文本覆盖）。
ARMS: dict[str, tuple[dict, dict, dict]] = {
    "base": ({}, {}, {}),
    # UCB 每组追加次数：默认 256 > search_n=128，在生产档下可能退化（一组就吃光预算）
    # 上游 c55d82d 反向调到 512（配合它的新预算 search_n=12288 才有意义）
    "g8": ({"search_group_size": 8}, {}, {}),
    "g16": ({"search_group_size": 16}, {}, {}),
    "g32": ({"search_group_size": 32}, {}, {}),
    "g64": ({"search_group_size": 64}, {}, {}),
    "g512": ({"search_group_size": 512}, {}, {}),
    "g1024": ({"search_group_size": 1024}, {}, {}),
    # 均匀分配对照（关掉 UCB 动态分配）
    "noucb": ({"use_ucb": "false"}, {}, {}),
    # 探索项 cpuct × expected_search_stdev 以乘积出现，只需扫其积
    "cpuct0.25": ({"search_cpuct": 0.25}, {}, {}),
    "cpuct0.5": ({"search_cpuct": 0.5}, {}, {}),
    "cpuct2": ({"search_cpuct": 2.0}, {}, {}),
    "cpuct4": ({"search_cpuct": 4.0}, {}, {}),
    # PT 偏好折算率：**trainer 级**（真正生效的那一个）。上游 c55d82d 把引擎默认层
    # 2.0 → 1.0，剧本搜索档覆盖为 1.7271（= 我们的现值）；这里直接扫 1.0 / 1.7271 / 3.0
    "pt1": ({}, {}, {r"^(\s*pt_favor_rate\s*=\s*).*$": r"\g<1>1.0"}),
    "pt3": ({}, {}, {r"^(\s*pt_favor_rate\s*=\s*).*$": r"\g<1>3.0"}),
    "pt2": ({}, {}, {r"^(\s*pt_favor_rate\s*=\s*).*$": r"\g<1>2.0"}),
    # 剪枝宽度（**唯一会加钱的档**：1.5× / 2×）
    "topk6": ({}, {"MATSURI_NORMAL_SEARCH_TOPK": "6"}, {}),
    "topk8": ({}, {"MATSURI_NORMAL_SEARCH_TOPK": "8"}, {}),
    "topk2": ({}, {"MATSURI_NORMAL_SEARCH_TOPK": "2"}, {}),
}


def apply_text_overrides(text: str, over: dict) -> str:
    """按 {正则: 替换} 逐条改写配置文本；每条都必须真的命中，否则报错（防死臂）。"""
    for pat, rep in over.items():
        new, n = re.subn(pat, rep, text, flags=re.M)
        if n == 0:
            raise SystemExit(f"配置文本覆盖未命中：{pat!r}（防死臂检查失败）")
        text = new
    return text


def play(binary: Path, cfg: Path, seed: int, n: int, topk: int,
         knobs: dict, env_over: dict, cfg_over: dict) -> dict:
    search = [f"search_n = {n}", f"radical_factor_max = 0.0"]
    for k, v in knobs.items():
        search.append(f"{k} = {v}")
    (REPO / "matsuri_config.toml").write_text(
        apply_text_overrides(cfg.read_text(encoding="utf8"), cfg_over), encoding="utf8")
    (REPO / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {UMA}\ncards = [{', '.join(map(str, CARDS))}]\n"
        f"[search]\n" + "\n".join(search) + "\n", encoding="utf8")
    env = {**os.environ, "MATSURI_NORMAL_SEARCH_N": str(n),
           "MATSURI_NORMAL_SEARCH_TOPK": str(topk), **env_over}
    t0 = time.time()
    import subprocess
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="73517-73536")
    ap.add_argument("--out", default="/tmp/search_knobs.json")
    ap.add_argument("--binary", default=str(REPO / "target/release/matsuri"))
    ap.add_argument("--config", default=str(BASE_CFG))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--arms", default="base,g8,g16,g32,g64,noucb,cpuct0.5,cpuct2,pt1,topk6")
    args = ap.parse_args()
    binary, cfg = Path(args.binary), Path(args.config)
    tags = [t.strip() for t in args.arms.split(",") if t.strip()]
    unknown = [t for t in tags if t not in ARMS]
    if unknown:
        print(f"未知臂：{unknown}；可选 {sorted(ARMS)}")
        return 2
    lo, hi = (int(x) for x in args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))

    runs = {t: [] for t in tags}
    total = len(seeds) * len(tags)
    done = 0
    for seed in seeds:
        for tag in tags:
            knobs, env_over, cfg_over = ARMS[tag]
            rec = play(binary, cfg, seed, args.n, args.topk, knobs, env_over, cfg_over)
            runs[tag].append(rec)
            done += 1
            print(f"[{tag}] {done}/{total} seed={seed} score={rec['score']} "
                  f"pt={rec.get('pt')} lv={rec.get('lv')} {rec['secs']}s", flush=True)

    summary = []
    for tag in tags:
        rs = [r for r in runs[tag] if r["score"] > 0]
        s = sorted(r["score"] for r in rs)
        summary.append({"tag": tag, "knobs": ARMS[tag][0], "env": ARMS[tag][1],
                        "cfg_over": list(ARMS[tag][2]),
                        "n_games": len(rs), "secs_mean": mean([r["secs"] for r in rs]),
                        "score_mean": mean([r["score"] for r in rs]),
                        "score_min": s[0] if s else 0, "score_max": s[-1] if s else 0,
                        "worst5_mean": mean(s[:5]), "best5_mean": mean(s[-5:]),
                        "pt_mean": mean([r["pt"] for r in rs]),
                        "five_per_axis": [mean([r["five"][i] for r in rs if len(r["five"]) == 5])
                                          for i in range(5)],
                        "act_lv_mean": [mean([r["lv"][i] for r in rs if len(r["lv"]) == 5])
                                        for i in range(5)]})
    base = runs[tags[0]]
    out = {"binary": str(binary), "config": str(cfg), "search_n": args.n,
           "topk": args.topk, "seeds": [seeds[0], seeds[-1]], "summary": summary,
           "paired_vs_first": {t: paired(base, runs[t]) for t in tags[1:]},
           "runs": runs}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({"summary": summary, "paired_vs_first": out["paired_vs_first"]},
                     ensure_ascii=False, indent=1))
    (REPO / "matsuri_config.toml").unlink(missing_ok=True)
    (REPO / "ai_config.toml").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
