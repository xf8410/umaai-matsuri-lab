"""転成（`[fusion]`）120 组合的 UCB1 多臂老虎机搜索（骏大祭）。

臂空间 = 一个合法的 `[fusion]` 预设：
  year2 从 5 个基础 buff（101 羁绊 / 201 得意率 / 301 提示 / 401 助力 / 501 体力减轻）里
  **有序**挑 2 个映射到 601 / 701 → 5×4 = 20 种；
  year3 把剩下 3 个映射到 801 / 901 / 1001 → 3! = 6 种；合计 **120 种**。
这正是 `data/matsuri_config.toml` 里那套融合校验（两年合计覆盖全部 5 个基础 buff、
target 各用一次）允许的全部组合。

为什么用 UCB1 而不是直接穷举：120 个臂穷举得起，但**预算分配**不一样。
UCB1 把局数堆到当前看起来有希望的臂上，「找出最优臂」这件事上比均匀分配省局数；
均匀穷举的好处是每个臂的估计精度一样。这里两种模式都跑，用同一种子口径直接对比：
  --mode ucb          每臂先各拉 1 个 block（保证每个臂都被摸过），剩余预算按 UCB1 分配
  --mode exhaustive   每臂固定拉同样多的 block（预算与 ucb 模式相同）
比的是：花掉同样局数后，两种分配选出来的最优臂在**留出种子**上的真实均分谁高。

奖励归一化：模拟器给定 (seed, 配置) 是**确定性**的，所以「同一个 block 重复拉」没有噪声，
UCB 会退化成穷举。噪声来自**种子**：第 k 次拉臂 i 用的是第 k 个种子 block
（`screen_seed + k*block … +block-1`），不同 block 的均分自然有抽样波动，这才是老虎机的随机性。
分数按 `REWARD_LO/HI` 线性压到 [0,1] 再用 UCB1 公式（置信项 `c*sqrt(2 ln t / n_i)`）。

与教诲路线正交：`ROUTE_PRIOR`（value.rs）按**基础 buff id** 编码，换転成预设不需要重蒸馏路线，
所以这一轴可以独立搜（RESULTS 7.4/7.5）。默认不钉位、不开 MC 搜索，纯手写打分 + 路线先验。

用法：
    python3 scripts/fusion_ucb.py --mode ucb --budget 960 --block 4 \
        --overlay lab/config/route-prior.toml --out artifacts/fusion-ucb
    python3 scripts/fusion_ucb.py --mode exhaustive --budget 960 --block 4 \
        --overlay lab/config/route-prior.toml --out artifacts/fusion-exh
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from route_search import paired, play  # noqa: E402

BASES = [101, 201, 301, 401, 501]
YEAR2_TARGETS = [601, 701]
YEAR3_TARGETS = [801, 901, 1001]
BASE_NAMES = {101: "羁绊", 201: "得意率", 301: "提示", 401: "助力", 501: "体力减轻"}

# 奖励归一化区间：留出集实测评分跨度 54622–80835，取一个略宽的整数边界
REWARD_LO, REWARD_HI = 60000.0, 82000.0


def all_arms() -> list[tuple[tuple[int, int], tuple[int, int, int]]]:
    """120 个臂：(year2 的两个 base, year3 的三个 base)，位置对应 target 顺序。"""
    arms = []
    for y2 in itertools.permutations(BASES, 2):
        rest = [b for b in BASES if b not in y2]
        for y3 in itertools.permutations(rest, 3):
            arms.append((y2, y3))
    return arms


def arm_label(arm: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    y2, y3 = arm
    a = " ".join(f"{b}→{t}" for b, t in zip(y2, YEAR2_TARGETS))
    c = " ".join(f"{b}→{t}" for b, t in zip(y3, YEAR3_TARGETS))
    return f"year2[{a}] year3[{c}]"


def fusion_toml(arm: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    y2, y3 = arm
    pairs = ", ".join(f"{{ base = {b}, target = {t} }}"
                      for b, t in zip(y2, YEAR2_TARGETS))
    y3_lines = "\n".join(f"    {{ base = {b}, target = {t} }},"
                         for b, t in zip(y3, YEAR3_TARGETS))
    return f"[fusion]\nyear2 = [{pairs}]\nyear3 = [\n{y3_lines}\n]\n"


def overlay_with_fusion(base_overlay: str, arm) -> str:
    """把覆盖层里的 `[fusion]` 段整段换成这个臂（列表是整段替换，不能只改一行）。"""
    lines = base_overlay.splitlines()
    out, skipping = [], False
    for line in lines:
        s = line.strip()
        if s.startswith("["):
            skipping = s == "[fusion]"
            if skipping:
                continue
        if skipping and not s.startswith("["):
            continue
        out.append(line)
    return fusion_toml(arm) + "\n" + "\n".join(out).lstrip("\n") + "\n"


def make_workdir(repo: Path, tag: str) -> Path:
    work = Path(f"/tmp/fusion-ucb-{tag}")
    work.mkdir(parents=True, exist_ok=True)
    link = work / "data"
    if not link.exists():
        link.symlink_to(repo / "data")
    return work


def parse_fusion(text: str) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    """从覆盖层里读回 (year2 bases, year3 bases)，用来定位「起点臂」。

    两种合法写法都要认：`[[fusion.year2]]` 分节写法，和 `[fusion]` 下的内联/多行数组写法
    （`year2 = [{ base = 101, … }, …]`）。多行数组里 `{ base = … }` 单独成行，
    所属组由**上一个出现的键名**决定。
    """
    y2: list[int] = []
    y3: list[int] = []
    section, pending = "", None
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("["):
            section = s.strip("[] ")
            pending = None
            continue
        if not s or s.startswith("#"):
            continue
        key = s.split("=", 1)[0].strip()
        if key in ("year2", "year3"):
            pending = key
        if section == "fusion.year2":
            which = "year2"
        elif section == "fusion.year3":
            which = "year3"
        else:
            which = pending
        if which is None:
            continue
        for base in re.findall(r"base\s*=\s*(\d+)", s):
            (y2 if which == "year2" else y3).append(int(base))
    if len(y2) != 2 or len(y3) != 3:
        return None
    return tuple(y2), tuple(y3)


def block_seeds(screen_seed: int, block: int, k: int) -> list[int]:
    return [screen_seed + k * block + j for j in range(block)]


def pull(binary: Path, work: Path, overlay: str, seeds: list[int],
         uma: str, cards: str) -> list[int]:
    (work / "matsuri_config.toml").write_text(overlay, encoding="utf8")
    scores = []
    for seed in seeds:
        r = play(binary, work, seed, uma, cards, "")
        if r["score"] == 0:
            print(f"  seed {seed} 没跑出终局块: {r.get('err', '')[:200]}", file=sys.stderr)
            continue
        scores.append(r["score"])
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="lab/config/route-prior.toml")
    ap.add_argument("--mode", choices=["ucb", "exhaustive"], default="ucb")
    ap.add_argument("--budget", type=int, default=960, help="筛选阶段总局数上限")
    ap.add_argument("--block", type=int, default=4, help="每次拉臂跑几局（一个种子 block）")
    ap.add_argument("--ucb-c", type=float, default=0.35, help="UCB1 置信项系数（归一化奖励口径）")
    ap.add_argument("--screen-seed", type=int, default=62300)
    ap.add_argument("--confirm-seed", type=int, default=62400)
    ap.add_argument("--confirm-count", type=int, default=100, help="留出矩阵局数（仓库硬上限 100）")
    ap.add_argument("--topk", type=int, default=2, help="留出集上确认前几名（外加起点臂）")
    ap.add_argument("--uma", default="102102")
    ap.add_argument("--cards", default="303124,303114,303004,303224,303204,303234")
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--out", default="artifacts/fusion-ucb")
    ap.add_argument("--no-confirm", action="store_true", help="只搜不确认")
    args = ap.parse_args()

    repo = Path.cwd()
    binary = (repo / args.binary).resolve()
    if not binary.exists():
        print(f"找不到可执行文件: {binary}", file=sys.stderr)
        return 2
    base_text = Path(args.overlay).read_text(encoding="utf8")
    arms = all_arms()
    start_arm = parse_fusion(base_text)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = make_workdir(repo, args.mode)
    overlays = {i: overlay_with_fusion(base_text, arm) for i, arm in enumerate(arms)}

    n_arms, block = len(arms), args.block
    pulls_per_arm = max(args.budget // (n_arms * block), 1)
    budget_pulls = args.budget // block
    print(f"臂 {n_arms} 个 · block {block} 局 · 预算 {args.budget} 局 = {budget_pulls} 次拉臂 · "
          f"模式 {args.mode}（均匀穷举 = 每臂 {pulls_per_arm} 次）", flush=True)
    if start_arm in arms:
        print(f"起点臂 = {arm_label(start_arm)}（#{arms.index(start_arm)}）", flush=True)
    else:
        print(f"起点臂 {start_arm} 不在 120 组合里（覆盖层没写 [fusion]？）", flush=True)

    # 每个臂：blocks[k] = 第 k 次拉臂的均分；games = 累计局数
    blocks: dict[int, list[float]] = {i: [] for i in range(n_arms)}
    scores: dict[int, list[int]] = {i: [] for i in range(n_arms)}
    t0 = time.time()
    total_pulls, total_games = 0, 0

    def do_pull(i: int, k: int) -> float:
        nonlocal total_pulls, total_games
        seeds = block_seeds(args.screen_seed, block, k)
        got = pull(binary, work, overlays[i], seeds, args.uma, args.cards)
        total_pulls += 1
        total_games += len(got)
        if not got:
            return 0.0
        mean = sum(got) / len(got)
        blocks[i].append(mean)
        scores[i].extend(got)
        return mean

    # 第一轮：每个臂各拉一次，UCB1 才有 n_i >= 1
    for i in range(n_arms):
        do_pull(i, 0)
    print(f"初始轮 {n_arms} 次拉臂 / {total_games} 局 · {time.time() - t0:.0f}s", flush=True)

    if args.mode == "exhaustive":
        for k in range(1, pulls_per_arm):
            for i in range(n_arms):
                do_pull(i, k)
            print(f"均匀轮 {k + 1}/{pulls_per_arm} · {total_games} 局 · "
                  f"{time.time() - t0:.0f}s", flush=True)
    else:
        # UCB1：臂的奖励已归一化到 [0,1]，置信项按拉臂总次数 t 算
        while total_pulls < budget_pulls:
            t = total_pulls
            best_i, best_ucb = -1, -1.0
            for i in range(n_arms):
                n_i = len(blocks[i])
                mean_r = sum(max(0.0, min(1.0, (b - REWARD_LO) / (REWARD_HI - REWARD_LO)))
                             for b in blocks[i]) / n_i
                ucb = mean_r + args.ucb_c * math.sqrt(2.0 * math.log(t) / n_i)
                if ucb > best_ucb:
                    best_i, best_ucb = i, ucb
            do_pull(best_i, len(blocks[best_i]))
            if total_pulls % 40 == 0:
                top = max(range(n_arms), key=lambda i: sum(blocks[i]) / len(blocks[i]))
                print(f"  pull {total_pulls}/{budget_pulls} · 局 {total_games} · "
                      f"当前最优臂 #{top} {sum(blocks[top]) / len(blocks[top]):.1f} · "
                      f"{time.time() - t0:.0f}s", flush=True)

    def arm_mean(i: int) -> float:
        return sum(scores[i]) / len(scores[i]) if scores[i] else 0.0

    order = sorted(range(n_arms), key=lambda i: -arm_mean(i))
    pulls_used = {i: len(blocks[i]) for i in range(n_arms)}
    result = {
        "mode": args.mode, "block": block, "games": total_games, "pulls": total_pulls,
        "seconds": round(time.time() - t0, 1),
        "pull_distribution": {
            "min": min(pulls_used.values()), "max": max(pulls_used.values()),
            "pulled_once": sum(1 for v in pulls_used.values() if v == 1),
        },
        "best": [{"arm": arm_label(arms[i]), "index": i,
                  "screen_mean": round(arm_mean(i), 1),
                  "screen_min": min(scores[i]), "screen_max": max(scores[i]),
                  "games": len(scores[i]), "pulls": pulls_used[i]}
                 for i in order[:10]],
    }
    (out / "screen.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                    encoding="utf8")
    print(f"\n筛选完成：{total_games} 局 / {total_pulls} 次拉臂 / {time.time() - t0:.0f}s")
    print(f"拉臂分配 min {result['pull_distribution']['min']} "
          f"max {result['pull_distribution']['max']} "
          f"（只被摸过一次的臂 {result['pull_distribution']['pulled_once']} 个）")
    for row in result["best"][:5]:
        print(f"  #{row['index']:>3} {row['screen_mean']:>8.1f} "
              f"({row['games']} 局, {row['pulls']} 拉) {row['arm']}")

    if args.no_confirm:
        return 0

    # 留出集确认：前 topk 名 + 起点臂，各一个 <=100 局矩阵（种子全程未参与筛选）
    confirm_idx = list(dict.fromkeys(order[:args.topk] +
                                     ([arms.index(start_arm)] if start_arm in arms else [])))
    seeds = [args.confirm_seed + i for i in range(args.confirm_count)]
    confirm: dict[int, dict] = {}
    for i in confirm_idx:
        got = pull(binary, work, overlays[i], seeds, args.uma, args.cards)
        confirm[i] = {"arm": arm_label(arms[i]), "n": len(got), "scores": got,
                      "mean": round(sum(got) / len(got), 1),
                      "min": min(got), "max": max(got)}
        print(f"留出 {confirm[i]['mean']:>8.1f} min {confirm[i]['min']} "
              f"max {confirm[i]['max']} · #{i} {confirm[i]['arm']}", flush=True)
    base_key = arms.index(start_arm) if start_arm in arms else confirm_idx[-1]
    report = {"mode": args.mode, "games_screen": total_games,
              "confirm_seed": args.confirm_seed, "confirm_count": args.confirm_count,
              "arms": {str(k): {kk: vv for kk, vv in v.items() if kk != "scores"}
                       for k, v in confirm.items()}}
    for k, v in confirm.items():
        if k == base_key:
            continue
        mean, t, wins = paired(v["scores"], confirm[base_key]["scores"])
        report.setdefault("paired_vs_start", {})[str(k)] = {
            "arm": v["arm"], "delta": round(mean, 1), "t": round(t, 2),
            "wins": wins, "n": len(v["scores"])}
        print(f"配对 #{k} − 起点 #{base_key}: Δ{mean:+.1f} t={t:.2f} "
              f"{wins}/{len(v['scores'])}", flush=True)
    (out / "confirm.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
