"""遗传算法迭代 trainer 权重（骏大祭）。

搜索空间 = `lab/config/d-status.toml` 的 `[trainer]` 全部数值旋钮（含 5 个 dynamic），
`[fusion]` 暂时固定（120 种组合是另一个离散问题，见 scripts/fusion_ucb.py）。

协议照搬旧仓库 xf8410/umaai-rs 的 `crates/umasim/src/genetic_optimizer.rs`
（GaParams 定稿值 + 44 代实测记录），我原来那版是简化到失真的：

| 项 | 旧版（我） | 现在（照旧仓库） |
|---|---|---|
| 种群 / 代数 | 16 / 12 | 24 / 40（旧仓库 pop 60 / gens 40，CI 传 50 / 100） |
| σ 变异步长 | 全程写死 0.12 | 0.15 → 0.03 线性衰减，停滞 5 代回升一次 |
| 评估 | 单级，所有个体同一批种子 | 两级：初筛 12 局 → 前 6 名晋升精评 40 局 |
| 精英 / 锦标赛 / 交叉 | 2 / k=3 / 0.7 | 4 / k=3 / 0.9 |
| 缓存 | 无 | (基因组指纹, 级别) 命中即免跑 |
| 过拟合哨兵 | 只在最后看一次 gap | 每代冠军跑留出集 12 局，gap 逐代落盘 |
| 入榜口径 | 末代最优 | 全程**精评**最优（初筛值不入榜） |

旧仓库 ga_generations.csv 显示收益在 44 代之后仍在涨（gen44 68179 → gen52 68253
全局最优），所以「连续权重空间已经饱和」那个结论是 12 代跑出来的，不成立。

防过拟合：精评种子（--start-seed 起 40 个）只用于选择，留出种子
（--holdout-seed 起 100 个）只在最后评一次并做配对 t。两者都是 CRN
（同一批种子跑所有个体），所以个体之间的差是配对差，种子主效应被消掉。

用法：
    python3 scripts/ga.py --pop 24 --gens 40 --out artifacts/ga
    python3 scripts/ga.py --validate lab/config/ga-best.toml   # 留出集正式评估
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

# ---- 基因组：字段名 → (下界, 上界, 是否整数) --------------------------------
BOUNDS: dict[str, tuple[float, float, bool]] = {
    "pt_favor_rate": (1.0, 4.0, False),
    "rest_vital": (30.0, 70.0, True),
    "checkpoint_margin": (0.0, 10.0, True),
    "act_level_target": (6.0, 10.0, True),
    "status_weights.0": (0.3, 3.0, False),
    "status_weights.1": (0.3, 3.0, False),
    "status_weights.2": (0.3, 3.0, False),
    "status_weights.3": (0.3, 3.0, False),
    "status_weights.4": (0.3, 3.0, False),
    "weights.vital": (0.0, 4.0, False),
    "weights.motivation": (0.0, 80.0, False),
    "weights.friendship": (0.0, 4.0, False),
    "weights.hint": (0.0, 4.0, False),
    "weights.gauge": (0.0, 60.0, False),
    "weights.act_level": (0.0, 400.0, False),
    "weights.checkpoint": (200.0, 2000.0, False),
    "weights.fat": (0.0, 300.0, False),
    "advice.card": (0.2, 2.5, False),
    "advice.npc": (0.0, 2.5, False),
    "advice.train": (0.2, 2.5, False),
    "advice.keiko": (0.2, 2.5, False),
    "advice.level": (0.2, 2.5, False),
    "advice.milestone": (0.2, 2.5, False),
    # 轮次 7 新加的三项：前两项是打分表漏掉的 buff 效果（实测都是噪声 / 负收益，
    # 留给 GA 自己否掉），route 是离线搜出的落位先验强度（≥200 即完全接管落位）
    "advice.status_limit": (0.0, 4.0, False),
    "advice.assist": (0.0, 60.0, False),
    "advice.route": (0.0, 2000.0, False),
    "dynamic.status_catchup": (0.0, 4.0, False),
    "dynamic.act_catchup": (0.0, 2.0, False),
    "dynamic.vital_decay": (0.0, 1.0, False),
    "dynamic.vital_idle": (0.0, 4.0, False),
    "dynamic.keiko_tempo": (0.0, 4.0, False),
}

# 变异步长 = 量程的比例；大旋钮小步走，免得一步跳出盆地
SIGMA = 0.12

FUSION = """[fusion]
year2 = [{ base = 101, target = 601 }, { base = 201, target = 701 }]
year3 = [
    { base = 501, target = 801 },
    { base = 301, target = 901 },
    { base = 401, target = 1001 },
]
"""

SCORE_RE = re.compile(r"^评分\s+(\d+)", re.M)
PT_RE = re.compile(r"^PT\s+(\d+)", re.M)
FIVE_RE = re.compile(r"│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│")
LV_RE = re.compile(r"演目Lv\s*\[([0-9, ]+)\]")


def genome_to_toml(g: dict[str, float]) -> str:
    """基因组 → matsuri_config.toml 覆盖层（[trainer] 是整段替换，必须给全每个键）。"""
    sw = ", ".join(f"{g[f'status_weights.{i}']:.4f}" for i in range(5))
    return f"""{FUSION}
[trainer]
pt_favor_rate = {g['pt_favor_rate']:.4f}
rest_vital = {int(g['rest_vital'])}
checkpoint_margin = {int(g['checkpoint_margin'])}
act_level_target = {int(g['act_level_target'])}
self_race = false
friend_outing_early = true
status_weights = [{sw}]

[trainer.weights]
vital = {g['weights.vital']:.4f}
motivation = {g['weights.motivation']:.4f}
friendship = {g['weights.friendship']:.4f}
hint = {g['weights.hint']:.4f}
gauge = {g['weights.gauge']:.4f}
act_level = {g['weights.act_level']:.4f}
checkpoint = {g['weights.checkpoint']:.4f}
fat = {g['weights.fat']:.4f}

[trainer.advice]
card = {g['advice.card']:.4f}
npc = {g['advice.npc']:.4f}
train = {g['advice.train']:.4f}
keiko = {g['advice.keiko']:.4f}
level = {g['advice.level']:.4f}
milestone = {g['advice.milestone']:.4f}
status_limit = {g['advice.status_limit']:.4f}
assist = {g['advice.assist']:.4f}
route = {g['advice.route']:.4f}

[trainer.dynamic]
status_catchup = {g['dynamic.status_catchup']:.4f}
act_catchup = {g['dynamic.act_catchup']:.4f}
vital_decay = {g['dynamic.vital_decay']:.4f}
vital_idle = {g['dynamic.vital_idle']:.4f}
keiko_tempo = {g['dynamic.keiko_tempo']:.4f}
"""


def play(binary: Path, repo: Path, seed: int, uma: str, cards: str,
         route: str = "") -> tuple[int, dict]:
    """跑一局，返回 (评分, 明细)。评分 0 = 这局没跑出终局块（配置坏了）。"""
    cards_toml = ", ".join(c.strip() for c in cards.split(",") if c.strip())
    (repo / "ai_config.toml").write_text(
        f"[sim]\nseed = {seed}\numa = {uma}\ncards = [{cards_toml}]\n", encoding="utf8")
    env = {**os.environ, "MATSURI_ASSIGN_PIN": route} if route else None
    proc = subprocess.run([str(binary), "matsuri"], cwd=repo, capture_output=True,
                          text=True, errors="replace", timeout=600, env=env)
    out = proc.stdout
    m = SCORE_RE.search(out)
    if not m:
        return 0, {"err": out[-400:] + proc.stderr[-400:]}
    five = FIVE_RE.search(out)
    lv = LV_RE.search(proc.stderr)
    detail = {
        "pt": int(PT_RE.search(out).group(1)) if PT_RE.search(out) else 0,
        "five": [int(x) for x in five.groups()] if five else [],
        "lv": [int(x) for x in lv.group(1).split(",")] if lv else [],
    }
    return int(m.group(1)), detail


def paired_stats(a: list[int], b: list[int]) -> tuple[float, float, int]:
    """同种子配对差：均值 / t 值 / a 赢的局数。GA、BO、CMA-ES 共用一套判据。"""
    paired = [x - y for x, y in zip(a, b)]
    n = len(paired)
    if not n:
        return 0.0, 0.0, 0
    mean = sum(paired) / n
    var = sum((x - mean) ** 2 for x in paired) / n
    return mean, (mean / ((var ** 0.5) / (n ** 0.5)) if var else 0.0), \
        sum(1 for x in paired if x > 0)


def evaluate(g: dict[str, float], repo: Path, binary: Path, seeds: list[int],
             uma: str, cards: str, route: str = "") -> dict:
    """CRN：同一个体跑完整个种子集，取均分。"""
    (repo / "matsuri_config.toml").write_text(genome_to_toml(g), encoding="utf8")
    scores, pts, fives, lvs = [], [], [], []
    for seed in seeds:
        score, detail = play(binary, repo, seed, uma, cards, route)
        if score == 0:
            print(f"  seed {seed} 没跑出终局块: {detail.get('err', '')[:200]}",
                  file=sys.stderr)
            continue
        scores.append(score)
        pts.append(detail["pt"])
        if detail["five"]:
            fives.append(sum(detail["five"]))
        if detail["lv"]:
            lvs.append(detail["lv"])
    for f in ("matsuri_config.toml", "ai_config.toml"):
        (repo / f).unlink(missing_ok=True)
    if not scores:
        return {"fitness": 0.0, "n": 0}
    n = len(scores)
    return {
        "fitness": sum(scores) / n,
        "n": n,
        "min": min(scores),
        "max": max(scores),
        "pt": sum(pts) / n,
        "five": sum(fives) / len(fives) if fives else 0.0,
        "lv": [sum(x[i] for x in lvs) / len(lvs) for i in range(5)] if lvs else [],
        "scores": scores,
    }


def clamp_mutate(g: dict[str, float], rng: random.Random, sigma: float = SIGMA) -> dict[str, float]:
    """高斯变异：每个基因独立以 1/基因数 概率触发（旧仓库 genetic_optimizer.rs 的 pm 口径），
    步长 sigma × 量程。sigma 由主循环按代调度（0.15 → 0.03，停滞回升），不再写死。"""
    child = dict(g)
    per = 1.0 / len(BOUNDS)
    for key, (lo, hi, is_int) in BOUNDS.items():
        if rng.random() < per:
            span = hi - lo
            child[key] = min(hi, max(lo, child[key] + rng.gauss(0.0, sigma * span)))
            if is_int:
                child[key] = float(round(child[key]))
    return child


def crossover(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict[str, float]:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in BOUNDS}


def genome_hash(g: dict[str, float]) -> str:
    """基因组指纹：评估缓存键。精英跨代原样保留、变异偶尔撞回同一个体时，
    不必重跑同一批种子（旧仓库 genetic_optimizer.rs::genome_hash 的同一动机）。"""
    return hashlib.blake2b(
        json.dumps({k: round(g[k], 6) for k in BOUNDS}, sort_keys=True).encode(),
        digest_size=8).hexdigest()


def sigma_for_gen(gen: int, gens: int, sigma0: float, sigma_min: float,
                  reset: bool) -> float:
    """σ 调度：停滞代回升到 sigma0，否则按代线性衰减到 sigma_min。
    照抄旧仓库 GaOptimizer::sigma_for_gen —— 我原来的版本 σ 全程写死 0.12，
    前期不够大跳不出盆地、后期又太大收敛不住。"""
    if reset or gens <= 1:
        return sigma0
    t = gen / (gens - 1)
    return sigma0 - (sigma0 - sigma_min) * t


def seed_genome() -> dict[str, float]:
    """起点 = 当前冠军档 d-status（其余旋钮取 d-status 的值，未开的 dynamic 取 0）。"""
    g = {
        "pt_favor_rate": 2.2, "rest_vital": 48.0, "checkpoint_margin": 4.0,
        "act_level_target": 10.0,
        "weights.vital": 1.4, "weights.motivation": 35.0, "weights.friendship": 1.0,
        "weights.hint": 1.0, "weights.gauge": 28.0, "weights.act_level": 180.0,
        "weights.checkpoint": 1000.0, "weights.fat": 130.0,
        "advice.card": 1.1, "advice.npc": 0.6, "advice.train": 1.0,
        "advice.keiko": 1.3, "advice.level": 1.3, "advice.milestone": 1.1,
        "advice.status_limit": 0.0, "advice.assist": 0.0, "advice.route": 500.0,
        "dynamic.status_catchup": 1.0, "dynamic.act_catchup": 0.0,
        "dynamic.vital_decay": 0.0, "dynamic.vital_idle": 0.0,
        "dynamic.keiko_tempo": 0.0,
    }
    for i in range(5):
        g[f"status_weights.{i}"] = 1.0
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=40, help="精评种子数（从 --start-seed 起）")
    ap.add_argument("--screen-seeds", type=int, default=12,
                    help="初筛种子数（精评种子集的前 N 个，CRN 子集）")
    ap.add_argument("--promote", type=int, default=6,
                    help="每代按初筛值晋升精评的名额（旧仓库 = elitism）")
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--holdout-seed", type=int, default=61544)
    ap.add_argument("--holdout-count", type=int, default=100)
    ap.add_argument("--uma", default="102102")
    ap.add_argument("--cards", default="303124,303114,303004,303224,303204,303234")
    ap.add_argument("--pin-route", default="",
                    help="钉住教示路线（原样传 MATSURI_ASSIGN_PIN）；GA 只搜权重，路线固定")
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--out", default="artifacts/ga")
    ap.add_argument("--elite", type=int, default=4, help="精英保留数")
    ap.add_argument("--tournament-k", type=int, default=3)
    ap.add_argument("--cx-rate", type=float, default=0.9, help="均匀交叉率")
    ap.add_argument("--sigma0", type=float, default=0.15, help="高斯变异 σ 初值（×量程）")
    ap.add_argument("--sigma-min", type=float, default=0.03, help="σ 下界")
    ap.add_argument("--stagnation-gens", type=int, default=5,
                    help="连续 N 代精评最优无提升 → σ 回升一次")
    ap.add_argument("--rng-seed", type=int, default=20261102)
    ap.add_argument("--validate", default="", help="只评估这个覆盖层文件（留出集），不搜索")
    ap.add_argument("--start", default="",
                    help="起点覆盖层文件（如 lab/config/route-prior.toml）；默认用 seed_genome()")
    args = ap.parse_args()

    repo = Path.cwd()
    binary = (repo / args.binary).resolve()
    if not binary.exists():
        print(f"找不到可执行文件: {binary}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.rng_seed)

    if args.validate:
        # 留出集正式评估：直接把已有覆盖层当基因组用（解析回 dict）
        text = Path(args.validate).read_text(encoding="utf8")
        g = parse_overlay(text)
        seeds = [args.holdout_seed + i for i in range(args.holdout_count)]
        r = evaluate(g, repo, binary, seeds, args.uma, args.cards, args.pin_route)
        print(json.dumps({k: v for k, v in r.items() if k != "scores"},
                         ensure_ascii=False, indent=2))
        return 0

    full_seeds = [args.start_seed + i for i in range(args.seeds)]
    screen_seeds = full_seeds[:args.screen_seeds]
    holdout = [args.holdout_seed + i for i in range(args.holdout_count)]
    print(f"初筛种子 {screen_seeds[0]}–{screen_seeds[-1]}（{len(screen_seeds)} 个） · "
          f"精评种子 {full_seeds[0]}–{full_seeds[-1]}（{len(full_seeds)} 个） · "
          f"留出种子 {holdout[0]}–{holdout[-1]}（{len(holdout)} 个）", flush=True)

    # 评估缓存：键 = (基因组指纹, 级别)。精英跨代原样保留，命中即免跑。
    cache: dict[tuple[str, str], dict] = {}
    stats = {"screen": 0, "full": 0, "hits": 0, "games": 0}

    def ev(g: dict[str, float], level: str) -> dict:
        seeds = screen_seeds if level == "s" else full_seeds
        key = (genome_hash(g), level)
        hit = cache.get(key)
        if hit is not None:
            stats["hits"] += 1
            return hit
        r = evaluate(g, repo, binary, seeds, args.uma, args.cards, args.pin_route)
        cache[key] = r
        stats["full" if level == "f" else "screen"] += 1
        stats["games"] += len(seeds)
        return r

    # 基线：起点基因组（默认 d-status，或 --start 指定的覆盖层）在训练集上的分数，
    # 用来算 GA 到底赚了多少
    base_label = Path(args.start).stem if args.start else "d-status"
    if args.start:
        base_g = parse_overlay(Path(args.start).read_text(encoding="utf8"))
    else:
        base_g = seed_genome()
    base_r = ev(base_g, "f")
    print(f"基线 {base_label} 精评均分 {base_r['fitness']:.1f}（n={base_r['n']}）", flush=True)

    # 个体 = 基因组 + 两级评估位（None = 尚未评）。display 优先精评值，
    # 与旧仓库 Individual::display_fitness 同口径：跨代可比的只有精评值。
    pop: list[dict] = [{"g": base_g, "s": None, "f": None}]
    while len(pop) < args.pop:
        pop.append({"g": clamp_mutate(base_g, rng, args.sigma0), "s": None, "f": None})

    def display(ind: dict) -> float:
        r = ind["f"] or ind["s"]
        return r["fitness"] if r else float("-inf")

    history = []
    best_f: tuple[float, dict] | None = None   # 全程精评最优（跨代入榜）
    stagnation = 0
    resets = 0
    t0 = time.time()
    for gen in range(args.gens):
        sigma = sigma_for_gen(gen, args.gens, args.sigma0, args.sigma_min,
                              stagnation >= args.stagnation_gens)
        if stagnation >= args.stagnation_gens:
            stagnation = 0
            resets += 1
        prev_best = best_f[0] if best_f else None

        # 1) 初筛：尚无任何评估值的个体全部过初筛
        for ind in pop:
            if ind["s"] is None and ind["f"] is None:
                ind["s"] = ev(ind["g"], "s")

        # 2) 精评晋升：按当前值取前 --promote 名补精评（随机 tie-break 保序，
        #    避免 sort 比较器违反全序——旧仓库在这里踩过 Rust panic 的坑）
        tb = [rng.random() for _ in pop]
        order = sorted(range(len(pop)), key=lambda i: (-display(pop[i]), -tb[i], i))
        for i in order[:max(1, min(args.promote, len(pop)))]:
            if pop[i]["f"] is None:
                pop[i]["f"] = ev(pop[i]["g"], "f")

        # 3) 排序刷新：精评优先，确定性 tie-break（基因组指纹小者优先）
        order = sorted(range(len(pop)),
                       key=lambda i: (-display(pop[i]), genome_hash(pop[i]["g"])))
        champ = pop[order[0]]
        if champ["f"] is not None and (best_f is None or champ["f"]["fitness"] > best_f[0]):
            best_f = (champ["f"]["fitness"], champ["g"])
        improved = best_f is not None and (prev_best is None or best_f[0] > prev_best + 1e-9)
        stagnation = 0 if improved else stagnation + 1

        # 4) 过拟合哨兵：本代冠军的留出集小样（12 局）−精评值。不参与选择。
        champ_h = evaluate(champ["g"], repo, binary, holdout[:args.screen_seeds],
                           args.uma, args.cards, args.pin_route)
        gap = (champ["f"]["fitness"] - champ_h["fitness"]) if champ["f"] else None
        history.append({
            "gen": gen, "sigma": round(sigma, 4),
            "best_screen": round(display(champ), 1),
            "best_full": round(champ["f"]["fitness"], 1) if champ["f"] else None,
            "global_best_full": round(best_f[0], 1) if best_f else None,
            "mean": round(sum(display(pop[i]) for i in range(len(pop))) / len(pop), 1),
            "full_count": sum(1 for i in pop if i["f"] is not None),
            "stagnation": stagnation, "resets": resets,
            "holdout12": round(champ_h["fitness"], 1),
            "overfit_gap": round(gap, 1) if gap is not None else None,
            "cache_hits": stats["hits"], "games": stats["games"],
            "base": round(base_r["fitness"], 1), "elapsed_s": round(time.time() - t0),
        })
        print(f"gen {gen:>2} σ{sigma:.3f} best "
              f"{history[-1]['best_full'] if champ['f'] else history[-1]['best_screen']} "
              f"(全局 {history[-1]['global_best_full']}) "
              f"h12 {history[-1]['holdout12']} gap {history[-1]['overfit_gap']} "
              f"停滞{stagnation}/{args.stagnation_gens} 局{stats['games']} "
              f"· {time.time() - t0:.0f}s", flush=True)
        (out / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf8")
        if gen == args.gens - 1:
            break

        # 5) 繁殖：精英直接保留（带评估位，跨代命中缓存），其余锦标赛 + 均匀交叉 + 高斯变异
        new_pop = [pop[i] for i in order[:min(args.elite, len(pop))]]
        while len(new_pop) < args.pop:
            a = pop[tournament(order, pop, rng, args.tournament_k)]["g"]
            b = pop[tournament(order, pop, rng, args.tournament_k)]["g"]
            child = crossover(a, b, rng) if rng.random() < args.cx_rate else dict(a)
            new_pop.append({"g": clamp_mutate(child, rng, sigma), "s": None, "f": None})
        pop = new_pop

    # 全程精评最优（不是末代最优——末代可能只是初筛值高）
    champion = best_f[1]
    champ_r = cache[(genome_hash(champion), "f")]
    (out / "best.toml").write_text(genome_to_toml(champion), encoding="utf8")
    (out / "best.json").write_text(json.dumps(
        {"genome": champion, "train": {k: v for k, v in champ_r.items() if k != "scores"},
         "baseline_train": {k: v for k, v in base_r.items() if k != "scores"},
         "budget": stats, "stagnation_resets": resets},
        ensure_ascii=False, indent=2), encoding="utf8")
    print(f"\n冠军精评 {champ_r['fitness']:.1f}（基线 {base_r['fitness']:.1f}，"
          f"Δ{champ_r['fitness'] - base_r['fitness']:+.1f}）· 共 {stats['games']} 局"
          f"（缓存命中 {stats['hits']}）", flush=True)

    # 留出集：训练种子一个都没参与过，这才是「GA 有没有真本事」的判据
    print("留出集评估冠军 …", flush=True)
    champ_h = evaluate(champion, repo, binary, holdout, args.uma, args.cards, args.pin_route)
    print("留出集评估基线 …", flush=True)
    base_h = evaluate(base_g, repo, binary, holdout, args.uma, args.cards, args.pin_route)
    mean, _t, wins = paired_stats(champ_h["scores"], base_h["scores"])
    report = {
        "champion_holdout": {k: v for k, v in champ_h.items() if k != "scores"},
        "baseline_holdout": {k: v for k, v in base_h.items() if k != "scores"},
        "paired_delta": round(mean, 1), "paired_t": round(_t, 2),
        "champion_wins": wins, "n": len(champ_h["scores"]),
    }
    (out / "holdout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def tournament(order: list[int], pop: list[dict], rng: random.Random, k: int = 3) -> int:
    """锦标赛选择：从已排好的 order 里随机抽 k 个，取适应度最高者。"""
    cands = rng.sample(order, min(k, len(order)))
    return max(cands, key=lambda i: (pop[i]["f"] or pop[i]["s"])["fitness"])


def parse_overlay(text: str) -> dict[str, float]:
    """把已有的 matsuri_config.toml 覆盖层解析回基因组（--validate/--start 用）。"""
    g = seed_genome()
    # 轮次 7 新增的三个旋钮：文件里没写就按「未开启」处理。否则老覆盖层
    # （ga-best.toml 没有 route 键）会继承 seed_genome 的 route=500，
    # 等于悄悄改写旧基线，历史数字再也复现不出来。
    for key in ("advice.status_limit", "advice.assist", "advice.route"):
        g[key] = 0.0
    section = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            section = s.strip("[]")
            continue
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, value = s.partition("=")
        key, value = key.strip(), value.strip()
        if key == "status_weights":
            vals = [float(x) for x in value.strip("[]").split(",")]
            for i, v in enumerate(vals):
                g[f"status_weights.{i}"] = v
            continue
        full = {"trainer": key, "trainer.weights": f"weights.{key}",
                "trainer.advice": f"advice.{key}",
                "trainer.dynamic": f"dynamic.{key}"}.get(section)
        if full in BOUNDS:
            g[full] = float(value)
    return g


if __name__ == "__main__":
    sys.exit(main())
