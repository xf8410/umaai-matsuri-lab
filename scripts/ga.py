"""遗传算法迭代 trainer 权重（骏大祭）。

为什么现在能上 GA：单局 0.15s，100 局 12s，所以「一代 16 个个体 × 24 个种子」只要 1 分钟，
GA 完全跑在本地。仓库 CI 的 100 局硬上限不受影响——GA 只用训练种子集搜索，
搜完的冠军再用完整 100 局（含留出集）在 CI 上做一次正式记录，避免用调参种子自证。

搜索空间 = `lab/config/d-status.toml` 的 `[trainer]` 全部数值旋钮（含 5 个 dynamic），
`[fusion]` 暂时固定（120 种组合是另一个离散问题，先不掺进连续搜索里）。

防过拟合：训练种子 61444–61477（24 个），留出种子 61544–61643（100 个）只在最后评一次。
两者都是 CRN（同一批种子跑所有个体），所以个体之间的差是配对差，种子主效应被消掉。

用法：
    python3 scripts/ga.py --pop 16 --gens 12 --seeds 24 --out artifacts/ga
    python3 scripts/ga.py --validate lab/config/ga-best.toml   # 留出集正式评估
"""

from __future__ import annotations

import argparse
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


def clamp_mutate(g: dict[str, float], rng: random.Random, rate: float = 1.0) -> dict[str, float]:
    child = dict(g)
    # 每个基因独立以 rate/len 概率变异，保证平均改动 1–2 个旋钮（单变量可归因）
    per = max(rate / len(BOUNDS), 0.02)
    for key, (lo, hi, is_int) in BOUNDS.items():
        if rng.random() < per * 4:
            span = hi - lo
            child[key] = min(hi, max(lo, child[key] + rng.gauss(0.0, SIGMA * span)))
            if is_int:
                child[key] = float(round(child[key]))
    return child


def crossover(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict[str, float]:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in BOUNDS}


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
        "dynamic.status_catchup": 1.0, "dynamic.act_catchup": 0.0,
        "dynamic.vital_decay": 0.0, "dynamic.vital_idle": 0.0,
        "dynamic.keiko_tempo": 0.0,
    }
    for i in range(5):
        g[f"status_weights.{i}"] = 1.0
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=24, help="训练种子数（从 --start-seed 起）")
    ap.add_argument("--start-seed", type=int, default=61444)
    ap.add_argument("--holdout-seed", type=int, default=61544)
    ap.add_argument("--holdout-count", type=int, default=100)
    ap.add_argument("--uma", default="102102")
    ap.add_argument("--cards", default="303124,303114,303004,303224,303204,303234")
    ap.add_argument("--pin-route", default="",
                    help="钉住教示路线（原样传 MATSURI_ASSIGN_PIN）；GA 只搜权重，路线固定")
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--out", default="artifacts/ga")
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--rng-seed", type=int, default=20261102)
    ap.add_argument("--validate", default="", help="只评估这个覆盖层文件（留出集），不搜索")
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

    train = [args.start_seed + i for i in range(args.seeds)]
    holdout = [args.holdout_seed + i for i in range(args.holdout_count)]
    print(f"训练种子 {train[0]}–{train[-1]}（{len(train)} 个） · "
          f"留出种子 {holdout[0]}–{holdout[-1]}（{len(holdout)} 个）", flush=True)

    # 基线：起点基因组（= 冠军档 d-status）在训练集上的分数，用来算 GA 到底赚了多少
    base_g = seed_genome()
    base_r = evaluate(base_g, repo, binary, train, args.uma, args.cards, args.pin_route)
    print(f"基线 d-status 训练集均分 {base_r['fitness']:.1f}（n={base_r['n']}）", flush=True)

    pop = [base_g] + [clamp_mutate(base_g, rng, rate=1.5) for _ in range(args.pop - 1)]
    fit: list[dict] = [None] * len(pop)  # type: ignore[list-item]
    history = []
    t0 = time.time()
    for gen in range(args.gens):
        for i, g in enumerate(pop):
            if fit[i] is None:
                fit[i] = evaluate(g, repo, binary, train, args.uma, args.cards, args.pin_route)
        order = sorted(range(len(pop)), key=lambda i: -fit[i]["fitness"])
        best = fit[order[0]]
        history.append({"gen": gen, "best": round(best["fitness"], 1),
                        "mean": round(sum(fit[i]["fitness"] for i in range(len(pop))) / len(pop), 1),
                        "base": round(base_r["fitness"], 1)})
        print(f"gen {gen:>2} best {best['fitness']:.1f} "
              f"(Δ{best['fitness'] - base_r['fitness']:+.1f}) "
              f"mean {history[-1]['mean']:.1f} · {time.time() - t0:.0f}s", flush=True)
        (out / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf8")
        if gen == args.gens - 1:
            break
        # 精英保留 + 锦标赛选择 + 交叉/变异
        new_pop = [pop[i] for i in order[:args.elite]]
        new_fit: list[dict] = [fit[i] for i in order[:args.elite]]
        while len(new_pop) < args.pop:
            a, b = (pop[tournament(order, fit, rng)] for _ in range(2))
            child = crossover(a, b, rng) if rng.random() < 0.7 else dict(a)
            child = clamp_mutate(child, rng, rate=1.0 if rng.random() < 0.7 else 3.0)
            new_pop.append(child)
            new_fit.append(None)  # type: ignore[arg-type]
        pop, fit = new_pop, new_fit

    order = sorted(range(len(pop)), key=lambda i: -fit[i]["fitness"])
    champion = pop[order[0]]
    champ_r = fit[order[0]]
    (out / "best.toml").write_text(genome_to_toml(champion), encoding="utf8")
    (out / "best.json").write_text(json.dumps(
        {"genome": champion, "train": {k: v for k, v in champ_r.items() if k != "scores"},
         "baseline_train": {k: v for k, v in base_r.items() if k != "scores"}},
        ensure_ascii=False, indent=2), encoding="utf8")
    print(f"\n冠军训练集 {champ_r['fitness']:.1f}（基线 {base_r['fitness']:.1f}，"
          f"Δ{champ_r['fitness'] - base_r['fitness']:+.1f}）", flush=True)

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


def tournament(order: list[int], fit: list[dict], rng: random.Random, k: int = 3) -> int:
    cands = rng.sample(order, min(k, len(order)))
    return max(cands, key=lambda i: fit[i]["fitness"])


def parse_overlay(text: str) -> dict[str, float]:
    """把已有的 matsuri_config.toml 覆盖层解析回基因组（--validate 用）。"""
    g = seed_genome()
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
