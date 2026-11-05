"""PBT（Population Based Training）：把四个优化器当一个种群跑，周期性交换权重块。

7.2 里 GA / BO / CMA-ES / SA 是**串行各跑一遍再比**，各自烧各自的预算。PBT 改成：
一个种群里每个成员**自带一个优化器算子**（这就是 PBT 意义上的「超参」），
每一代 评估 → 排序 → 淘汰复制 → 算子探索，算子和它的步长随基因组一起被继承。

三件事是 PBT 的核心，都实现了：

1. **ready/reap（淘汰复制）**：垫底的 `--reap` 个成员被替换成随机挑的优胜成员**副本**，
   连算子和步长一起拷过来，再把步长按 `--hp-perturb` 乘一个扰动（继承 + 探索超参空间）。
2. **explore（算子探索）**：每个成员用自己的算子改自己的基因组
   - `ga`：`clamp_mutate(rate=σ)`，σ 是成员的变异强度
   - `es`：单位立方体里的 (1+1)-ES 高斯步，σ 按 1/5 成功法则自适应（成功 ×1.15，失败 ×0.85）
   - `sa`：Metropolis 步，温度 T 随代数几何降温，Δ<0 按 exp(Δ/T) 接受
   - `bo`：不自己扰动，而是从**全种群共享**的 RBF GP + EI 代理上取建议点
3. **周期性交换权重块**：每 `--swap-every` 代，把一个优胜成员的**连续基因块**
   （`--swap-block` 个相邻旋钮，按 BOUNDS 顺序）整块拼进另一个成员。
   这是块级交叉，比逐基因交叉更能保住「一组旋钮的协同」——例如
   `pt_favor_rate` 和 `weights.checkpoint` 是同一个量纲口径下的搭配。

代理模型是全种群共享的（BO 成员把大家跑过的所有点都喂进 GP），这是 PBT + 代理的常规组合：
昂贵的黑盒评估一次都不浪费。

判据与前几轮一致：训练种子只用 `--start-seed` 起的 `--seeds` 个（CRN 同种子配对），
留出种子 `--holdout-seed` 起的 100 局**全程不参与搜索**，最后只评一次，报配对 t。

用法：
    python3 scripts/pbt.py --pop 8 --gens 10 --seeds 30 --start-seed 62500 \
        --holdout-seed 62600 --start lab/config/ga-r2.toml --out artifacts/pbt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bopt import KEYS, GP, expected_improvement, from_unit, to_unit  # noqa: E402
from ga import BOUNDS, evaluate, genome_to_toml, parse_overlay, paired_stats  # noqa: E402
from ga import seed_genome  # noqa: E402

OPS = ["ga", "es", "sa", "bo"]

def clamp_mutate_rate(g, rng, rate: float = 1.0):
    """旧版 ga.clamp_mutate 语义（pbt 的 hp 按此设计）：rate 控制变异**概率**
    （平均每代改 4×rate 个旋钮），步长固定 0.12×量程。新版 clamp_mutate 改成了
    sigma 控制步长、概率固定 1/len，直接把 hp 传 sigma 会语义错位。"""
    child = dict(g)
    per = max(rate / len(BOUNDS), 0.02)
    for key, (lo, hi, is_int) in BOUNDS.items():
        if rng.random() < per * 4:
            span = hi - lo
            child[key] = min(hi, max(lo, child[key] + rng.gauss(0.0, 0.12 * span)))
            if is_int:
                child[key] = float(round(child[key]))
    return child

SIGMA = 0.08  # ES / SA 在单位立方体里的基准步长


class Member:
    __slots__ = ("g", "op", "hp", "fitness", "prev", "g_before", "result", "dirty", "tag")

    def __init__(self, g: dict[str, float], op: str, hp: float, tag: str = ""):
        self.g = g
        self.op = op
        self.hp = hp            # ga: 变异强度 · es: σ · sa: 温度系数 · bo: 未用
        self.fitness = 0.0
        self.prev: float | None = None   # 走这一步之前的分数（ES 1/5 法则、SA 接受判定要用）
        self.g_before: dict[str, float] | None = None   # 走这一步之前的基因组（SA 回退用）
        self.result: dict = {}
        self.dirty = True
        self.tag = tag

    def clone(self, tag: str) -> "Member":
        m = Member(dict(self.g), self.op, self.hp, tag)
        m.fitness, m.result, m.dirty = self.fitness, self.result, False
        m.prev, m.g_before = None, None
        return m


def hp_bounds(op: str) -> tuple[float, float]:
    return {"ga": (0.3, 4.0), "es": (0.02, 0.30), "sa": (0.3, 4.0), "bo": (1.0, 1.0)}[op]


def sa_temp(gen: int, gens: int, hp: float, args) -> float:
    """SA 温度：几何降温 1200 → 60，再乘成员自己继承来的系数。"""
    ratio = args.sa_t1 / args.sa_t0
    return max(1e-3, args.sa_t0 * ratio ** (gen / max(gens - 1, 1))) * hp


def step_member(m: Member, rng: random.Random, gen: int, gens: int,
                hist_X: list[np.ndarray], hist_y: list[float], args) -> None:
    """explore：按成员自己的算子改基因组（原地）。改之前先把旧基因组/旧分数存下来，
    下一代的评估阶段用得上（ES 的 1/5 成功法则、SA 的 Metropolis 回退）。"""
    m.prev, m.g_before = m.fitness, dict(m.g)
    if m.op == "ga":
        m.g = clamp_mutate_rate(m.g, rng, rate=m.hp)
    elif m.op == "es":
        # 逐维独立高斯步（单位立方体口径），σ = m.hp 由成功法则自适应
        x = to_unit(m.g) + np.array([rng.gauss(0.0, m.hp) for _ in KEYS])
        m.g = from_unit(np.clip(x, 0.0, 1.0))
    elif m.op == "sa":
        temp = sa_temp(gen, gens, m.hp, args)
        step = SIGMA * (0.3 + temp / args.sa_t0)
        dims = rng.sample(range(len(KEYS)), args.sa_dims)
        x = to_unit(m.g)
        for i in dims:
            x[i] = float(np.clip(x[i] + rng.gauss(0.0, step), 0.0, 1.0))
        m.g = from_unit(x)
        m.tag = f"sa T={temp:.0f}"
    elif m.op == "bo" and len(hist_X) >= 8:
        X = np.array(hist_X)
        y = np.array(hist_y)
        gp = GP()
        try:
            gp.fit(X, y)
        except (np.linalg.LinAlgError, ValueError):
            m.g = clamp_mutate_rate(m.g, rng, rate=1.0)
            return
        npr = args.npr
        cand = np.clip(np.vstack([npr.random((512, len(KEYS))),
                                  X[npr.integers(len(X))] + npr.normal(0, 0.08, (64, len(KEYS)))]),
                       0.0, 1.0)
        mean, sd = gp.predict(cand)
        ei = expected_improvement(mean, sd, float(y.max()))
        m.g = from_unit(cand[int(np.argmax(ei))])
        m.tag = f"bo ell={gp.ell:.2f}"
    else:
        m.g = clamp_mutate_rate(m.g, rng, rate=1.0)
    m.dirty = True


def post_step_adapt(m: Member, rng: random.Random, gen: int, gens: int, args) -> str:
    """评估完一代之后：ES 按 1/5 成功法则调 σ，SA 按 Metropolis 决定是否回退。

    SA 的接受判定必须**延后**到分数出来之后 —— PBT 里每代每个成员只跑一次评估，
    所以是「先走一步、看到分数再决定要不要退回原地」，等价于把 Metropolis 判据
    用在同一个候选上，不额外花局数。
    """
    if m.prev is None or m.g_before is None:
        return ""
    if m.op == "es":
        lo, hi = hp_bounds("es")
        m.hp = min(hi, max(lo, m.hp * (1.15 if m.fitness > m.prev else 0.85)))
        m.prev, m.g_before = None, None
        return f"es σ={m.hp:.3f}"
    if m.op == "sa" and m.fitness < m.prev:
        temp = sa_temp(max(gen - 1, 0), gens, m.hp, args)
        if rng.random() > math.exp((m.fitness - m.prev) / max(temp, 1e-9)):
            m.g, m.fitness, m.dirty = m.g_before, m.prev, False
            m.prev, m.g_before = None, None
            return "sa 拒绝，退回"
    m.prev, m.g_before = None, None
    return ""


def swap_blocks(pop: list[Member], order: list[int], rng: random.Random, args) -> int:
    """周期性权重块交换：优胜成员的连续基因块 → 另一个成员。返回实际交换次数。"""
    n = len(KEYS)
    width = min(args.swap_block, n)
    done = 0
    for _ in range(args.swap_pairs):
        donor = pop[rng.choice(order[:max(1, len(order) // 2)])]
        recv = pop[rng.choice(order[max(1, len(order) // 2):])]
        if donor is recv:
            continue
        start = rng.randrange(0, n - width + 1)
        block = KEYS[start:start + width]
        recv.g = dict(recv.g)
        for key in block:
            lo, hi, is_int = BOUNDS[key]
            v = donor.g[key]
            # 块内基因带一点抖动，避免变成纯克隆（PBT 的 explore 要求）
            v = min(hi, max(lo, v + rng.gauss(0.0, 0.15 * (hi - lo))))
            recv.g[key] = float(round(v)) if is_int else v
        recv.dirty = True
        recv.tag = f"swap[{start}:{start + width}]"
        done += 1
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--start-seed", type=int, default=62500)
    ap.add_argument("--holdout-seed", type=int, default=62600)
    ap.add_argument("--holdout-count", type=int, default=100)
    ap.add_argument("--reap", type=int, default=2, help="每代淘汰（被优胜副本替换）几个成员")
    ap.add_argument("--hp-perturb", type=float, default=0.25, help="继承超参时的乘性扰动幅度")
    ap.add_argument("--swap-every", type=int, default=2, help="每隔几代做一次权重块交换")
    ap.add_argument("--swap-block", type=int, default=5, help="每次交换的连续基因数")
    ap.add_argument("--swap-pairs", type=int, default=2, help="每次交换几对")
    ap.add_argument("--sa-t0", type=float, default=1200.0)
    ap.add_argument("--sa-t1", type=float, default=60.0)
    ap.add_argument("--sa-dims", type=int, default=4)
    ap.add_argument("--uma", default="102102")
    ap.add_argument("--cards", default="303124,303114,303004,303224,303204,303234")
    ap.add_argument("--pin-route", default="", help="默认不钉位：落位先验已在 genome 里（advice.route）")
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--start", default="lab/config/ga-r2.toml")
    ap.add_argument("--out", default="artifacts/pbt")
    ap.add_argument("--rng-seed", type=int, default=20261103)
    args = ap.parse_args()

    repo = Path.cwd()
    binary = (repo / args.binary).resolve()
    if not binary.exists():
        print(f"找不到可执行文件: {binary}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.rng_seed)
    # BO 成员要在单位立方体里撒候选点，用独立的 numpy 生成器（不复用 random.Random）
    args.npr = np.random.default_rng(args.rng_seed + 1)

    train = [args.start_seed + i for i in range(args.seeds)]
    holdout = [args.holdout_seed + i for i in range(args.holdout_count)]
    base_g = (parse_overlay(Path(args.start).read_text(encoding="utf8")) if args.start
              else seed_genome())
    print(f"训练种子 {train[0]}–{train[-1]} · 留出 {holdout[0]}–{holdout[-1]}（{len(holdout)} 局）",
          flush=True)

    # 初始种群：起点基因组 + 变异体，算子轮流分配（每个算子至少有一个成员）
    pop: list[Member] = []
    for i in range(args.pop):
        op = OPS[i % len(OPS)]
        lo, hi = hp_bounds(op)
        g = dict(base_g) if i == 0 else clamp_mutate_rate(base_g, rng, rate=1.5)
        pop.append(Member(g, op, float(np.exp(rng.uniform(math.log(lo), math.log(hi)))),
                          tag="seed" if i == 0 else "init"))

    base_r = evaluate(base_g, repo, binary, train, args.uma, args.cards, args.pin_route)
    print(f"起点 {Path(args.start).stem or 'seed_genome'} 训练集均分 "
          f"{base_r['fitness']:.1f}（n={base_r['n']}）", flush=True)

    hist_X: list[np.ndarray] = []
    hist_y: list[float] = []
    history, t0 = [], time.time()
    best_ever = (base_r["fitness"], dict(base_g), "start")

    for gen in range(args.gens):
        # ---- 评估（只有 dirty 的成员才实跑；同基因组 + CRN 同种子必得同分，不必重算） ----
        for m in pop:
            if not m.dirty:
                continue
            r = evaluate(m.g, repo, binary, train, args.uma, args.cards, args.pin_route)
            m.result, m.fitness, m.dirty = r, r["fitness"], False
            hist_X.append(to_unit(m.g))
            hist_y.append(m.fitness)
            if m.fitness > best_ever[0]:
                best_ever = (m.fitness, dict(m.g), f"gen{gen}:{m.op}")
        notes = [n for m in pop if (n := post_step_adapt(m, rng, gen, args.gens, args))]
        if notes:
            print(f"  步长/接受：{', '.join(notes)}", flush=True)
        order = sorted(range(len(pop)), key=lambda i: -pop[i].fitness)
        best, mean = pop[order[0]].fitness, sum(m.fitness for m in pop) / len(pop)
        census: dict[str, int] = {}
        for m in pop:
            census[m.op] = census.get(m.op, 0) + 1
        history.append({"gen": gen, "best": round(best, 1), "mean": round(mean, 1),
                        "base": round(base_r["fitness"], 1),
                        "ops": census, "evals": len(hist_y),
                        "seconds": round(time.time() - t0, 1)})
        print(f"gen {gen:>2} best {best:.1f} (Δ{best - base_r['fitness']:+.1f}) mean {mean:.1f} · "
              f"算子 {census} · 评估 {len(hist_y)} 次 · {time.time() - t0:.0f}s", flush=True)
        (out / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2),
                                         encoding="utf8")

        if gen == args.gens - 1:
            break

        # ---- reap：垫底成员换成优胜成员的副本（基因组 + 算子 + 步长一起继承） ----
        for rank in order[len(pop) - args.reap:]:
            src = pop[rng.choice(order[:max(1, len(order) // 2)])]
            heir = src.clone(tag=f"copy:{src.op}")
            lo, hi = hp_bounds(src.op)
            factor = 1.0 + rng.uniform(-args.hp_perturb, args.hp_perturb)
            heir.hp = min(hi, max(lo, src.hp * factor))
            # 基因组没变 → 训练集分数照抄，不必重跑（dirty=False）；
            # 继承来的步长会在下面的 explore 阶段生效。
            pop[rank] = heir
        order = sorted(range(len(pop)), key=lambda i: -pop[i].fitness)

        # ---- 周期性权重块交换 ----
        if args.swap_every and (gen + 1) % args.swap_every == 0:
            n_swap = swap_blocks(pop, order, rng, args)
            print(f"  权重块交换 {n_swap} 对（块宽 {args.swap_block} 个基因）", flush=True)

        # ---- explore：每个成员用自己的算子走一步 ----
        for i in range(len(pop)):
            step_member(pop[i], rng, gen, args.gens, hist_X, hist_y, args)

    champ_fitness, champ_g, champ_tag = best_ever
    (out / "best.toml").write_text(genome_to_toml(champ_g), encoding="utf8")
    print(f"\n种群历史最优 训练集 {champ_fitness:.1f}（{champ_tag}，"
          f"起点 {base_r['fitness']:.1f}，Δ{champ_fitness - base_r['fitness']:+.1f}）", flush=True)

    print("留出集评估冠军 …", flush=True)
    champ_h = evaluate(champ_g, repo, binary, holdout, args.uma, args.cards, args.pin_route)
    print("留出集评估起点 …", flush=True)
    base_h = evaluate(base_g, repo, binary, holdout, args.uma, args.cards, args.pin_route)
    mean, t, wins = paired_stats(champ_h["scores"], base_h["scores"])
    report = {
        "champion_tag": champ_tag,
        "champion_holdout": {k: v for k, v in champ_h.items() if k != "scores"},
        "baseline_holdout": {k: v for k, v in base_h.items() if k != "scores"},
        "paired_delta": round(mean, 1), "paired_t": round(t, 2),
        "champion_wins": wins, "n": len(champ_h["scores"]),
        "evals": len(hist_y), "history": history, "genome": champ_g,
    }
    (out / "holdout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "genome")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
