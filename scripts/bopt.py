"""贝叶斯优化 / CMA-ES 调手写逻辑权重（骏大祭）。

为什么要这两个，而不是继续用 GA：
- 我们的目标函数是**昂贵黑盒**：一次评估 = 30 局模拟 ≈ 4.5s，29 维连续、有噪声、无梯度。
  GA 只用「谁分高谁活」的排序信息，一次评估的信息量被浪费掉大半；
  贝叶斯优化用**代理模型（GP）**把所有历史评估拟合成后验，再用采集函数（EI）决定下一个点，
  同样的评估预算能多榨出很多——这正是上一版仓库里缺的一块（那边只有 Bayesian bootstrap
  做标签，`scripts/ramen_nn/labels.py`，不是贝叶斯优化）。
- CMA-ES 是另一条路：不建代理，而是自适应协方差矩阵，对 29 维这种中低维、
  各维量纲差好几个数量级（`act_level` 0–400 vs `hint` 0–4）的问题特别合适，
  因为它会自己学出各维的步长与相关性。

两者共用同一套评估口径（CRN 配对种子 + 留出集验收），所以可以直接比谁更省评估。

评估口径：
- 训练种子默认 61700–61729（30 个，全新块；GA 用的是 61444–61473，不重叠）
- 留出种子默认 61800–61899（100 个，只在最后评一次，不参与任何选择）
- 起点 = `lab/config/ga-best.toml`（GA 冠军），所以只会更好不会更差
- 条件 = 卡组 A + 轮次 6 搜出的教示路线（钉死），只搜权重

用法：
    python3 scripts/bopt.py --method bo    --evals 120 --out artifacts/bo
    python3 scripts/bopt.py --method cmaes --gens 15   --out artifacts/cmaes
    python3 scripts/bopt.py --validate lab/config/bo-best.toml
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ga import BOUNDS, evaluate, genome_to_toml, parse_overlay, paired_stats  # noqa: E402

KEYS = list(BOUNDS)
LO = np.array([BOUNDS[k][0] for k in KEYS])
HI = np.array([BOUNDS[k][1] for k in KEYS])
INT_MASK = np.array([BOUNDS[k][2] for k in KEYS])

ROUTE_S = ("0:501-101-301-401-201,1:501-101-201-301-401,2:601-501-301-401-701,"
           "3:601-301-501-401-701,4:801-601-1001-901-701,5:801-1001-601-901-701")


def to_unit(g: dict[str, float]) -> np.ndarray:
    """基因组 → [0,1]^29（GP / CMA-ES 都在单位立方体里跑，量纲差异交给它们自己学）"""
    return np.clip((np.array([g[k] for k in KEYS]) - LO) / (HI - LO), 0.0, 1.0)


def from_unit(x: np.ndarray) -> dict[str, float]:
    """[0,1]^29 → 基因组（整数基因四舍五入，全部夹回合法区间）"""
    raw = LO + np.clip(x, 0.0, 1.0) * (HI - LO)
    out = {}
    for i, k in enumerate(KEYS):
        v = float(raw[i])
        if INT_MASK[i]:
            v = float(round(v))
        out[k] = min(HI[i], max(LO[i], v))
    return out


class Evaluator:
    """带缓存的评估器：同一基因组（量化到 1e-6）只实算一次，重复点直接命中缓存。"""

    def __init__(self, repo: Path, binary: Path, seeds: list[int], args):
        self.repo, self.binary, self.seeds, self.args = repo, binary, seeds, args
        self.cache: dict[tuple, dict] = {}
        self.calls = 0
        self.t0 = time.time()

    def __call__(self, x: np.ndarray) -> float:
        g = from_unit(x)
        key = tuple(round(g[k], 6) for k in KEYS)
        hit = self.cache.get(key)
        if hit is not None:
            return hit["fitness"]
        r = evaluate(g, self.repo, self.binary, self.seeds, self.args.uma,
                     self.args.cards, self.args.pin_route)
        self.calls += 1
        r["genome"] = g
        self.cache[key] = r
        if r["n"] < len(self.seeds):
            print(f"  警告：只有 {r['n']}/{len(self.seeds)} 局跑出终局块", file=sys.stderr)
        print(f"  [{self.calls:>3}] fitness {r['fitness']:>9.1f} "
              f"(PT {r.get('pt', 0):.0f} 五维 {r.get('five', 0):.0f}) "
              f"· {time.time() - self.t0:.0f}s", flush=True)
        return r["fitness"]

    def best(self) -> tuple[float, dict]:
        item = max(self.cache.values(), key=lambda r: r["fitness"])
        return item["fitness"], item["genome"]


# --------------------------------------------------------------------------
# 高斯过程代理（RBF + 常数均值，超参按对数边际似然拟合）
# --------------------------------------------------------------------------
class GP:
    """29 维单长度尺度 RBF GP。噪声取 y 方差的固定比例（无重复点，学不出来）。"""

    def __init__(self, ell: float = 0.5, noise_frac: float = 0.05):
        self.ell, self.noise_frac = ell, noise_frac
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.mu = 0.0
        self.sf = 1.0
        self.sn = 1.0
        self.K_inv = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X, self.y = X.copy(), y.copy()
        self.mu = float(y.mean())
        self.sf = float(y.std() or 1.0)
        self.sn = self.noise_frac * self.sf
        yn = (y - self.mu) / self.sf

        def nll(params: np.ndarray) -> float:
            ell = float(np.exp(params[0]))
            K = self._k(X, X, ell) + (self.sn / self.sf) ** 2 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 1e9
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
            return float(0.5 * yn @ alpha + np.log(np.diag(L)).sum())

        from scipy.optimize import minimize
        res = minimize(nll, np.array([np.log(self.ell)]), method="L-BFGS-B",
                       bounds=[(np.log(0.05), np.log(8.0))])
        self.ell = float(np.exp(res.x[0]))
        K = self._k(X, X, self.ell) + (self.sn / self.sf) ** 2 * np.eye(len(X))
        L = np.linalg.cholesky(K)
        self.K_inv = np.linalg.inv(K)
        self.alpha = np.linalg.solve(L.T, np.linalg.solve(L, yn))
        self._L = L

    def _k(self, A: np.ndarray, B: np.ndarray, ell: float) -> np.ndarray:
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / (ell * ell))

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.X is not None
        Ks = self._k(Xs, self.X, self.ell)
        mean = self.mu + self.sf * (Ks @ self.alpha)
        v = np.linalg.solve(self._L, Ks.T)
        var = self.sf ** 2 * (1.0 - (v ** 2).sum(0)) + self.sn ** 2
        return mean, np.sqrt(np.maximum(var, 1e-12))


def expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    from scipy.stats import norm
    z = (mean - best) / sd
    return (mean - best) * norm.cdf(z) + sd * norm.pdf(z)


def run_bo(ev: Evaluator, args, x0: np.ndarray, log_path: Path) -> None:
    """贝叶斯优化：Sobol 式随机初始设计 → GP 拟合 → EI 采集 → 迭代。"""
    rng = np.random.default_rng(args.rng_seed)
    d = len(x0)
    X = [x0]
    # 初始设计一半是拉丁超立方（保证各维都被扫到），一半是起点邻域的高斯扰动
    # （29 维下 GP 样本极稀，起点邻域的信息量比全局撒点高）
    n_init = max(8, args.evals // 5)
    n_lhs, n_loc = n_init // 2, n_init - n_init // 2
    base = np.tile(np.arange(n_lhs)[:, None], (1, d))
    for j in range(d):
        base[:, j] = rng.permutation(base[:, j])
    lhs = np.clip((base + rng.random((n_lhs, d))) / n_lhs, 0, 1)
    loc = np.clip(x0 + rng.normal(0, 0.08, (n_loc, d)), 0, 1)
    X.extend(lhs)
    X.extend(loc)
    y = [ev(x) for x in X]
    history = []
    while ev.calls < args.evals:
        Xm, ym = np.array(X), np.array(y)
        gp = GP(noise_frac=args.noise_frac)
        gp.fit(Xm, ym)
        # 采集函数最大化：64 个随机起点 + 历史最优点邻域，各做 L-BFGS-B
        from scipy.optimize import minimize
        cand = rng.random((512, len(x0)))
        best_x = Xm[int(np.argmax(ym))]
        cand[:64] = np.clip(best_x + rng.normal(0, 0.05, (64, len(x0))), 0, 1)
        mu, sd = gp.predict(cand)
        ei = expected_improvement(mu, sd, float(ym.max()))
        starts = cand[np.argsort(-ei)[:8]]
        best = None
        for s in starts:
            res = minimize(lambda z: -expected_improvement(*gp.predict(z[None, :]),
                                                           float(ym.max()))[0],
                           s, bounds=[(0.0, 1.0)] * len(x0), method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        x_new = np.clip(best.x, 0, 1)
        y_new = ev(x_new)
        X.append(x_new)
        y.append(y_new)
        history.append({"evals": ev.calls, "best": float(np.max(y)),
                        "ell": round(gp.ell, 4), "x": x_new.tolist()})
        log_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf8")
        print(f"  GP ell={gp.ell:.3f} 当前最优 {np.max(y):.1f}", flush=True)


def run_cmaes(ev: Evaluator, args, x0: np.ndarray, log_path: Path) -> None:
    """CMA-ES（`cma` 包）：自适应协方差，边界 [0,1]^29，sigma0 = 0.15。"""
    import cma
    es = cma.CMAEvolutionStrategy(list(x0), args.sigma0, {
        "bounds": [0.0, 1.0], "seed": args.rng_seed, "maxiter": args.gens,
        "popsize": args.popsize, "verbose": -9, "CMA_stds": args.sigma0,
    })
    history = []
    while not es.stop() and es.countiter < args.gens:
        xs = es.ask()
        ys = [-ev(np.clip(np.array(x), 0, 1)) for x in xs]   # CMA 求最小 → 取负
        es.tell(xs, ys)
        es.disp()
        history.append({"gen": int(es.countiter), "best": float(-min(ys)),
                        "overall": float(-es.result.fbest), "sigma": float(es.sigma)})
        log_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf8")
    print(f"CMA-ES 结束：{es.result.xbest} → {-es.result.fbest:.1f}", flush=True)


def run_sa(ev: Evaluator, args, x0: np.ndarray, log_path: Path) -> None:
    """模拟退火：允许**按温度概率接受变差**的移动翻山头，温度几何降温到接近贪心。

    邻域 = 随机挑 `--sa-dims` 个维度做高斯扰动（步长 `--sa-step`，单位立方量纲）。
    温度以「评分点」为单位：Δ<0 时以 exp(Δ/T) 接受，T 从 `--sa-t0` 降到 `--sa-t1`。
    """
    rng = np.random.default_rng(args.rng_seed)
    cur, cur_fit = np.array(x0, dtype=float), ev(x0)
    best_x, best_fit = cur.copy(), cur_fit
    history = []
    accepted = 0
    for i in range(args.evals):
        frac = i / max(1, args.evals - 1)
        temp = args.sa_t0 * (args.sa_t1 / args.sa_t0) ** frac
        step = args.sa_step * (1.0 - 0.7 * frac)      # 后期步长也收，避免在原点乱跳
        dims = rng.choice(len(cur), size=args.sa_dims, replace=False)
        cand = cur.copy()
        cand[dims] = np.clip(cand[dims] + rng.normal(0, step, len(dims)), 0, 1)
        fit = ev(cand)
        delta = fit - cur_fit
        if delta > 0 or rng.random() < float(np.exp(delta / max(temp, 1e-9))):
            cur, cur_fit = cand, fit
            accepted += 1
        if fit > best_fit:
            best_x, best_fit = cand.copy(), fit
        history.append({"step": i, "temp": round(temp, 1), "cur": round(cur_fit, 1),
                        "best": round(best_fit, 1), "accepted": accepted,
                        "x": best_x.tolist()})
        log_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf8")
        print(f"  [{i:>3}] T={temp:>7.1f} 当前 {cur_fit:>9.1f} 最优 {best_fit:>9.1f} "
              f"接受 {accepted}", flush=True)
    print(f"SA 结束：最优 {best_fit:.1f}（接受 {accepted}/{args.evals} 步）", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["bo", "cmaes", "sa"], default="bo")
    ap.add_argument("--start", default="lab/config/ga-best.toml", help="起点覆盖层")
    ap.add_argument("--evals", type=int, default=120, help="BO 的评估预算")
    ap.add_argument("--gens", type=int, default=15, help="CMA-ES 代数")
    ap.add_argument("--popsize", type=int, default=0, help="CMA-ES 种群（0=自动）")
    ap.add_argument("--sigma0", type=float, default=0.15)
    ap.add_argument("--sa-t0", type=float, default=1200.0, help="退火初温（评分点）")
    ap.add_argument("--sa-t1", type=float, default=60.0, help="退火终温（评分点）")
    ap.add_argument("--sa-step", type=float, default=0.12, help="邻域高斯步长（单位立方）")
    ap.add_argument("--sa-dims", type=int, default=4, help="每步扰动几个维度")
    ap.add_argument("--noise-frac", type=float, default=0.05, help="GP 噪声 = 该比例 × y 标准差")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--start-seed", type=int, default=61700)
    ap.add_argument("--holdout-seed", type=int, default=61800)
    ap.add_argument("--holdout-count", type=int, default=100)
    ap.add_argument("--uma", default="102102")
    ap.add_argument("--cards", default="303124,303114,303004,303224,303204,303234")
    ap.add_argument("--pin-route", default=ROUTE_S)
    ap.add_argument("--binary", default="target/release/matsuri")
    ap.add_argument("--out", default="artifacts/bopt")
    ap.add_argument("--rng-seed", type=int, default=20261102)
    ap.add_argument("--validate", default="", help="只评估这个覆盖层（留出集），不搜索")
    args = ap.parse_args()
    if args.popsize <= 0:
        args.popsize = 4 + int(3 * np.log(len(KEYS)))

    repo = Path.cwd()
    binary = (repo / args.binary).resolve()
    if not binary.exists():
        print(f"找不到可执行文件: {binary}", file=sys.stderr)
        return 2
    out = repo / args.out
    out.mkdir(parents=True, exist_ok=True)

    holdout = [args.holdout_seed + i for i in range(args.holdout_count)]
    if args.validate:
        g = parse_overlay(Path(args.validate).read_text(encoding="utf8"))
        r = evaluate(g, repo, binary, holdout, args.uma, args.cards, args.pin_route)
        print(json.dumps({k: v for k, v in r.items() if k != "scores"},
                         ensure_ascii=False, indent=2))
        return 0

    x0 = to_unit(parse_overlay((repo / args.start).read_text(encoding="utf8")))
    train = [args.start_seed + i for i in range(args.seeds)]
    print(f"方法 {args.method} · 起点 {args.start} · 训练种子 {train[0]}–{train[-1]} · "
          f"留出 {holdout[0]}–{holdout[-1]}", flush=True)
    ev = Evaluator(repo, binary, train, args)

    log = out / "history.json"
    if args.method == "bo":
        run_bo(ev, args, x0, log)
    elif args.method == "cmaes":
        run_cmaes(ev, args, x0, log)
    else:
        run_sa(ev, args, x0, log)

    fit, champ = ev.best()
    print(f"\n搜索最优 训练集 {fit:.1f}（评估 {ev.calls} 次，"
          f"{time.time() - ev.t0:.0f}s）", flush=True)
    (out / "best.toml").write_text(genome_to_toml(champ), encoding="utf8")
    (out / "best.json").write_text(json.dumps(
        {"method": args.method, "evals": ev.calls, "train_fitness": round(fit, 1),
         "genome": champ}, ensure_ascii=False, indent=2), encoding="utf8")

    # 留出集验收：冠军 vs 起点（ga-best），同种子配对
    print("留出集评估冠军 …", flush=True)
    ch = evaluate(champ, repo, binary, holdout, args.uma, args.cards, args.pin_route)
    print("留出集评估起点 …", flush=True)
    base = evaluate(from_unit(x0), repo, binary, holdout, args.uma, args.cards,
                    args.pin_route)
    mean, t, wins = paired_stats(ch["scores"], base["scores"])
    report = {
        "method": args.method, "evals": ev.calls,
        "champion_holdout": {k: v for k, v in ch.items() if k != "scores"},
        "start_holdout": {k: v for k, v in base.items() if k != "scores"},
        "paired_delta": round(mean, 1), "paired_t": round(t, 2),
        "champion_wins": wins, "n": len(base["scores"]),
    }
    (out / "holdout.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
