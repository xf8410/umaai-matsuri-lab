# GA 轮次 9（减半 ON，34 基因）：训练集 +1281 但留出集 −562，冠军不采纳

日期：2026-11-03　分支：`lab/gates-variantC`　起点：`lab/config/ga-r2.toml`
协议：`scripts/ga.py --pop 24 --gens 40 --seeds 30 --start-seed 62200 --holdout-seed 62300
--holdout-count 100 --start lab/config/ga-r2.toml --out artifacts/ga-r3`
（训练 62200–62229，留出 62300–62399 全程未参与选择；玉藻十字 102102 + 卡组 A，无根卡）

## 本轮改动（§8.8-1 落实）

1. **减半 ON 纳入基因**：`status_halve_threshold`(0–2400, int)、`friend_outing_urgency`(0–400)、
   `weights.friend_outing`(0–400) 三个新基因，31 → 34。此前 `genome_to_toml` 不输出这三键，
   覆盖层 `[trainer]` 整段替换 → **轮次 8 的 GA 评估实际跑在 halve=0 / friend_outing=0 上**，
   与 ga-r2.toml 的实机口径不一致（这是本轮修掉的隐性 bug）。
2. `genome_to_toml` 补全 `status_halve_factor=0.5 / friend_outing_deadline=0 / level_step_bonus=0`，
   覆盖层不再依赖 serde 缺省。
3. **roundtrip 验证**：`parse_overlay(ga-r2.toml)` → `genome_to_toml` 生成的覆盖层与 ga-r2 原文
   在 seeds 62000–62002 上逐局评分完全一致（76115 / 69968 / 78731）。

## 结果

- 共 11610 局（缓存命中 265），σ 0.150→0.030，两次停滞重启。
- **训练集**（62200–62229，30 局精评）：冠军 73882.4 vs 基线 ga-r2 72601.4，**Δ+1280.9**。
- **留出集**（62300–62399，100 局，只评一次）：

| | 冠军 | 基线 ga-r2 | Δ |
|---|---|---|---|
| 育成评分 | 72357.6 | **72919.5** | **−561.9（t=−1.56，胜 38/100）** |
| min | 59947 | 61851 | −1904 |
| PT | 10582.6 | 10703.4 | −120.7 |
| 五维合计 | 11207.8 | 11235.5 | −27.8 |
| 演目Lv [速耐力根智] | [9.96,9.88,9.67,7.31,9.80] | [9.97,9.80,9.77,7.30,9.88] | ≈ |

- 冠军基因相对 ga-r2 的主要动作：`status_halve_threshold` 1200→1292、
  `weights.motivation` 23.66→8.63、`keiko_tempo` 0.486→1.29、`pt_favor_rate` 1.73→1.36、
  `advice.keiko` 0.763→0.961、`status_weights[耐]` 0.988→1.143。

## 结论

**过拟合哨兵命中，冠军不采纳，`ga-r2.toml` 维持冠军**：训练集 +1281 全部蒸发为留出集
−562（t=−1.56 方向为负），min 还恶化 1904。轮次 8 的 ga-r2 已经是这套 34 维连续空间在
当前卡组/马上的实际最优盆地；再搜只是在训练种子上刻痕。

- §8.8-1 关闭：减半 ON 的 GA 重跑已完成（含 ga.py 的 halve/friend_outing 基因化修复，
  该修复本身是正遗产，留在脚本里）。
- 后续若再跑 GA，建议换**训练环境**（换卡组/换马做多环境鲁棒选择）而不是换种子重搜同一环境。

## 附件

- `best.toml`：轮次 9 冠军覆盖层（仅存档，不采纳）。
- `holdout.json` / `history.json`：留出集对照 / 逐代历史。
- `artifacts/ga-r3/run.log`（未入库）：完整运行日志。
