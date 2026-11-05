//! 手写训练员（S7）：守门链 + 打分表 + 事件选项 + 教诲分配择优
//!
//! 目标口径见 [`crate::value`]：官方评分（五维查表 + PT×折算率），PT 再乘配置的
//! `pt_favor_rate`（`1.0` = 官方评分）。所有决策都是**确定性**的（不消耗决策随机流），
//! 保证回放 / CRN / 基线可复现；策略旋钮全部来自 `MatsuriConfig.trainer`（可扫参）。
//!
//! 决策面（引擎契约只给 `Trainer` 三个 hook，教诲分配由剧本在 `Begin` / `End` 主动询问）：
//! - `select_action`：普通动作候选 → 守门链 + 打分表；教诲分配候选 → 价值矩阵择优；
//! - `select_event_choice`：事件选项 → 通用期望值打分（追加稽古「上 / 下」等按语义自动比）；
//! - `select_choice`：旧接口兜底（无事件数据时按通用期望值打分）。

use anyhow::{Result, bail};
use rand::SeedableRng;
use rand::rngs::StdRng;
use umaai_core::Ill;
use umaai_core::gamedata::{EventChoice, EventData};
use umasim::game::{Game, Trainer};
use umasim::search::{FlatSearch, FlatSearchGame, NoTerminal, SearchConfig};

use crate::config::MatsuriConfig;
use crate::diag;
use crate::game::{Action, MatsuriGame};
use crate::value::Model;

/// 手写训练员（无状态：策略只来自随局配置；便于 `FlatSearch` rollout 复用）
pub struct MatsuriTrainer {
    /// 剧本配置（融合预设 + 训练员旋钮）
    config: MatsuriConfig,
    /// 蒙特卡洛搜索配置（引擎 `FlatSearch`：扁平 MC + UCB 动态分配 + CRN）
    ///
    /// `None` = 纯手写打分。**rollout 基策必须是 `None`**：否则搜索里的每次 rollout
    /// 又会在教诲决策点发起搜索，成本按深度指数爆炸。
    search: Option<SearchConfig>,
    /// 教诲分配候选的剪枝宽度（24 个互异落位全搜太贵，只搜手写打分最高的前 K 个）
    search_topk: usize,
    /// 合宿训练回合的前瞻搜索配置（攻略优化点 3：夏合宿 4 轮组合前瞻）
    ///
    /// `None` = 合宿回合仍走守门②的逐回合贪心 `action_value`。开启后，合宿训练回合
    /// 改用引擎 `FlatSearch` 对候选训练做蒙特卡洛前瞻（rollout 到终局），让「这一轮练什么」
    /// 把后续合宿轮 / 全局收益一并计入，而非只看当回合。rollout 基策同样 `None`（防递归）。
    camp_search: Option<SearchConfig>,
    /// 合宿前瞻的候选剪枝宽度（按贪心 `action_value` 排序后只搜前 K 个训练位）
    camp_search_topk: usize,
    /// 常规（非合宿 / 非教诲）回合的前瞻搜索配置（攻略优化点 1）
    ///
    /// `None` = 常规回合走守门链后的逐回合贪心打分表。开启后，对通过守门链的自由决策回合
    /// 用 `FlatSearch` MC rollout 到终局择优——让「这回合练不练当前（可能很差的）稽古」把
    /// **下一回合稽古 lineup 重新洗牌**后的期望收益一并计入，而非只看当回合贪心分。
    /// rollout 基策同样 `None`（防递归）。常规回合多（~65/局），成本高于合宿前瞻。
    normal_search: Option<SearchConfig>,
    /// 常规前瞻的候选剪枝宽度（按贪心 `action_value` 排序后只搜前 K 个动作）
    normal_search_topk: usize,
    /// 事件选项决策点的前瞻搜索配置（上游 21b278c 的第四路；`None` = 走静态期望值打分）
    ///
    /// 事件选项不是 `Action`：用 dummy token 走 `FlatSearch::search_with` 复用同一套
    /// UCB / CRN，闭包内 fork → 应用当前选项 → 从 `event_queue` 的下一个事件续跑 →
    /// 再把后续 stage 跑到终局。rollout 基策同样 `None`（防递归）。
    event_search: Option<SearchConfig>,
}

impl MatsuriTrainer {
    /// 用剧本配置构造训练员（不开蒙特卡洛）
    pub fn new(config: MatsuriConfig) -> Self {
        Self {
            config,
            search: None,
            search_topk: 4,
            camp_search: None,
            camp_search_topk: 4,
            normal_search: None,
            normal_search_topk: 4,
            event_search: None,
        }
    }

    /// 打开教诲决策点的蒙特卡洛搜索（`search_n = 0` 视为关闭）
    #[must_use]
    pub fn with_search(mut self, search: SearchConfig, topk: usize) -> Self {
        self.search = (search.search_n > 0).then_some(search);
        self.search_topk = topk.max(1);
        self
    }

    /// 打开合宿训练回合的前瞻蒙特卡洛搜索（攻略优化点 3；`search_n = 0` 视为关闭）
    #[must_use]
    pub fn with_camp_search(mut self, search: SearchConfig, topk: usize) -> Self {
        self.camp_search = (search.search_n > 0).then_some(search);
        self.camp_search_topk = topk.max(1);
        self
    }

    /// 打开常规回合的前瞻蒙特卡洛搜索（攻略优化点 1：稽古洗牌前瞻；`search_n = 0` 视为关闭）
    #[must_use]
    pub fn with_normal_search(mut self, search: SearchConfig, topk: usize) -> Self {
        self.normal_search = (search.search_n > 0).then_some(search);
        self.normal_search_topk = topk.max(1);
        self
    }

    /// 打开事件选项决策点的前瞻蒙特卡洛搜索（`search_n = 0` 视为关闭）
    ///
    /// 事件选项一般只有 2–3 个，全部进搜索（不剪枝）；超过 [`EVENT_MAX_CHOICES`] 的
    /// 异常事件退回静态打分，避免候选爆炸。
    #[must_use]
    pub fn with_event_search(mut self, search: SearchConfig) -> Self {
        self.event_search = (search.search_n > 0).then_some(search);
        self
    }

    /// 取剧本配置（宿主 / 训练员参数可读）
    #[must_use]
    pub fn config(&self) -> &MatsuriConfig {
        &self.config
    }

    /// 是否需要优先治病（变胖只影响训练速度、每回合 20% 自解，不占动作；不擅长 / 具名病才治）
    fn needs_clinic(game: &MatsuriGame) -> bool {
        let flags = &game.uma().flags;
        flags.bad_trainer || flags.ill.bits() & !Ill::FAT.bits() != 0
    }

    /// 教诲分配的蒙特卡洛择优（引擎 `FlatSearch`：扁平 MC + UCB + CRN 配对种子）
    ///
    /// 手写打分把 24 个互异落位排序后只留前 `search_topk` 个进搜索，每个候选各跑
    /// `search_n` 次 rollout 到终局；中选口径是搜索的排名加权平均分（`radical_factor`
    /// 按剩余回合确定性衰减）。返回 `None` = 剪枝后不足 2 个候选，无需搜索。
    fn search_advice(
        &self,
        game: &MatsuriGame,
        actions: &[Action],
        scored: &[(f64, usize)],
        search: &SearchConfig,
    ) -> Result<Option<(usize, f64)>> {
        let cand: Vec<Action> = scored
            .iter()
            .take(self.search_topk)
            .map(|&(_, index)| actions[index].clone())
            .collect();
        if cand.len() < 2 {
            return Ok(None);
        }
        let searcher = FlatSearch::<MatsuriGame>::new(search.clone())
            // rollout 基策 = 同一份配置的手写训练员，且**不再开搜索**（防递归）
            .with_rollout_trainer(MatsuriTrainer::new(self.config.clone()))
            .with_parallel(true);
        // 搜索流独立派生：不吃决策随机流（决策确定性是 CRN / 回放的契约），
        // 按 (主种子, 回合) 定值，同一局面重复搜索逐位一致
        let mut rng = StdRng::seed_from_u64(
            game.rule_master().unwrap_or(0) ^ (game.turn().max(0) as u64) << 40,
        );
        // rollout 里每回合都会打诊断行，几万局下来会淹掉终局块 → 静默
        // （guard 出块即恢复，本决策点自己的诊断行照打）
        let (best, value) = {
            let _quiet = umaai_core::diag::DiagGuard::suppress();
            let out = searcher.search::<NoTerminal, _>(game, &cand, &mut rng, |_| NoTerminal)?;
            (
                out.best_action_idx,
                out.action_results[out.best_action_idx].weighted_mean(out.radical_factor),
            )
        };
        diag!(
            "【搜索】T{} 教诲 MC 候选 {} 个 × {} 次 → 中选第 {} 位 值={:.1}",
            game.turn(),
            cand.len(),
            search.search_n,
            best,
            value
        );
        Ok(Some((scored[best].1, value)))
    }

    /// 蒙特卡洛前瞻择优的公共内核（教诲 / 合宿 / 常规三处共用）：引擎 `FlatSearch` 对
    /// 手写贪心排序后的前 `topk` 个候选各跑 `search.search_n` 次 rollout 到终局，中选口径
    /// 换成搜索到的终局期望（`radical_factor` 按剩余回合衰减）。rollout 基策 = 同一份配置的
    /// 手写训练员且**不开任何搜索**（防递归）。返回 `None` = 剪枝后不足 2 个候选，退回贪心。
    fn mc_pick(
        &self,
        game: &MatsuriGame,
        actions: &[Action],
        scored: &[(f64, usize)],
        search: &SearchConfig,
        topk: usize,
        label: &str,
    ) -> Result<Option<(usize, f64)>> {
        let cand: Vec<Action> = scored
            .iter()
            .take(topk)
            .map(|&(_, index)| actions[index].clone())
            .collect();
        if cand.len() < 2 {
            return Ok(None);
        }
        let searcher = FlatSearch::<MatsuriGame>::new(search.clone())
            .with_rollout_trainer(MatsuriTrainer::new(self.config.clone()))
            .with_parallel(true);
        let mut rng = StdRng::seed_from_u64(
            game.rule_master().unwrap_or(0) ^ (game.turn().max(0) as u64) << 40,
        );
        let (best, value) = {
            let _quiet = umaai_core::diag::DiagGuard::suppress();
            let out = searcher.search::<NoTerminal, _>(game, &cand, &mut rng, |_| NoTerminal)?;
            (
                out.best_action_idx,
                out.action_results[out.best_action_idx].weighted_mean(out.radical_factor),
            )
        };
        diag!(
            "【搜索】T{} {} MC 候选 {} 个 × {} 次 → 中选第 {} 位 值={:.1}",
            game.turn(),
            label,
            cand.len(),
            search.search_n,
            best,
            value
        );
        Ok(Some((scored[best].1, value)))
    }

    /// 合宿训练回合的蒙特卡洛前瞻（攻略优化点 3）
    fn search_camp(
        &self,
        game: &MatsuriGame,
        actions: &[Action],
        scored: &[(f64, usize)],
        search: &SearchConfig,
    ) -> Result<Option<(usize, f64)>> {
        self.mc_pick(game, actions, scored, search, self.camp_search_topk, "合宿前瞻")
    }

    /// 常规回合的蒙特卡洛前瞻（攻略优化点 1：稽古洗牌前瞻）
    fn search_normal(
        &self,
        game: &MatsuriGame,
        actions: &[Action],
        scored: &[(f64, usize)],
        search: &SearchConfig,
    ) -> Result<Option<(usize, f64)>> {
        self.mc_pick(game, actions, scored, search, self.normal_search_topk, "常规前瞻")
    }

    /// 事件选项的蒙特卡洛前瞻（上游 21b278c 第四路）
    ///
    /// 与 `mc_pick` 的差别：候选不是 `Action` 而是「选项下标」，所以不能走
    /// `apply_root_action`——改用 `search_with` 自己接管 rollout：fork → 应用当前选项 →
    /// **从 `event_queue` 的下一个事件续跑**（否则会漏掉同 stage 剩余事件）→ 跑完后续
    /// stage 到终局 → `search_score`。事件流只在闭包里取用，不碰调用方的决策流。
    fn search_event(
        &self,
        game: &MatsuriGame,
        event: &EventData,
        choices: &[Vec<EventChoice>],
        rng: &mut StdRng,
        search: &SearchConfig,
    ) -> Result<Option<usize>> {
        if game.event_queue.get(game.event_cursor).is_none() {
            // 不在事件队列上下文里（例如宿主直接调 select_event_choice）：无法续跑，退回静态打分
            return Ok(None);
        }
        let tokens: Vec<Action> = (0..choices.len()).map(event_token).collect();
        let pt = self.config.trainer.pt_favor_rate;
        let searcher = FlatSearch::<MatsuriGame>::new(search.clone())
            .with_rollout_trainer(MatsuriTrainer::new(self.config.clone()))
            .with_parallel(true);
        let out = {
            let _quiet = umaai_core::diag::DiagGuard::suppress();
            searcher.search_with(game, &tokens, rng, |g, token, seed| {
                let choice = choice_of_token(token)?;
                let mut sim = g.fork_for_rollout(seed);
                if sim.event_queue.get(sim.event_cursor).is_none() {
                    bail!("事件搜索缺少事件队列游标");
                }
                let mut streams = sim.base.take_rule_streams()?;
                // 当前事件的这一选项
                sim.apply_event(event, choice, &mut streams.event)?;
                sim.event_cursor += 1;
                // 同一 stage 的剩余事件从下一个继续（用 rollout 基策决策 / 结算）
                let fallback = MatsuriTrainer::new(sim.config.clone());
                let mut sim_rng = StdRng::seed_from_u64(seed);
                while sim.event_cursor < sim.event_queue.len() {
                    let next = sim.event_queue[sim.event_cursor].clone();
                    sim.run_event(&next, &fallback, &mut sim_rng, &mut streams.event)?;
                    sim.event_cursor += 1;
                }
                sim.event_queue.clear();
                sim.event_cursor = 0;
                sim.base.put_rule_streams(streams);
                while sim.next() {
                    sim.run_stage(&fallback, &mut sim_rng)?;
                }
                sim.on_simulation_end(&fallback, &mut sim_rng)?;
                sim.search_score(pt)
            })?
        };
        let picked = out.best_action_idx;
        diag!(
            "【搜索】T{} 事件#{} {} 选项{} 个 × {} 次 → 中选第 {} 项 值={:.1}",
            game.turn(),
            event.id,
            event.name,
            choices.len(),
            search.search_n,
            picked,
            out.action_results[picked].weighted_mean(out.radical_factor)
        );
        Ok(Some(choice_of_token(&tokens[picked])?))
    }
}

/// 事件选项搜索的 dummy token 上限（选项数超过它就退回静态打分，避免候选爆炸）
const EVENT_MAX_CHOICES: usize = 10;

/// dummy action token：0–4 用 `Train`，5–9 用 `ActTrain`（仅作候选标识，不会被执行）
fn event_token(index: usize) -> Action {
    if index < 5 {
        Action::Train(index as u8)
    } else {
        Action::ActTrain((index - 5) as u8)
    }
}

/// token → 事件选项下标
fn choice_of_token(token: &Action) -> Result<usize> {
    match token {
        Action::Train(index) => Ok(usize::from(*index)),
        Action::ActTrain(index) => Ok(usize::from(*index) + 5),
        other => bail!("事件搜索 token 非法：{other}"),
    }
}

impl Trainer<MatsuriGame> for MatsuriTrainer {
    fn select_action(
        &self,
        game: &MatsuriGame,
        actions: &[Action],
        _rng: &mut StdRng,
    ) -> Result<usize> {
        if actions.is_empty() {
            bail!("候选动作为空，无法选择");
        }
        if actions.len() == 1 {
            diag!(
                "【决策】T{} 唯一候选 {}（无需比较）",
                game.turn(),
                actions[0]
            );
            return Ok(0);
        }
        // 教诲分配决策点：候选全是分配方案 → 价值矩阵择优（确定性）
        if actions
            .iter()
            .all(|action| matches!(action, Action::AssignAdvice(_)))
        {
            let model = Model::new(game, &self.config.trainer)?;
            // 手写打分先给全部候选排序：既是无搜索时的答案，也是有搜索时的剪枝依据
            let mut scored: Vec<(f64, usize)> = Vec::with_capacity(actions.len());
            for (index, action) in actions.iter().enumerate() {
                let Action::AssignAdvice(assign) = action else {
                    continue;
                };
                scored.push((model.assignment_value(assign), index));
            }
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            let (mut best_value, mut best) = scored
                .first()
                .map_or((f64::NEG_INFINITY, 0), |&(value, index)| (value, index));
            // 蒙特卡洛：中选口径换成「搜索到的终局期望」，手写分只负责剪枝
            if let Some(search) = self.search.as_ref()
                && let Some((index, value)) = self.search_advice(game, actions, &scored, search)?
            {
                best = index;
                best_value = value;
            }
            diag!(
                "【决策】T{} 教诲分配 值={:.1} 方案 {}",
                game.turn(),
                best_value,
                actions
                    .get(best)
                    .map(ToString::to_string)
                    .unwrap_or_default()
            );
            return Ok(best);
        }

        // 守门 ①：生病 / 不擅长且可选治病时优先治病
        if Self::needs_clinic(game)
            && let Some(index) = actions.iter().position(|action| *action == Action::Clinic)
        {
            diag!("【决策】T{} 守门·治病优先（不擅长 / 生病）", game.turn());
            return Ok(index);
        }

        // 守门 ②：合宿禁令（合宿训练回合内休息 / 比赛 = 浪费训练等级 buff，强制不选）
        // 体力危险线 32 以下仍保命休息
        if game.plan.camp_train(game.forced_rest) && game.uma().vital > 32 {
            let training_actions: Vec<usize> = actions
                .iter()
                .enumerate()
                .filter(|(_, action)| matches!(action, Action::Train(_) | Action::ActTrain(_)))
                .map(|(index, _)| index)
                .collect();
            if !training_actions.is_empty() {
                // 合宿期间只从训练/稽古里选（进打分表但排除休息/比赛/出行）
                let model = Model::new(game, &self.config.trainer)?;
                // 手写贪心给全部合宿训练位排序：既是无搜索时的答案，也是有搜索时的剪枝依据
                let mut scored: Vec<(f64, usize)> = Vec::with_capacity(training_actions.len());
                for &index in &training_actions {
                    let value = model.action_value(&actions[index]);
                    if value.is_nan() {
                        continue;
                    }
                    scored.push((value, index));
                }
                scored.sort_by(|a, b| b.0.total_cmp(&a.0));
                let (mut best_value, mut best) = scored
                    .first()
                    .map_or((f64::NEG_INFINITY, training_actions[0]), |&(v, i)| (v, i));
                // 攻略优化点 3：合宿前瞻蒙特卡洛（中选口径换成搜索到的终局期望，贪心分只负责剪枝）
                if let Some(search) = self.camp_search.as_ref()
                    && let Some((index, value)) = self.search_camp(game, actions, &scored, search)?
                {
                    best = index;
                    best_value = value;
                }
                diag!(
                    "【决策】T{} 守门·合宿全练 {} 值={:.1}",
                    game.turn(),
                    actions
                        .get(best)
                        .map(ToString::to_string)
                        .unwrap_or_default(),
                    best_value
                );
                return Ok(best);
            }
        }

        // 守门 ③：友人外出截止线（截止回合前必须把剩余段用完，否则强制友人外出）
        //
        // 截止回合 = `friend_outing_deadline`（0 = Senior 合宿起点 ≈ 第三年 10 月上）。
        // 估算截止前可用回合时排除约 18% 的比赛 / 强制回合；剩余段数 > 可用回合即提速。
        if let Some(index) = actions
            .iter()
            .position(|action| *action == Action::FriendOuting)
        {
            let remaining_uses = game.friend.out_used.iter().filter(|used| !**used).count();
            let deadline = if self.config.trainer.friend_outing_deadline > 0 {
                self.config.trainer.friend_outing_deadline
            } else {
                game.data.camp.windows.get(2).map_or(60, |window| window.start)
            };
            let turns_to_deadline = (deadline - game.turn()).max(0) as usize;
            let usable_turns = (turns_to_deadline as f64 * 0.82) as usize;
            if remaining_uses > 0 && remaining_uses > usable_turns {
                diag!(
                    "【决策】T{} 守门·友人外出截止（剩{}段/截止T{}前可用{}回合）",
                    game.turn(),
                    remaining_uses,
                    deadline,
                    usable_turns
                );
                return Ok(index);
            }
        }

        // 守门 ④：体力危险线（vital < 32 且非合宿 → 强制休息）
        if game.uma().vital < 32
            && !game.plan.camp_train(game.forced_rest)
            && let Some(index) = actions.iter().position(|action| *action == Action::Rest)
        {
            diag!(
                "【决策】T{} 守门·体力危险（{}<32）强制休息",
                game.turn(),
                game.uma().vital
            );
            return Ok(index);
        }

        // 守门 ⑤：心情崩坏（干劲≤2 且体力>50 → 强制出行修心情）
        // GA 战报「摆烂病」根因 + 作者确认：祭典杯友人不给心情盾，掉心情影响大
        if game.uma().motivation <= 2
            && game.uma().vital > 50
            && !game.plan.camp_train(game.forced_rest)
        {
            // 优先友人出行（同时修心情 + 演目 Lv）
            if let Some(index) = actions
                .iter()
                .position(|action| *action == Action::FriendOuting)
            {
                diag!("【决策】T{} 守门·心情崩坏→友人出行", game.turn());
                return Ok(index);
            }
            if let Some(index) = actions
                .iter()
                .position(|action| *action == Action::NormalOuting)
            {
                diag!("【决策】T{} 守门·心情崩坏→普通出行", game.turn());
                return Ok(index);
            }
        }

        // 其余全部进入打分表（训练 / 稽古 / 休息 / 出行 / 比赛同权比较）
        let model = Model::new(game, &self.config.trainer)?;
        // 手写贪心给全部候选排序：既是无搜索时的答案，也是有搜索时的剪枝依据
        let mut scored: Vec<(f64, usize)> = Vec::with_capacity(actions.len());
        for (index, action) in actions.iter().enumerate() {
            let value = model.action_value(action);
            if value.is_nan() {
                continue;
            }
            scored.push((value, index));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut best_value, mut best) = scored
            .first()
            .map_or((f64::NEG_INFINITY, 0), |&(value, index)| (value, index));
        // 攻略优化点 1：常规回合稽古洗牌前瞻（中选口径换成搜索到的终局期望，贪心分只负责剪枝）
        if let Some(search) = self.normal_search.as_ref()
            && let Some((index, value)) = self.search_normal(game, actions, &scored, search)?
        {
            best = index;
            best_value = value;
        }
        diag!(
            "【决策】T{} 动作 {} 值={:.1}",
            game.turn(),
            actions
                .get(best)
                .map(ToString::to_string)
                .unwrap_or_default(),
            best_value
        );
        Ok(best)
    }

    fn select_choice(
        &self,
        game: &MatsuriGame,
        choices: &[Vec<EventChoice>],
        _rng: &mut StdRng,
    ) -> Result<usize> {
        if choices.is_empty() {
            return Ok(0);
        }
        if choices.len() == 1 {
            return Ok(0);
        }
        let model = Model::new(game, &self.config.trainer)?;
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (index, group) in choices.iter().enumerate() {
            let value = model.event_choice_value(0, index, group);
            if value > best_value {
                best_value = value;
                best = index;
            }
        }
        diag!(
            "【决策】T{} 选项 {best} 期望={:.1}（旧接口）",
            game.turn(),
            best_value
        );
        Ok(best)
    }

    fn select_event_choice(
        &self,
        game: &MatsuriGame,
        event: &EventData,
        choices: &[Vec<EventChoice>],
        rng: &mut StdRng,
    ) -> Result<usize> {
        if choices.is_empty() {
            return Ok(0);
        }
        if choices.len() == 1 {
            return Ok(0);
        }
        let model = Model::new(game, &self.config.trainer)?;
        let mut best = 0;
        let mut best_value = f64::NEG_INFINITY;
        for (index, group) in choices.iter().enumerate() {
            let value = model.event_choice_value(event.id, index, group);
            if value > best_value {
                best_value = value;
                best = index;
            }
        }
        // 事件前瞻：中选口径换成搜索到的终局期望，静态分只负责「搜不动时」的兜底
        if choices.len() <= EVENT_MAX_CHOICES
            && let Some(search) = self.event_search.as_ref()
            && let Some(index) = self.search_event(game, event, choices, rng, search)?
        {
            best = index;
        }
        diag!(
            "【决策】T{} 事件#{} {} 选项{} 期望={:.1}",
            game.turn(),
            event.id,
            event.name,
            best,
            best_value
        );
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn trainer_prefers_heal_when_bad_trainer() {
        let mut game = crate::testing::sample_game();
        game.uma_mut().flags.bad_trainer = true;
        game.uma_mut().flags.ill.insert(Ill::BAD_TRAINER);
        let trainer = MatsuriTrainer::new(MatsuriConfig::default());
        let mut rng = StdRng::seed_from_u64(1);
        let actions = vec![Action::Train(0), Action::Clinic, Action::Rest];
        let index = trainer
            .select_action(&game, &actions, &mut rng)
            .expect("选择动作");
        assert_eq!(actions[index], Action::Clinic, "不擅长训练时优先治病");
    }

    #[test]
    fn trainer_reads_trainer_knobs() {
        let text = std::fs::read_to_string(crate::testing::data_dir().join("matsuri_config.toml"))
            .expect("仓内配置默认层应存在");
        let config = MatsuriConfig::parse(&text, None).expect("默认层应可解析");
        let trainer = MatsuriTrainer::new(config);
        assert_eq!(trainer.config().trainer.pt_favor_rate, 2.0);
    }
}
