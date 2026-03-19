# Polymarket NBA 错配扫描器 — 产品文档 v4

## 核心理念

**只在高确信度的价值洼地出现时推送买入信号。宁可漏掉机会，也不发垃圾信号。**

每条推送都应该让用户感到："这个我要认真看看"，而不是"又来了一条"。

---

## 一、系统架构

```
Polymarket Gamma API ──→ Scanner ──→ MismatchEngine ──→ Telegram推送
                              ↕            ↕
                     SportsbookClient   GamePredictor
                     (Pinnacle赔率)     (预测模型)
                              ↕            ↕
                         ESPNClient     ResultResolver
                     (伤病/B2B/战绩)    (结果回填)
                              ↕            ↕
                      InjuryChecker    ExpertCommittee
                     (实时伤病核查)    (4-Agent AI审议)
                              ↕            ↕
                        AutoLearner    GameMemory
                     (自学习系统)      (赛后复盘记忆)
                              ↕
                        NotionSync
                     (历史记录同步)
```

### 项目结构
```
prediction-market/
├── main.py                  # 主入口 + Scanner + 推送 + 复盘
├── scoring_params.json      # 自学习参数（自动优化）
├── core/
│   ├── database.py          # SQLite数据库封装
│   ├── sportsbook_client.py # Pinnacle赔率获取
│   ├── espn_client.py       # ESPN战绩/伤病/B2B + 伤病清理
│   ├── mismatch_engine.py   # 核心评分引擎（预测优先版）
│   ├── clob_client.py       # CLOB真实买卖价获取
│   ├── injury_checker.py    # 核心球员伤病检查
│   ├── committee.py         # 4-Agent专家委员会
│   ├── price_watcher.py     # 价格追踪（内部用，不推送）
│   ├── auto_trader.py       # 自动交易（一键下单）
│   ├── telegram_bot.py      # Telegram交互式Bot
│   └── notion_sync.py       # Notion同步
├── ml/
│   ├── ml_predictor.py      # ML修正层
│   ├── game_predictor.py    # 主预测模型
│   ├── nba_features.py      # 特征构建（NBA API）
│   ├── game_memory.py       # 赛后复盘记忆系统
│   ├── auto_learner.py      # 自学习系统（autoresearch风格）
│   ├── player_ratings.py    # 球员评分系统
│   ├── team_ratings.py      # OpenSkill球队评分
│   ├── backtester.py        # 回测分析
│   └── result_resolver.py   # 比赛结果回填
└── requirements.txt
```

---

## 二、推送类型（4种）

### 保留的推送

| # | 类型 | 时间 | 门槛 | 说明 |
|---|------|------|------|------|
| 1 | **比赛预测信号** | 每小时扫描 | 评分≥45 + 信心≥65% + edge≥2% + Pinnacle一致 + 伤病检查 + 委员会通过 | 核心推送，经过最严格过滤 |
| 2 | **期货预测信号** | 每天08:00 | 概率≥20% + edge≥8% + 价格≤25% + 评分≥50 + 委员会通过 | 长线投资，委员会审议 |
| 3 | **每晚精选推荐** | 每天19:00 | 评分≥55 + 信心≥68% + edge≥2.5% + Pinnacle一致 + 伤病检查 + 委员会通过 | 当晚最高确定性，最多2场 |
| 4 | **每日复盘报表** | 每天09:00 | — | 昨日胜负、7天累计、盈亏追踪 |

### 已删除的推送（v4移除）

| 推送 | 删除原因 |
|------|----------|
| 低价买入机会 | 绕过所有过滤器（无评分、无Pinnacle验证、无委员会），edge失真 |
| 每日伤病报告 | 单独推没用，伤病信息已融入比赛推送的AI分析中 |

### 所有交易类推送的统一标准
- ✅ 必须过评分门槛
- ✅ 必须Pinnacle一致（比赛盘）
- ✅ 必须过专家委员会审议（4-Agent流程）
- ✅ 必须有双方球队伤病检查
- ✅ 必须有AI双方球队对战分析
- ✅ edge用衰减函数，大edge=不推（大分歧≠大机会）

---

## 三、评分引擎（预测优先版）

### 比赛盘评分（0-100）

| 维度 | 权重 | 评分逻辑 |
|------|------|----------|
| 模型信心 | 35分 | 模型预测赢家的概率（≥80%=35, ≥70%=25-35, ≥60%=15-25） |
| 价值边际 | 30分 | 混合概率 - 市场价格（用衰减后的effective edge） |
| Pinnacle一致性 | 20分 | 模型和Pinnacle是否同意赢家（一致=20, 不一致=0） |
| 伤病优势 | 10分 | RAPTOR Elo惩罚差值 |
| B2B休息 | 5分 | 对手B2B=加分 |

### 推送条件
```python
should_push = (
    model_prob >= adj_confidence       # 动态信心门槛（主/客场不同）
    and effective_value_edge >= 0.02   # 衰减后edge≥2%
    and buy_price <= 0.65              # 价格上限65¢
    and buy_price >= 0.30              # 不买<30¢冷门
    and final_score >= 45              # 最低评分45分
    and pinnacle_agrees                # Pinnacle必须与模型一致
)
```

### Edge可信度衰减
```
edge_confidence = 1 / (1 + (value_edge / 0.03)²)

raw_edge=2% → effective=1.7% (可信度85%)  ← 甜区
raw_edge=5% → effective=1.1% (可信度22%)  ← 可疑
raw_edge=10% → effective=0.9% (可信度9%) ← 不推
```

### 概率混合（贝叶斯 log-odds 空间）
```
blended = logit_mix(model, pinnacle, w=model_weight)
分歧收缩: 当 blended 与 market 差距过大时拉向 market
```

---

## 四、专家委员会（4-Agent审议）

所有交易类推送推前必须通过委员会审议：

| Agent | 职责 | 输出 |
|-------|------|------|
| 数据专家 | 近况、伤病（双方球员名单）、战绩、主客场 | 量化摘要 |
| 对位专家 | 节奏匹配、关键球员对位、风格克制 | 倾向判断 |
| 风控专家 | 价格合理性、风险点、历史教训 | 仓位建议 |
| 决策者 | 综合3位分析师 → buy/pass/wait | JSON决策 |

- 比赛盘：verdict="buy" 且 confidence≥60% 才推送
- 精选盘：verdict="buy" 且 confidence≥65% 才推送
- 期货盘：verdict="buy" 且 confidence≥60% 才推送
- 委员会收到双方球队完整伤病名单（球员名+状态+核心缺阵标注）

---

## 五、伤病系统

### 数据流
```
ESPN全局伤病端点 → _refresh_all_injuries() → DB injuries表
                                                   ↓
                              get_injury_impact() → 评分引擎（Elo惩罚）
                              InjuryChecker → 推送前核心球员检查
                              cleanup_recovered_players() → 每日清理
```

### 伤病量化 (RAPTOR模型)
```
Elo惩罚 = 球员RAPTOR值 × 状态乘数 × 12

状态乘数:
  Out = 1.0, Doubtful = 0.75, Questionable = 0.40
  Day-To-Day = 0.25, Probable = 0.10
```

### 每日伤病清理（v4新增）
- **07:00 + 17:00** 自动刷新ESPN伤病数据
- `cleanup_recovered_players()`: 重新拉取ESPN → DELETE旧数据 → INSERT新数据
- 不在新数据中的球员 = 已康复 → 自动清除
- 23:30 深度复盘时再次刷新

---

## 六、每晚深度复盘（v4新增）

### 23:30 `send_nightly_review()`

每晚比赛结束后自动运行，AI全面复盘当天所有比赛：

1. **回填结果** — 确保今天的比赛结果都拉到
2. **拉取全量信号** — 所有评分过的比赛（推送+未推送）
3. **AI深度分析**（6个维度）：
   - 今日总结（一句话概括）
   - 亮点（哪些预测准？为什么？）
   - 失误（哪些预测错了？具体原因+比分）
   - 模式发现（高评分是否更准？主客偏差？edge可信度？）
   - 明日建议（明天选比赛注意什么）
   - 策略调整建议（参数调整方向）
4. **推送到Telegram** — 完整复盘报告
5. **自我进化** — 基于全量信号调参
6. **刷新伤病** — 清除康复球员
7. **存赛后记忆** — 每场教训存入game_memory，下次遇到同队自动调取

### 与其他复盘机制的区别

| 机制 | 时间 | 范围 | 深度 |
|------|------|------|------|
| `send_daily_summary` | 09:00 | 推送信号 | 数字报表 |
| `send_nightly_review` | 23:30 | 全量信号 | AI深度分析+策略调整 |
| `game_memory.post_mortem` | 结果回填时 | 单场 | LLM分析+Box Score异常 |
| `_daily_evolution` | 复盘后 | 昨日推送 | 自动调参 |

---

## 七、每日时间线

```
06:00  更新球员评分
07:00  伤病数据刷新（清除康复球员）
08:00  期货扫描 → 委员会审议 → 推送（最多2个）
09:00  早报（数字报表：昨日胜负、7天ROI）
每1h   比赛扫描（积累候选）+ 结果回填
17:00  伤病数据再刷新（赛前）
19:00  每晚精选（最严门槛，最多2场）
23:30  深度复盘（AI分析全天比赛 + 存记忆 + 调参 + 清伤病）
00:00  重置每日推送计数 + 候选池 + 交易额度
```

---

## 八、去噪机制

1. **Pinnacle一致性**：模型和Pinnacle必须同意赢家方向
2. **Edge可信度衰减**：大edge不代表大机会，代表大分歧
3. **多次扫描验证**：扫描≥2次（观察2小时），确认信号稳定
4. **专家委员会**：4-Agent AI审议，信心不够则否决
5. **伤病实时检查**：核心球员缺阵可能否决推送
6. **24小时去重**：同一比赛只推一次
7. **每日限额**：最多10条推送
8. **价格区间过滤**：只在30-65¢区间买入
9. **客队惩罚**：客队需要更高门槛（历史胜率仅32%）

---

## 九、数据源

### Polymarket 数据
| 端点 | 用途 |
|------|------|
| Gamma API /events | 比赛/期货市场列表 |
| CLOB Token Prices | 真实买入/卖出价（优先于Gamma中间价） |

### 赔率源：The Odds API v4 (Pinnacle)
| 端点 | 用途 | 配额 |
|------|------|------|
| GET /odds | Pinnacle h2h赔率 | 1 credit/次 |
| GET /scores | 比赛结果回填 | 免费 |

### ESPN数据
| 数据类型 | TTL | 用途 |
|----------|-----|------|
| 战绩 | 30分钟 | 基本面 |
| 伤病 | 1小时 | RAPTOR量化 + 每日清理 |
| 赛程 | 24小时 | B2B检测 |

---

## 十、自学习系统

### AutoLearner（autoresearch风格）
- 每次有新结果时运行参数优化实验
- 自动调整 `scoring_params.json` 中的参数
- 评估指标：推送胜率 × ROI

### 每日自我进化
- 输的均价偏高 → 降低 max_buy_price
- 胜率>75% → 放宽信心门槛
- 胜率<40% → 收紧信心门槛

### 赛后记忆系统
- Box Score异常检测（球员表现远超/远低于均值）
- LLM分析输赢原因
- 存入game_memory，下次遇到同队自动调取

---

## 十一、运维

### 运行模式
```bash
python main.py              # 持续扫描（全部定时任务）
python main.py --once       # 单次扫描
python main.py --health     # 数据源健康检查
python main.py --report     # ML表现报告
python main.py --backtest   # 回测数据报告
python main.py --stats      # 详细统计报告
python main.py --summary    # 手动发送早报
python main.py --review     # 手动运行深度复盘
python main.py --learn      # 手动自学习
python main.py --learn-report # 自学习进度报告
python main.py --bootstrap  # 导入历史数据+训练模型
python main.py --notion-sync # 同步历史推送到Notion
```

### 部署
- 服务器: 46.250.232.42
- 服务: systemd (polymarket-monitor.service)
- 数据库: nba_predictor.db (SQLite)

---

## 十二、推送流程图

```
扫描比赛
    ↓
获取赔率 + CLOB真实价格
    ↓
MismatchEngine 评分（预测优先）
    ↓
加入候选池（记录最优评分）
    ↓
扫描≥2次后检查推送条件
    ↓
伤病检查（双方球队，联网查最新）
    ↓
专家委员会审议（4-Agent，含双方伤病分析）
    ↓
推送Telegram（带交易按钮）
    ↓
Notion同步记录
    ↓
比赛结束后回填结果
    ↓
23:30 深度复盘（AI分析 + 存记忆 + 调参数 + 清伤病）
```
