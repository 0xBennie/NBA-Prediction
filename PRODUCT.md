# Polymarket NBA 错配扫描器 — 产品文档 v2

## 核心理念

**只在高确信度的价值洼地出现时推送买入信号。宁可漏掉机会，也不发垃圾信号。**

每条推送都应该让用户感到："这个我要认真看看"，而不是"又来了一条"。

---

## 一、系统架构

```
Polymarket Gamma API ──→ Scanner ──→ MismatchEngine ──→ Telegram推送
                              ↕            ↕
                     SportsbookClient   MLPredictor
                     (4级瀑布赔率源)    (ML修正层)
                              ↕            ↕
                         ESPNClient     ResultResolver
                     (伤病/B2B/战绩)    (结果回填)
                              ↕
                          Database
                        (7张SQLite表)
```

### 项目结构
```
prediction-market/
├── main.py              # 主入口 + Telegram推送 + Polymarket数据获取
├── core/
│   ├── database.py      # 7张表SQLite封装
│   ├── sportsbook_client.py  # 4级瀑布赔率源
│   ├── espn_client.py   # ESPN战绩/伤病(RAPTOR)/B2B
│   └── mismatch_engine.py    # 核心评分引擎
├── ml/
│   ├── ml_predictor.py  # ML修正层(LR→LightGBM)
│   └── result_resolver.py    # 比赛结果回填
├── ai_analyzer.py       # MiniMax AI分析(保留)
├── config.py            # 环境变量配置
└── requirements.txt
```

---

## 二、市场分类与策略

### 1. 单场比赛盘（Game-Day）
- 场景：某队今晚打比赛，Polymarket有胜负盘
- 数据：**Pinnacle去水赔率**(优先) + ESPN伤病(RAPTOR量化) + B2B检测
- 策略：发现Polymarket定价与Pinnacle公平概率的偏差 → 买入被低估方

### 2. 期货盘（Futures）
- 场景：赛季冠军、MVP、最佳新秀、分区冠军等
- 数据：ESPN战绩 + 价格走势 + 市场深度 + AI分析
- 策略：寻找被市场忽视的价值标的

---

## 三、数据源架构

### 赔率源：The Odds API v4 (Pinnacle)
| 端点 | 用途 | 配额消耗 | 去vig方法 |
|------|------|---------|-----------|
| GET /odds | Pinnacle h2h赔率 | 1 credit/次 | Shin法 |
| GET /scores | 比赛结果回填 | 免费 | — |
| GET /sports | 健康检查 | 免费 | — |

- **唯一数据源**: Pinnacle（sharp book锚，低利润率+接受职业投注 → 最接近真实概率）
- **批量获取**: 每次扫描1次API调用拉取所有NBA赔率，5分钟缓存
- 无Pinnacle赔率 → 跳过该场比赛，**不推送**
- 配额跟踪: 响应头 `x-requests-remaining` / `x-requests-used`

### ESPN数据
| 数据类型 | TTL | 用途 |
|----------|-----|------|
| 战绩 | 30分钟 | 基本面评分 |
| 伤病 | 1小时 | RAPTOR量化影响 |
| 赛程 | 24小时 | 背靠背(B2B)检测 |

### 伤病量化 (RAPTOR模型)
```
Elo惩罚 = 球员RAPTOR值 × 状态乘数 × 12

状态乘数:
  Out = 1.0, Doubtful = 0.75, Questionable = 0.40
  Day-To-Day = 0.25, Probable = 0.10
```

---

## 四、推送标准（评分体系）

### 原则
1. **只推送 BUY 信号**
2. **规则评分 + ML修正 → 最终评分 ≥ 阈值 才推送**
3. **只在价格相对低位时推送**
4. **同一标的24小时内只推送一次**
5. **每日最多推送10条**

### 单场比赛盘评分（0-100）

| 维度 | 权重 | 评分逻辑 |
|------|------|----------|
| Pinnacle边际(edge) | 40分 | Pinnacle公平概率 - Polymarket价格。3%=15, 5%=25, 8%+=40 |
| 盘口移动 | 25分 | 近期价格变动幅度+方向 |
| 伤病影响 | 20分 | RAPTOR Elo惩罚差值(主队伤更重=客队利好) |
| 价格位置 | 10分 | 10-25¢=10, 25-35¢=7, >35¢=3 |
| 背靠背(B2B) | 5分 | 对手B2B第二场=加分，自己B2B=减分 |

**规则阈值: 70分 → 推送**
**ML修正: ±15分** (冷启动<30条数据时不介入)

### 期货盘评分（0-100）

| 维度 | 权重 | 评分逻辑 |
|------|------|----------|
| 价格位置 | 25分 | 5-15¢=25, 15-25¢=20, 25-35¢=10, >35¢=0 |
| 基本面支撑 | 25分 | 球队/球员当前表现 vs 价格是否匹配 |
| 价格动量 | 20分 | 24h/7d价格趋势 |
| 成交量异常 | 15分 | 24h成交量占比突然放大 |
| 市场深度 | 15分 | 买卖价差(窄=流动性好) |
| 结算时间 | -10~0 | >90天扣分，>120天不推送 |

**规则阈值: 65分 → 推送**
**ML修正: ±15分**

---

## 五、ML预测层

### 模型进化路径
| 样本量 | 模型 | 说明 |
|--------|------|------|
| < 30条 | 不介入 | 冷启动，输出0修正 |
| 30-100 | LogisticRegression(C=0.1) | 强正则化防过拟合 |
| 100+ | LightGBM | 树模型，捕捉非线性 |

### 比赛盘特征（8维）
edge, poly_price, injury_delta, home_b2b, away_b2b, line_moved, volume_ratio, source_quality

### 期货盘特征（8维）
price, fundamental_edge, daily_change, weekly_change, volume_ratio, depth_score, days_to_resolution, market_type_enc

### 训练数据闭环
```
推送时存特征快照(ml_features) → 比赛结束后回填结果(push_results)
→ 每新增10条结算数据自动重训练 → 模型更新
```

### Kelly仓位建议
```
kelly_fraction = (win_prob × odds - (1 - win_prob)) / odds
建议仓位 = min(kelly / 4, 5%)  # 1/4 Kelly保守策略
```

---

## 六、推送格式

### 比赛盘推送 (HTML格式)
```
🏀 NBA比赛盘套利信号
━━━━━━━━━━━━━━━
📍 LAL 🔴背靠背 客场 @ BOS
💰 Polymarket客队: 35.2%
📊 Pinnacle公平值: 42.5%（来源: pinnacle_odds_api）
📈 套利边际: +7.3%
━━━━━━━━━━━━━━━
🎯 规则评分: 72/100
🤖 ML修正: +3.5
🏆 最终评分: 76/100
━━━━━━━━━━━━━━━
伤病影响: 主队 -45Elo / 客队 -12Elo
💡 Kelly建议仓位: 2.3%
⏰ 2026-03-14 20:30
```

### 期货盘推送
```
📅 NBA期货盘信号
━━━━━━━━━━━━━━━
📍 OKC — CHAMPION
💰 当前价格: 18.0%
📈 24h变动: +2.1%
📈 7d变动: +5.3%
━━━━━━━━━━━━━━━
🎯 规则评分: 70/100
🤖 ML修正: +2.0
🏆 最终评分: 72/100
💡 Kelly建议仓位: 1.8%
⏰ 2026-03-14 08:00
```

---

## 七、扫描频率

| 市场类型 | 扫描间隔 | 说明 |
|----------|---------|------|
| 单场比赛盘 | 1小时 | 每小时扫描一次 |
| 期货盘 | 每天08:00 | 期货价格变化慢 |
| 每日推送计数 | 00:00重置 | 防止信息过载 |

---

## 八、去噪机制

1. **24小时去重**：同一game_id/condition_id 24小时内只推送一次
2. **价格高位过滤**：价格 > 40¢ 的期货盘不推送
3. **低流动性过滤**：成交量 < $10,000 的市场不推送
4. **赔率源验证**：4级瀑布全部失败则不推送（不猜概率）
5. **结算时间过滤**：>120天的期货不推送
6. **每日限额**：每天最多推送10条

---

## 九、数据库 (SQLite, 7张表)

| 表名 | 用途 |
|------|------|
| alerted_games | 比赛盘推送去重 |
| alerted_futures | 期货盘推送去重 |
| injuries | 伤病缓存(含RAPTOR影响值) |
| standings | 战绩缓存(含胜率/PPG差) |
| price_history | 价格时序(盘口移动追踪) |
| ml_features | ML特征快照(推送时存入) |
| push_results | 推送结果(比赛结束后回填) |

---

## 十、退出策略

| 类型 | 条件 | 操作 |
|------|------|------|
| 止盈1 | +10¢ 或 +50% | 卖出30% |
| 止盈2 | +20¢ 或 +100% | 卖出40% |
| 止损 | -40% | 全部卖出 |
| 价格飙升 | +8¢ 快速上涨 | 提醒考虑部分止盈 |

---

## 十一、运维

### 运行模式
```bash
python main.py              # 持续扫描（含自动结果回填）
python main.py --once       # 单次扫描
python main.py --health     # 数据源健康检查 + API配额
python main.py --report     # ML表现报告
python -m ml.result_resolver         # 手动结果回填
python -m ml.result_resolver --report  # 查看胜率报告
```

### 定时任务（内置schedule）
| 任务 | 时间 | 说明 |
|------|------|------|
| scan_games | 每小时 | 扫描比赛盘 |
| scan_futures | 每天08:00 | 扫描期货盘 |
| resolve_results | 每天06:00 | 回填昨天比赛结果（免费API） |
| reset_daily_count | 每天00:00 | 重置推送计数 |

### 部署
- 服务器: 46.250.232.42
- 服务: systemd (polymarket-monitor.service)
- 数据库: nba_predictor.db (SQLite)
