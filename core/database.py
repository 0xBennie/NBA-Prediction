"""
database.py — SQLite 数据库初始化与操作封装
"""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA = """
-- ── 比赛盘推送去重 ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerted_games (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      TEXT UNIQUE NOT NULL,
    score        INTEGER,
    edge         REAL,
    source       TEXT,
    pushed_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 期货盘推送去重 ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerted_futures (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    condition_id TEXT UNIQUE NOT NULL,
    score        INTEGER,
    pushed_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 伤病缓存 ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS injuries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_abbr   TEXT NOT NULL,
    player_name TEXT NOT NULL,
    status      TEXT,
    impact      REAL DEFAULT 0,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team_abbr);

-- ── 战绩缓存 ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS standings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    team_abbr    TEXT UNIQUE NOT NULL,
    wins         INTEGER DEFAULT 0,
    losses       INTEGER DEFAULT 0,
    win_rate     REAL DEFAULT 0,
    ppg          REAL DEFAULT 0,
    opp_ppg      REAL DEFAULT 0,
    ppg_diff     REAL DEFAULT 0,
    games_played INTEGER DEFAULT 0,
    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 球员影响力评分（自动更新，替代硬编码RAPTOR）──────────────────
CREATE TABLE IF NOT EXISTS player_ratings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name  TEXT UNIQUE NOT NULL,
    team_abbr    TEXT,
    impact_score REAL NOT NULL DEFAULT 0,
    games_played INTEGER DEFAULT 0,
    pts_per_game REAL DEFAULT 0,
    reb_per_game REAL DEFAULT 0,
    ast_per_game REAL DEFAULT 0,
    plus_minus   REAL DEFAULT 0,
    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 价格时序（盘口移动追踪）────────────────────────────────────
CREATE TABLE IF NOT EXISTS price_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id             TEXT NOT NULL,
    timestamp           INTEGER NOT NULL,
    poly_price_home     REAL,
    poly_price_away     REAL,
    pinnacle_fair_home  REAL,
    pinnacle_fair_away  REAL,
    volume_24h          REAL,
    market_type         TEXT,
    condition_id        TEXT,
    days_to_resolution  INTEGER,
    created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_price_history_game
    ON price_history(game_id, timestamp DESC);

-- ── ML特征存档 ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_features (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    push_id       TEXT UNIQUE NOT NULL,
    market_type   TEXT NOT NULL,
    features_json TEXT NOT NULL,
    rule_score    INTEGER,
    ml_adjustment REAL DEFAULT 0,
    pushed_at     DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 推送结果记录（ML训练数据来源）──────────────────────────────
CREATE TABLE IF NOT EXISTS push_results (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    push_id            TEXT UNIQUE NOT NULL,
    market_type        TEXT,
    game_id            TEXT,
    away_team          TEXT,
    home_team          TEXT,
    poly_price_at_push REAL,
    pinnacle_prob      REAL,
    edge_at_push       REAL,
    actual_outcome     INTEGER,   -- 1=我们推的方向赢了, 0=输了, NULL=未结算
    pnl_per_unit       REAL,
    resolved_at        DATETIME,
    created_at         DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_push_results_unresolved
    ON push_results(actual_outcome) WHERE actual_outcome IS NULL;

-- ── 信号日志（全量回测数据）──────────────────────────────────────
CREATE TABLE IF NOT EXISTS signal_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id              TEXT NOT NULL,
    away_team            TEXT,
    home_team            TEXT,
    market_type          TEXT DEFAULT 'game',
    buy_side             TEXT,             -- "home" 或 "away"
    buy_team             TEXT,             -- 买入的队名缩写
    source               TEXT,             -- 赔率源 (pinnacle/draftkings/consensus)
    poly_price           REAL,             -- 扫描时Polymarket买入价
    pinnacle_prob        REAL,             -- 公平概率
    raw_edge             REAL,             -- 原始edge
    effective_edge       REAL,             -- 扣除成本后edge
    score                INTEGER,          -- 最终评分
    kelly                REAL,             -- ¼Kelly仓位建议
    was_pushed           INTEGER DEFAULT 0,-- 是否达到推送阈值
    breakdown_json       TEXT,             -- 评分明细JSON
    actual_outcome       INTEGER,          -- 1=客队赢, 0=客队输, NULL=未结算
    actual_score_home    INTEGER,
    actual_score_away    INTEGER,
    prediction_correct   INTEGER,          -- 1=我们推的方向赢了, 0=输了, NULL=未结算
    actual_roi           REAL,             -- 实际ROI: 赢=(1/price-1)-费, 输=-1.0
    hypo_pnl             REAL,             -- 假设按Kelly买入的盈亏
    resolved_at          DATETIME,
    scanned_at           DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_signal_log_game ON signal_log(game_id);
CREATE INDEX IF NOT EXISTS idx_signal_log_unresolved
    ON signal_log(actual_outcome) WHERE actual_outcome IS NULL;
"""


class Database:

    def __init__(self, db_path: str = "nba_predictor.db"):
        self.db_path = db_path
        self._init()

    def _init(self):
        with self.conn() as c:
            c.executescript(SCHEMA)
            # 迁移：给旧signal_log表加新字段
            existing = {row[1] for row in c.execute("PRAGMA table_info(signal_log)")}
            migrations = [
                ("buy_side", "TEXT"),
                ("buy_team", "TEXT"),
                ("source", "TEXT"),
                ("prediction_correct", "INTEGER"),
                ("actual_roi", "REAL"),
            ]
            for col, typ in migrations:
                if col not in existing:
                    c.execute(f"ALTER TABLE signal_log ADD COLUMN {col} {typ}")
                    logger.info(f"[DB] 迁移: signal_log +{col}")
        logger.info(f"[DB] 初始化完成: {self.db_path}")

    @contextmanager
    def conn(self):
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def execute(self, sql: str, params: tuple = ()):
        with self.conn() as c:
            return c.execute(sql, params).fetchall()

    def execute_one(self, sql: str, params: tuple = ()):
        with self.conn() as c:
            row = c.execute(sql, params).fetchone()
            return dict(row) if row else None

    def insert(self, sql: str, params: tuple = ()):
        with self.conn() as c:
            cur = c.execute(sql, params)
            return cur.lastrowid
