-- database/schema.sql

-- Drop existing tables in dependency order so the script is idempotent.
DROP TABLE IF EXISTS execution_journal;
DROP TABLE IF EXISTS portfolio_allocations;
DROP TABLE IF EXISTS liquidity_snapshots;
DROP TABLE IF EXISTS liquidity_runs;
DROP TABLE IF EXISTS portfolio_positions;
DROP TABLE IF EXISTS training_tasks;
DROP TABLE IF EXISTS trade_logs;
DROP TABLE IF EXISTS settings;
DROP TABLE IF EXISTS trades;
DROP TABLE IF EXISTS stats;

-- Core trading tables used by the FastAPI backend.
CREATE TABLE trade_logs (
    id INTEGER PRIMARY KEY,
    symbol TEXT,
    side TEXT,
    qty REAL,
    price REAL,
    status TEXT,
    reason TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX ix_trade_logs_symbol ON trade_logs (symbol);

CREATE TABLE settings (
    id INTEGER PRIMARY KEY,
    api_key TEXT,
    api_secret TEXT
);

CREATE TABLE training_tasks (
    id INTEGER PRIMARY KEY,
    symbols TEXT NOT NULL,
    "limit" INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    details TEXT
);

CREATE INDEX ix_training_tasks_status ON training_tasks (status);

-- Liquidity universe, allocations, and execution journal tables.
CREATE TABLE liquidity_runs (
    id INTEGER PRIMARY KEY,
    fetched_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    universe_size INTEGER NOT NULL,
    selection_size INTEGER NOT NULL,
    provider_primary TEXT,
    status TEXT NOT NULL DEFAULT 'completed',
    message TEXT
);

CREATE INDEX ix_liquidity_runs_fetched_at ON liquidity_runs (fetched_at);

CREATE TABLE liquidity_snapshots (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    price REAL,
    change_percent REAL,
    base_volume REAL,
    quote_volume REAL,
    market_cap REAL,
    liquidity_score REAL NOT NULL,
    momentum_score REAL,
    aggregate_score REAL NOT NULL,
    payload TEXT,
    FOREIGN KEY (run_id) REFERENCES liquidity_runs(id) ON DELETE CASCADE,
    UNIQUE (run_id, symbol)
);

CREATE INDEX ix_liquidity_snapshots_run_id ON liquidity_snapshots (run_id);
CREATE INDEX ix_liquidity_snapshots_symbol ON liquidity_snapshots (symbol);

CREATE TABLE portfolio_allocations (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    weight REAL NOT NULL,
    score REAL NOT NULL,
    reason TEXT,
    FOREIGN KEY (run_id) REFERENCES liquidity_runs(id) ON DELETE CASCADE,
    UNIQUE (run_id, symbol)
);

CREATE INDEX ix_portfolio_allocations_run_id ON portfolio_allocations (run_id);
CREATE INDEX ix_portfolio_allocations_symbol ON portfolio_allocations (symbol);

CREATE TABLE execution_journal (
    id INTEGER PRIMARY KEY,
    run_id INTEGER,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    target_weight REAL NOT NULL,
    quantity REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT,
    error TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    executed_at DATETIME,
    FOREIGN KEY (run_id) REFERENCES liquidity_runs(id) ON DELETE SET NULL
);

CREATE INDEX ix_execution_journal_run_id ON execution_journal (run_id);
CREATE INDEX ix_execution_journal_status ON execution_journal (status);
CREATE INDEX ix_execution_journal_symbol ON execution_journal (symbol);

-- Portfolio exposure snapshot used by the risk guard and scheduler telemetry.
CREATE TABLE portfolio_positions (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    quantity REAL NOT NULL DEFAULT 0,
    notional REAL NOT NULL DEFAULT 0,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (symbol)
);

-- Legacy demo tables retained for seed scripts and UI demos.
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    qty REAL NOT NULL,
    pnl REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE stats (
    id INTEGER PRIMARY KEY,
    balance REAL NOT NULL,
    total_pnl REAL NOT NULL,
    win_rate REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
