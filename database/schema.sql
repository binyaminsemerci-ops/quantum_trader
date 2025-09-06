-- database/schema.sql

DROP TABLE IF EXISTS trades;
DROP TABLE IF EXISTS stats;

CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    qty REAL NOT NULL,
    pnl REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL NOT NULL,
    total_pnl REAL NOT NULL,
    win_rate REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
