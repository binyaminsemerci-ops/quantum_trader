CREATE TABLE IF NOT EXISTS trade_journal (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    direction VARCHAR(10),
    entry_price FLOAT,
    exit_price FLOAT,
    pnl FLOAT,
    tp FLOAT,
    sl FLOAT,
    trailing_stop FLOAT,
    confidence FLOAT,
    model VARCHAR(50),
    features JSONB,
    policy_state JSONB,
    exit_reason VARCHAR(50) DEFAULT 'open',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_journal_symbol ON trade_journal(symbol);
CREATE INDEX IF NOT EXISTS idx_journal_timestamp ON trade_journal(timestamp);
