-- ============================================================================
-- Quantum Trader v3.0 - PostgreSQL Initialization
-- ============================================================================

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ============================================================================
-- EVENTS TABLE - Audit trail for all events
-- ============================================================================
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB NOT NULL,
    trace_id UUID,
    source VARCHAR(100),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_events_event_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX idx_events_trace_id ON events(trace_id);
CREATE INDEX idx_events_source ON events(source);
CREATE INDEX idx_events_data_gin ON events USING gin(event_data);

-- ============================================================================
-- POSITIONS TABLE - Trading positions history
-- ============================================================================
CREATE TABLE IF NOT EXISTS positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    size DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    exit_price DECIMAL(20, 8),
    pnl_usd DECIMAL(20, 8),
    pnl_pct DECIMAL(10, 4),
    leverage DECIMAL(10, 2),
    entry_confidence DECIMAL(5, 4),
    entry_model VARCHAR(50),
    exit_reason VARCHAR(50),
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_opened_at ON positions(opened_at DESC);
CREATE INDEX idx_positions_closed_at ON positions(closed_at DESC);
CREATE INDEX idx_positions_pnl ON positions(pnl_pct DESC);

-- ============================================================================
-- LEARNING_SAMPLES TABLE - Training data for continuous learning
-- ============================================================================
CREATE TABLE IF NOT EXISTS learning_samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction JSONB NOT NULL,
    outcome JSONB NOT NULL,
    features JSONB,
    prediction_confidence DECIMAL(5, 4),
    actual_pnl_pct DECIMAL(10, 4),
    prediction_error DECIMAL(10, 4),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_learning_samples_model ON learning_samples(model_name);
CREATE INDEX idx_learning_samples_timestamp ON learning_samples(timestamp DESC);
CREATE INDEX idx_learning_samples_symbol ON learning_samples(symbol);

-- ============================================================================
-- MODEL_PERFORMANCE TABLE - Model performance metrics
-- ============================================================================
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(20, 8) NOT NULL,
    sample_size INTEGER,
    window_start TIMESTAMPTZ,
    window_end TIMESTAMPTZ,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_model_performance_model ON model_performance(model_name);
CREATE INDEX idx_model_performance_created ON model_performance(created_at DESC);
CREATE INDEX idx_model_performance_metric ON model_performance(metric_name);

-- ============================================================================
-- HEALTH_EVENTS TABLE - Health monitoring events
-- ============================================================================
CREATE TABLE IF NOT EXISTS health_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL,
    details JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_health_events_service ON health_events(service_name);
CREATE INDEX idx_health_events_timestamp ON health_events(timestamp DESC);
CREATE INDEX idx_health_events_status ON health_events(status);

-- ============================================================================
-- ALERTS TABLE - System alerts and notifications
-- ============================================================================
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_alerts_type ON alerts(alert_type);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_resolved ON alerts(resolved);
CREATE INDEX idx_alerts_created ON alerts(created_at DESC);

-- ============================================================================
-- TRIGGERS - Auto-update timestamps
-- ============================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- VIEWS - Analytics views
-- ============================================================================

-- Daily PnL summary
CREATE OR REPLACE VIEW daily_pnl_summary AS
SELECT
    DATE(closed_at) as trade_date,
    COUNT(*) as total_trades,
    COUNT(*) FILTER (WHERE pnl_pct > 0) as winning_trades,
    COUNT(*) FILTER (WHERE pnl_pct < 0) as losing_trades,
    SUM(pnl_usd) as total_pnl_usd,
    AVG(pnl_pct) as avg_pnl_pct,
    MAX(pnl_pct) as max_pnl_pct,
    MIN(pnl_pct) as min_pnl_pct
FROM positions
WHERE closed_at IS NOT NULL
GROUP BY DATE(closed_at)
ORDER BY trade_date DESC;

-- Model performance summary
CREATE OR REPLACE VIEW model_performance_summary AS
SELECT
    entry_model,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE pnl_pct > 0) as winning_predictions,
    AVG(pnl_pct) as avg_pnl_pct,
    AVG(entry_confidence) as avg_confidence,
    SUM(pnl_usd) as total_pnl_usd
FROM positions
WHERE closed_at IS NOT NULL AND entry_model IS NOT NULL
GROUP BY entry_model
ORDER BY avg_pnl_pct DESC;

-- Service health summary
CREATE OR REPLACE VIEW service_health_summary AS
SELECT
    service_name,
    status,
    COUNT(*) as event_count,
    MAX(timestamp) as last_seen,
    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp))) as seconds_since_last_seen
FROM health_events
WHERE timestamp > NOW() - INTERVAL '1 hour'
GROUP BY service_name, status
ORDER BY service_name, status;

-- ============================================================================
-- GRANTS
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quantum_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quantum_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO quantum_user;

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert initial health event
INSERT INTO health_events (service_name, status, details)
VALUES ('database', 'HEALTHY', '{"message": "Database initialized successfully"}');

-- Log initialization
INSERT INTO events (event_type, event_data, source)
VALUES ('system.initialized', '{"version": "3.0.0", "timestamp": "2025-12-02T00:00:00Z"}', 'database');
