-- ============================================================================
-- ML/AI Pipeline Database Schemas
-- Creates all tables for the continuous learning system
-- ============================================================================

-- Drop existing tables (for clean setup)
DROP TABLE IF EXISTS model_performance_logs CASCADE;
DROP TABLE IF EXISTS drift_events CASCADE;
DROP TABLE IF EXISTS retraining_jobs CASCADE;
DROP TABLE IF EXISTS shadow_test_results CASCADE;
DROP TABLE IF EXISTS rl_versions CASCADE;
DROP TABLE IF EXISTS model_registry CASCADE;

-- ============================================================================
-- 1. Model Registry
-- ============================================================================

CREATE TABLE model_registry (
    model_id VARCHAR(255) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    metrics JSONB NOT NULL,
    training_config JSONB,
    training_data_range JSONB,
    feature_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    promoted_at TIMESTAMP,
    retired_at TIMESTAMP,
    file_path TEXT,
    file_size_bytes BIGINT,
    notes TEXT,
    UNIQUE(model_type, version)
);

CREATE INDEX idx_model_type_status ON model_registry(model_type, status);
CREATE INDEX idx_created_at ON model_registry(created_at DESC);
CREATE INDEX idx_promoted_at ON model_registry(promoted_at DESC) WHERE promoted_at IS NOT NULL;

COMMENT ON TABLE model_registry IS 'Stores metadata for all trained models';
COMMENT ON COLUMN model_registry.status IS 'Model lifecycle status: TRAINING, SHADOW, ACTIVE, RETIRED, FAILED';
COMMENT ON COLUMN model_registry.metrics IS 'Training metrics (RMSE, accuracy, etc.)';

-- ============================================================================
-- 2. Shadow Test Results
-- ============================================================================

CREATE TABLE shadow_test_results (
    prediction_id VARCHAR(255) PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    prediction_value DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    features JSONB,
    actual_outcome DOUBLE PRECISION,
    outcome_timestamp TIMESTAMP,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id) ON DELETE CASCADE
);

CREATE INDEX idx_shadow_model_timestamp ON shadow_test_results(model_id, timestamp DESC);
CREATE INDEX idx_shadow_symbol_timestamp ON shadow_test_results(symbol, timestamp DESC);
CREATE INDEX idx_shadow_outcome ON shadow_test_results(outcome_timestamp DESC) WHERE actual_outcome IS NOT NULL;

COMMENT ON TABLE shadow_test_results IS 'Stores predictions from shadow models for comparison with active models';
COMMENT ON COLUMN shadow_test_results.actual_outcome IS 'Actual PnL outcome after trade closes';

-- ============================================================================
-- 3. RL Versions
-- ============================================================================

CREATE TABLE rl_versions (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    checkpoint_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    metrics JSONB,
    UNIQUE(version, agent_type)
);

CREATE INDEX idx_rl_version ON rl_versions(version, agent_type);
CREATE INDEX idx_rl_created ON rl_versions(created_at DESC);

COMMENT ON TABLE rl_versions IS 'Stores checkpoints for RL agents (meta_strategy, position_sizing)';
COMMENT ON COLUMN rl_versions.agent_type IS 'Type of RL agent: meta_strategy, position_sizing';

-- ============================================================================
-- 4. Drift Events
-- ============================================================================

CREATE TABLE drift_events (
    id SERIAL PRIMARY KEY,
    drift_type VARCHAR(50) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    drift_score DOUBLE PRECISION NOT NULL,
    p_value DOUBLE PRECISION NOT NULL,
    threshold DOUBLE PRECISION NOT NULL,
    model_type VARCHAR(50),
    feature_name VARCHAR(255),
    detection_time TIMESTAMP NOT NULL,
    reference_stats JSONB,
    current_stats JSONB,
    trigger_retraining BOOLEAN DEFAULT FALSE,
    notes TEXT
);

CREATE INDEX idx_drift_time ON drift_events(detection_time DESC);
CREATE INDEX idx_drift_model ON drift_events(model_type, detection_time DESC);
CREATE INDEX idx_drift_trigger ON drift_events(trigger_retraining, detection_time DESC) WHERE trigger_retraining = TRUE;

COMMENT ON TABLE drift_events IS 'Records distribution drift detections (features, predictions, performance)';
COMMENT ON COLUMN drift_events.drift_type IS 'Type: FEATURE, PREDICTION, PERFORMANCE';
COMMENT ON COLUMN drift_events.severity IS 'Severity: LOW, MEDIUM, HIGH, CRITICAL';

-- ============================================================================
-- 5. Model Performance Logs
-- ============================================================================

CREATE TABLE model_performance_logs (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    n_trades INT NOT NULL,
    winrate DOUBLE PRECISION NOT NULL,
    winrate_ci_lower DOUBLE PRECISION NOT NULL,
    winrate_ci_upper DOUBLE PRECISION NOT NULL,
    avg_win DOUBLE PRECISION NOT NULL,
    avg_loss DOUBLE PRECISION NOT NULL,
    profit_factor DOUBLE PRECISION NOT NULL,
    sharpe_ratio DOUBLE PRECISION NOT NULL,
    accuracy DOUBLE PRECISION,
    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1 DOUBLE PRECISION,
    calibration_error DOUBLE PRECISION,
    directional_bias DOUBLE PRECISION,
    volatility_bias DOUBLE PRECISION,
    FOREIGN KEY (model_id) REFERENCES model_registry(model_id) ON DELETE CASCADE
);

CREATE INDEX idx_perf_model_period ON model_performance_logs(model_id, period_end DESC);
CREATE INDEX idx_perf_type_period ON model_performance_logs(model_type, period_end DESC);

COMMENT ON TABLE model_performance_logs IS 'Tracks model performance metrics over time';
COMMENT ON COLUMN model_performance_logs.calibration_error IS 'Mean absolute calibration error';

-- ============================================================================
-- 6. Retraining Jobs
-- ============================================================================

CREATE TABLE retraining_jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(255) UNIQUE NOT NULL,
    retraining_type VARCHAR(50) NOT NULL,
    model_types JSONB NOT NULL,
    trigger_reason TEXT NOT NULL,
    data_start_date TIMESTAMP NOT NULL,
    data_end_date TIMESTAMP NOT NULL,
    training_config JSONB NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    models_trained INT DEFAULT 0,
    models_succeeded INT DEFAULT 0,
    models_failed INT DEFAULT 0,
    trained_model_ids JSONB,
    error_message TEXT
);

CREATE INDEX idx_retrain_status ON retraining_jobs(status, created_at DESC);
CREATE INDEX idx_retrain_created ON retraining_jobs(created_at DESC);

COMMENT ON TABLE retraining_jobs IS 'Tracks automated retraining job execution';
COMMENT ON COLUMN retraining_jobs.retraining_type IS 'Type: FULL, PARTIAL, INCREMENTAL';
COMMENT ON COLUMN retraining_jobs.status IS 'Status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED';

-- ============================================================================
-- Sample Data / Views
-- ============================================================================

-- View: Active Models Summary
CREATE OR REPLACE VIEW active_models_summary AS
SELECT 
    model_type,
    model_id,
    version,
    metrics->>'val_rmse' as val_rmse,
    metrics->>'val_accuracy' as val_accuracy,
    feature_count,
    promoted_at,
    created_at
FROM model_registry
WHERE status = 'active'
ORDER BY model_type;

COMMENT ON VIEW active_models_summary IS 'Quick overview of currently active models';

-- View: Shadow Models Summary
CREATE OR REPLACE VIEW shadow_models_summary AS
SELECT 
    model_type,
    model_id,
    version,
    metrics->>'val_rmse' as val_rmse,
    metrics->>'val_accuracy' as val_accuracy,
    feature_count,
    created_at,
    (SELECT COUNT(*) FROM shadow_test_results WHERE shadow_test_results.model_id = model_registry.model_id) as prediction_count
FROM model_registry
WHERE status = 'shadow'
ORDER BY model_type, created_at DESC;

COMMENT ON VIEW shadow_models_summary IS 'Overview of shadow models being tested';

-- View: Recent Drift Events
CREATE OR REPLACE VIEW recent_drift_events AS
SELECT 
    drift_type,
    severity,
    model_type,
    feature_name,
    drift_score,
    p_value,
    trigger_retraining,
    detection_time
FROM drift_events
WHERE detection_time >= NOW() - INTERVAL '7 days'
ORDER BY detection_time DESC;

COMMENT ON VIEW recent_drift_events IS 'Drift events from the last 7 days';

-- View: Model Performance Dashboard
CREATE OR REPLACE VIEW model_performance_dashboard AS
SELECT 
    mpl.model_type,
    mpl.model_id,
    mpl.period_end,
    mpl.n_trades,
    mpl.winrate,
    mpl.profit_factor,
    mpl.sharpe_ratio,
    mpl.calibration_error,
    mr.status as model_status
FROM model_performance_logs mpl
JOIN model_registry mr ON mpl.model_id = mr.model_id
WHERE mpl.period_end >= NOW() - INTERVAL '30 days'
ORDER BY mpl.period_end DESC;

COMMENT ON VIEW model_performance_dashboard IS 'Performance metrics for the last 30 days';

-- ============================================================================
-- Maintenance Functions
-- ============================================================================

-- Function: Clean old shadow test results (keep last 90 days)
CREATE OR REPLACE FUNCTION cleanup_old_shadow_results()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM shadow_test_results
    WHERE timestamp < NOW() - INTERVAL '90 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_shadow_results IS 'Deletes shadow test results older than 90 days';

-- Function: Archive retired models
CREATE OR REPLACE FUNCTION archive_retired_models()
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- Mark old retired models for archival
    UPDATE model_registry
    SET notes = COALESCE(notes || ' ', '') || '[ARCHIVED]'
    WHERE status = 'retired'
      AND retired_at < NOW() - INTERVAL '180 days'
      AND notes NOT LIKE '%[ARCHIVED]%';
    
    GET DIAGNOSTICS archived_count = ROW_COUNT;
    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION archive_retired_models IS 'Marks retired models older than 180 days for archival';

-- ============================================================================
-- Initial Setup Complete
-- ============================================================================

-- Grant permissions (adjust as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO quantum_trader_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO quantum_trader_app;

SELECT 'ML/AI Pipeline database schema created successfully!' AS status;
