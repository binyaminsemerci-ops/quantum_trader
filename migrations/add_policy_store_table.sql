-- Migration: Add policy_store table for Strategy Runtime Engine
-- Purpose: Store global trading policies with Redis fallback support
-- Date: 2025-01-XX

CREATE TABLE IF NOT EXISTS policy_store (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_policy_store_updated_at ON policy_store(updated_at);

-- Insert default policies
INSERT INTO policy_store (key, value, updated_at)
VALUES 
    ('risk_mode', 'NORMAL', CURRENT_TIMESTAMP),
    ('min_confidence', '0.5', CURRENT_TIMESTAMP)
ON CONFLICT (key) DO NOTHING;

-- Add comment
COMMENT ON TABLE policy_store IS 'Global trading policies set by Meta Strategy Controller';
