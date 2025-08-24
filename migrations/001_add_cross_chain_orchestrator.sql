-- Cross-Chain Orchestrator Database Schema
-- Migration: 001_add_cross_chain_orchestrator
-- Date: 2025-08-23
-- Description: Add tables for cross-chain strategy orchestration

-- Enable UUID extension if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Cross-chain strategies table
CREATE TABLE IF NOT EXISTS cross_chain_strategies (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_address VARCHAR(42) NOT NULL,
    
    -- Strategy configuration
    source_chain VARCHAR(50) NOT NULL,
    target_chain VARCHAR(50) NOT NULL DEFAULT 'hyperliquid',
    source_token VARCHAR(20) NOT NULL,
    target_token VARCHAR(20) NOT NULL,
    amount DECIMAL(20, 8) NOT NULL,
    risk_tolerance VARCHAR(20) DEFAULT 'medium',
    
    -- Status and execution
    status VARCHAR(30) DEFAULT 'pending' NOT NULL,
    strategy_config JSONB DEFAULT '{}',
    automation_rules_config JSONB DEFAULT '[]',
    ai_generated BOOLEAN DEFAULT FALSE,
    
    -- Route and execution data
    route_quotes JSONB DEFAULT '[]',
    selected_route_data JSONB,
    selected_route_index INTEGER,
    
    -- Costs and timing
    total_fees_usd DECIMAL(10, 2) DEFAULT 0.0,
    estimated_completion TIMESTAMP,
    actual_completion TIMESTAMP,
    
    -- AI analysis results
    ai_confidence_score DECIMAL(5, 2),
    ai_analysis_data JSONB,
    risk_assessment_data JSONB,
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Bridge transactions table
CREATE TABLE IF NOT EXISTS bridge_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES cross_chain_strategies(id) ON DELETE CASCADE,
    
    -- Transaction details
    source_tx_hash VARCHAR(66),
    target_tx_hash VARCHAR(66),
    bridge_provider VARCHAR(50) NOT NULL,
    
    -- Transaction data
    source_chain VARCHAR(50) NOT NULL,
    target_chain VARCHAR(50) NOT NULL,
    source_token VARCHAR(20) NOT NULL,
    target_token VARCHAR(20) NOT NULL,
    amount DECIMAL(20, 8) NOT NULL,
    
    -- Status and costs
    status VARCHAR(30) DEFAULT 'pending' NOT NULL,
    fee_usd DECIMAL(10, 2) DEFAULT 0.0,
    gas_fee_usd DECIMAL(10, 2) DEFAULT 0.0,
    actual_output_amount DECIMAL(20, 8),
    
    -- Timing
    estimated_time_minutes INTEGER,
    execution_started_at TIMESTAMP,
    execution_completed_at TIMESTAMP,
    
    -- Technical details
    route_data JSONB,
    slippage_tolerance DECIMAL(5, 4) DEFAULT 0.01,
    price_impact DECIMAL(5, 4),
    
    -- Error handling
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Strategy execution logs table
CREATE TABLE IF NOT EXISTS strategy_execution_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES cross_chain_strategies(id) ON DELETE CASCADE,
    
    -- Log details
    log_level VARCHAR(20) NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    details JSONB,
    
    -- Context
    execution_step VARCHAR(50),
    provider VARCHAR(50),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Strategy automation rules table
CREATE TABLE IF NOT EXISTS strategy_automation_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    strategy_id UUID NOT NULL REFERENCES cross_chain_strategies(id) ON DELETE CASCADE,
    
    -- Rule details
    rule_type VARCHAR(50) NOT NULL,
    rule_config JSONB NOT NULL,
    
    -- Status
    status VARCHAR(30) DEFAULT 'pending' NOT NULL,
    external_rule_id VARCHAR(100),
    
    -- Execution tracking
    execution_count INTEGER DEFAULT 0,
    last_execution TIMESTAMP,
    last_error TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Orchestrator statistics table
CREATE TABLE IF NOT EXISTS orchestrator_statistics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Date and metrics
    date TIMESTAMP NOT NULL,
    
    -- Strategy counts
    total_strategies INTEGER DEFAULT 0,
    completed_strategies INTEGER DEFAULT 0,
    failed_strategies INTEGER DEFAULT 0,
    cancelled_strategies INTEGER DEFAULT 0,
    
    -- Provider usage
    lifi_usage_count INTEGER DEFAULT 0,
    gluex_usage_count INTEGER DEFAULT 0,
    liquid_labs_usage_count INTEGER DEFAULT 0,
    
    -- Financial metrics
    total_volume_usd DECIMAL(15, 2) DEFAULT 0.0,
    total_fees_usd DECIMAL(10, 2) DEFAULT 0.0,
    average_fee_usd DECIMAL(10, 2) DEFAULT 0.0,
    
    -- Performance metrics
    average_execution_time_minutes DECIMAL(8, 2) DEFAULT 0.0,
    success_rate_percentage DECIMAL(5, 2) DEFAULT 0.0,
    
    -- AI metrics
    ai_generated_strategies INTEGER DEFAULT 0,
    average_ai_confidence DECIMAL(5, 2) DEFAULT 0.0,
    
    -- User metrics
    unique_users INTEGER DEFAULT 0,
    new_users INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_strategy_user_status ON cross_chain_strategies(user_address, status);
CREATE INDEX IF NOT EXISTS idx_strategy_created_at ON cross_chain_strategies(created_at);
CREATE INDEX IF NOT EXISTS idx_strategy_chains ON cross_chain_strategies(source_chain, target_chain);
CREATE INDEX IF NOT EXISTS idx_strategy_status ON cross_chain_strategies(status);

CREATE INDEX IF NOT EXISTS idx_bridge_tx_hashes ON bridge_transactions(source_tx_hash, target_tx_hash);
CREATE INDEX IF NOT EXISTS idx_bridge_provider_status ON bridge_transactions(bridge_provider, status);
CREATE INDEX IF NOT EXISTS idx_bridge_created_at ON bridge_transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_bridge_strategy_id ON bridge_transactions(strategy_id);

CREATE INDEX IF NOT EXISTS idx_log_strategy_level ON strategy_execution_logs(strategy_id, log_level);
CREATE INDEX IF NOT EXISTS idx_log_created_at ON strategy_execution_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_log_strategy_id ON strategy_execution_logs(strategy_id);

CREATE INDEX IF NOT EXISTS idx_automation_strategy_type ON strategy_automation_rules(strategy_id, rule_type);
CREATE INDEX IF NOT EXISTS idx_automation_status ON strategy_automation_rules(status);
CREATE INDEX IF NOT EXISTS idx_automation_external_id ON strategy_automation_rules(external_rule_id);
CREATE INDEX IF NOT EXISTS idx_automation_strategy_id ON strategy_automation_rules(strategy_id);

CREATE INDEX IF NOT EXISTS idx_stats_date ON orchestrator_statistics(date);
CREATE INDEX IF NOT EXISTS idx_stats_created_at ON orchestrator_statistics(created_at);

-- Create function to automatically update updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_cross_chain_strategies_updated_at 
    BEFORE UPDATE ON cross_chain_strategies 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_bridge_transactions_updated_at 
    BEFORE UPDATE ON bridge_transactions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_strategy_automation_rules_updated_at 
    BEFORE UPDATE ON strategy_automation_rules 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orchestrator_statistics_updated_at 
    BEFORE UPDATE ON orchestrator_statistics 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create enum types for better type safety (PostgreSQL specific)
DO $$ BEGIN
    CREATE TYPE strategy_status_enum AS ENUM (
        'pending', 'analyzing', 'quote_ready', 'executing_bridge',
        'waiting_confirmation', 'bridge_completed', 'executing_target',
        'completed', 'failed', 'cancelled'
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE bridge_provider_enum AS ENUM ('lifi', 'gluex', 'liquid_labs');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE log_level_enum AS ENUM ('info', 'warning', 'error');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Add comments for documentation
COMMENT ON TABLE cross_chain_strategies IS 'Main table for cross-chain strategy execution tracking';
COMMENT ON TABLE bridge_transactions IS 'Individual bridge transactions within strategies';
COMMENT ON TABLE strategy_execution_logs IS 'Detailed execution logs for strategies';
COMMENT ON TABLE strategy_automation_rules IS 'Automation rules linked to strategies';
COMMENT ON TABLE orchestrator_statistics IS 'Daily aggregated statistics for monitoring';

COMMENT ON COLUMN cross_chain_strategies.user_address IS 'Ethereum wallet address of the user';
COMMENT ON COLUMN cross_chain_strategies.ai_generated IS 'Whether this strategy was generated by AI';
COMMENT ON COLUMN cross_chain_strategies.ai_confidence_score IS 'AI confidence score (0-100)';
COMMENT ON COLUMN bridge_transactions.bridge_provider IS 'Provider used for bridge (lifi, gluex, liquid_labs)';
COMMENT ON COLUMN strategy_execution_logs.log_level IS 'Log severity: info, warning, error';

-- Insert initial test data (optional, for development)
-- Uncomment the following lines for test data:

/*
INSERT INTO cross_chain_strategies (
    user_address, source_chain, target_chain, source_token, target_token, 
    amount, status, ai_generated
) VALUES (
    '0x742C4B7c6bB8d7eA6e8D0Ac748fc6B5F10d85B3A',
    'ethereum', 'hyperliquid', 'ETH', 'USDC',
    1.0, 'completed', true
);
*/

-- Create view for strategy summary (useful for reporting)
CREATE OR REPLACE VIEW strategy_summary AS
SELECT 
    s.id,
    s.user_address,
    s.status,
    s.source_chain,
    s.target_chain,
    s.source_token,
    s.target_token,
    s.amount,
    s.total_fees_usd,
    s.ai_generated,
    s.created_at,
    s.updated_at,
    COUNT(bt.id) as bridge_transaction_count,
    COUNT(sar.id) as automation_rule_count,
    MAX(bt.execution_completed_at) as last_bridge_completion
FROM cross_chain_strategies s
LEFT JOIN bridge_transactions bt ON s.id = bt.strategy_id
LEFT JOIN strategy_automation_rules sar ON s.id = sar.strategy_id
GROUP BY s.id;

COMMENT ON VIEW strategy_summary IS 'Summary view of strategies with related counts';

-- Verify tables were created successfully
DO $$
BEGIN
    RAISE NOTICE 'Cross-chain orchestrator tables created successfully!';
    RAISE NOTICE 'Tables: cross_chain_strategies, bridge_transactions, strategy_execution_logs, strategy_automation_rules, orchestrator_statistics';
    RAISE NOTICE 'Indexes and triggers created for optimal performance';
END $$;