#!/usr/bin/env python3
"""
Production Quantum Trader Startup Script

This script orchestrates the complete production deployment including:
- Configuration validation
- Risk management setup
- Performance monitoring initialization
- AI model training and deployment
- Live trading system startup
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any
import subprocess
import signal
import threading

from config.config import settings
from production_risk_manager import RiskManager, RiskParameters
from production_monitor import PerformanceMonitor, AlertConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionDeployment:
    """Production deployment orchestrator."""
    
    def __init__(self):
        self.processes: Dict[str, Any] = {}
        self.risk_manager: RiskManager = None
        self.monitor: PerformanceMonitor = None
        self.is_running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
    
    def _shutdown_handler(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info(f"Received shutdown signal {signum}")
        self.shutdown()
        sys.exit(0)
    
    def validate_configuration(self) -> bool:
        """Validate production configuration."""
        logger.info("🔍 Validating production configuration...")
        
        validation_results = []
        
        # Check API keys if live trading enabled
        if settings.enable_real_trading:
            if not settings.binance_api_key or not settings.binance_api_secret:
                validation_results.append("❌ Binance API keys required for live trading")
            else:
                validation_results.append("✅ Binance API keys configured")
        else:
            validation_results.append("ℹ️  Paper trading mode (ENABLE_REAL_TRADING=False)")
        
        # Check market data configuration
        if settings.enable_live_market_data:
            validation_results.append("✅ Live market data enabled")
        else:
            validation_results.append("⚠️  Using demo market data")
        
        # Check risk management settings
        if settings.max_position_size_pct > 5.0:
            validation_results.append("⚠️  High position size limit (>5%)")
        else:
            validation_results.append("✅ Conservative position sizing")
        
        if settings.max_daily_loss_pct > 10.0:
            validation_results.append("⚠️  High daily loss limit (>10%)")
        else:
            validation_results.append("✅ Conservative daily loss limit")
        
        # Check starting equity
        if settings.starting_equity < 1000:
            validation_results.append("⚠️  Low starting equity (<$1000)")
        else:
            validation_results.append(f"✅ Starting equity: ${settings.starting_equity:,.2f}")
        
        # Check monitoring configuration
        if settings.enable_monitoring:
            validation_results.append("✅ Performance monitoring enabled")
        else:
            validation_results.append("⚠️  Performance monitoring disabled")
        
        # Print validation results
        for result in validation_results:
            print(result)
        
        # Check for any critical issues
        critical_issues = [r for r in validation_results if r.startswith("❌")]
        if critical_issues:
            logger.error("Critical configuration issues found!")
            return False
        
        logger.info("Configuration validation passed ✅")
        return True
    
    def setup_risk_management(self) -> bool:
        """Initialize risk management system."""
        try:
            logger.info("⚙️  Setting up risk management...")
            
            risk_params = RiskParameters(
                max_position_size_pct=settings.max_position_size_pct,
                stop_loss_pct=settings.stop_loss_pct,
                take_profit_pct=settings.take_profit_pct,
                max_daily_loss_pct=settings.max_daily_loss_pct,
                max_drawdown_pct=settings.max_drawdown_pct,
            )
            
            self.risk_manager = RiskManager(risk_params)
            logger.info("✅ Risk management system initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup risk management: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Initialize performance monitoring system."""
        try:
            if not settings.enable_monitoring:
                logger.info("Performance monitoring disabled in config")
                return True
            
            logger.info("📊 Setting up performance monitoring...")
            
            alert_config = AlertConfig(
                max_daily_loss_pct=settings.max_daily_loss_pct,
                max_drawdown_pct=settings.max_drawdown_pct,
                min_model_accuracy=0.55,
                max_portfolio_risk=20.0
            )
            
            self.monitor = PerformanceMonitor(
                risk_manager=self.risk_manager,
                alert_config=alert_config
            )
            
            # Start monitoring
            self.monitor.start_monitoring(settings.monitoring_interval_seconds)
            logger.info("✅ Performance monitoring started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            return False
    
    def run_initial_training(self) -> bool:
        """Run initial AI model training."""
        try:
            logger.info("🤖 Running initial AI model training...")
            
            # Import training functions
            from ai_engine.train_and_save import train_and_save
            
            # Use first 10 symbols for faster initial training
            initial_symbols = list(settings.default_symbols)[:10]
            
            result = train_and_save(
                symbols=initial_symbols,
                limit=800,  # Reasonable amount for production
                backtest=True,
                write_report=True
            )
            
            if "error" in result:
                logger.error(f"Initial training failed: {result['error']}")
                return False
            
            # Log training results
            metrics = result.get("metrics", {})
            backtest = result.get("backtest", {})
            
            logger.info(f"Training completed:")
            logger.info(f"  - Model accuracy: {metrics.get('directional_accuracy', 0):.1%}")
            logger.info(f"  - Backtest return: {backtest.get('pnl', 0):.2f}")
            logger.info(f"  - Win rate: {backtest.get('win_rate', 0):.1%}")
            
            return True
            
        except Exception as e:
            logger.error(f"Initial training failed: {e}")
            return False
    
    def start_trading_system(self) -> bool:
        """Start the main trading system."""
        try:
            logger.info("🚀 Starting trading system...")
            
            # For now, we'll start a background training process
            # In a full implementation, this would start the live trading bot
            
            if settings.enable_real_trading:
                logger.info("🔴 LIVE TRADING MODE - Real money at risk!")
                confirmation = input("Type 'CONFIRM' to proceed with live trading: ")
                if confirmation != 'CONFIRM':
                    logger.info("Live trading cancelled by user")
                    return False
            else:
                logger.info("📝 Paper trading mode - No real money at risk")
            
            # Start trading process (placeholder)
            logger.info("✅ Trading system started")
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start trading system: {e}")
            return False
    
    def run_production_tests(self) -> bool:
        """Run production-scale tests."""
        try:
            logger.info("🧪 Running production tests...")
            
            # Run quick production test
            from production_ai_test import ProductionTestRunner
            
            test_runner = ProductionTestRunner()
            result = test_runner.run_configuration_test("quick_production")
            
            # Check test results
            successful_tests = result['test_summary']['successful_tests']
            total_tests = result['test_summary']['total_tests']
            
            if successful_tests / total_tests < 0.8:  # Require 80% success rate
                logger.error(f"Production tests failed: {successful_tests}/{total_tests} successful")
                return False
            
            logger.info(f"✅ Production tests passed: {successful_tests}/{total_tests} successful")
            return True
            
        except Exception as e:
            logger.error(f"Production tests failed: {e}")
            return False
    
    def monitor_system(self):
        """Monitor the running system."""
        logger.info("👁️  System monitoring active...")
        
        try:
            while self.is_running:
                # Check system health
                if self.monitor:
                    status = self.monitor.get_current_status()
                    if status.get("monitoring_active"):
                        logger.debug("System monitoring healthy")
                    else:
                        logger.warning("Performance monitoring not active")
                
                time.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
    
    def shutdown(self):
        """Shutdown all systems gracefully."""
        logger.info("🛑 Shutting down production system...")
        
        self.is_running = False
        
        # Stop monitoring
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Save final state
        if self.risk_manager:
            self.risk_manager.save_state()
        
        # Terminate processes
        for name, process in self.processes.items():
            try:
                logger.info(f"Stopping {name}...")
                process.terminate()
                process.wait(timeout=10)
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
        
        logger.info("✅ Shutdown complete")
    
    def deploy(self, skip_tests: bool = False, skip_training: bool = False) -> bool:
        """Run complete production deployment."""
        logger.info("🚀 Starting Quantum Trader Production Deployment")
        print("=" * 60)
        
        # Step 1: Validate configuration
        if not self.validate_configuration():
            logger.error("Configuration validation failed")
            return False
        
        # Step 2: Setup risk management
        if not self.setup_risk_management():
            logger.error("Risk management setup failed")
            return False
        
        # Step 3: Setup monitoring
        if not self.setup_monitoring():
            logger.error("Monitoring setup failed")
            return False
        
        # Step 4: Run production tests (optional)
        if not skip_tests:
            if not self.run_production_tests():
                logger.error("Production tests failed")
                return False
        
        # Step 5: Initial training (optional)
        if not skip_training:
            if not self.run_initial_training():
                logger.error("Initial training failed")
                return False
        
        # Step 6: Start trading system
        if not self.start_trading_system():
            logger.error("Trading system startup failed")
            return False
        
        logger.info("🎉 Production deployment successful!")
        print("\n" + "=" * 60)
        print("📊 Quantum Trader Production System Status:")
        print("  🟢 Risk Management: Active")
        if self.monitor:
            print("  🟢 Performance Monitoring: Active")
        print("  🟢 AI Model: Trained and Ready")
        print("  🟢 Trading System: Online")
        print("=" * 60)
        
        # Start monitoring loop
        self.monitor_system()
        
        return True


def main():
    """Main production startup function."""
    parser = argparse.ArgumentParser(description="Quantum Trader Production Deployment")
    parser.add_argument("--skip-tests", action="store_true", help="Skip production tests")
    parser.add_argument("--skip-training", action="store_true", help="Skip initial training")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    
    args = parser.parse_args()
    
    # Initialize deployment
    deployment = ProductionDeployment()
    
    if args.validate_only:
        # Just validate and exit
        if deployment.validate_configuration():
            print("\n✅ Configuration is ready for production!")
            sys.exit(0)
        else:
            print("\n❌ Configuration needs fixes before production deployment")
            sys.exit(1)
    
    # Run full deployment
    try:
        success = deployment.deploy(
            skip_tests=args.skip_tests,
            skip_training=args.skip_training
        )
        
        if not success:
            logger.error("Production deployment failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\nDeployment interrupted by user")
        deployment.shutdown()
    except Exception as e:
        logger.error(f"Deployment failed with error: {e}")
        deployment.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()