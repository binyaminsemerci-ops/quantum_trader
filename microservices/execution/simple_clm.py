"""
Simple Continuous Learning Manager (SimpleCLM) - Trade Outcome Recorder

PURPOSE: Passive collection and persistence of closed-trade outcomes.
NOT responsible for: training, decision-making, or orchestration.

This is a truth recorder, not a learning engine.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class OutcomeLabel(str, Enum):
    """Trade outcome classification"""
    WIN = "WIN"
    LOSS = "LOSS"
    NEUTRAL = "NEUTRAL"


@dataclass
class TradeOutcome:
    """Complete trade outcome record"""
    # Required fields (validation enforced)
    timestamp: str          # UTC ISO-8601
    symbol: str
    side: str              # BUY/SELL
    entry_price: float
    exit_price: float
    pnl_percent: float
    confidence: float
    model_id: str
    
    # Derived fields (computed by SimpleCLM)
    outcome_label: str     # WIN/LOSS/NEUTRAL
    duration_seconds: Optional[float] = None
    
    # Optional context
    strategy_id: Optional[str] = None
    position_size: Optional[float] = None
    exit_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization"""
        return asdict(self)


class SimpleCLM:
    """
    Lightweight trade outcome collector.
    
    Responsibilities:
    - ✅ Receive closed trade events
    - ✅ Validate required fields
    - ✅ Label outcomes (WIN/LOSS/NEUTRAL)
    - ✅ Persist to append-only JSONL
    - ✅ Expose observability metrics
    - ✅ Detect starvation (no trades for N hours)
    
    NOT responsible for:
    - ❌ Training models
    - ❌ Making decisions
    - ❌ Scheduling retraining
    - ❌ Orchestrating learning
    """
    
    # Required input fields (validation enforced)
    REQUIRED_FIELDS = [
        "timestamp", "symbol", "side", "entry_price", "exit_price",
        "pnl_percent", "confidence", "model_id"
    ]
    
    def __init__(
        self,
        storage_path: str = "/home/qt/quantum_trader/data/clm_trades.jsonl",
        win_threshold: float = 0.5,      # PnL% > 0.5 = WIN
        loss_threshold: float = -0.5,    # PnL% < -0.5 = LOSS
        starvation_hours: float = 1.0,   # Alert if no trades for 1 hour
        stats_log_interval_seconds: int = 300,  # Log stats every 5 minutes
    ):
        self.storage_path = Path(storage_path)
        self.win_threshold = win_threshold
        self.loss_threshold = loss_threshold
        self.starvation_hours = starvation_hours
        self.stats_log_interval_seconds = stats_log_interval_seconds
        
        # Observability counters
        self.total_received = 0
        self.total_rejected = 0
        self.total_stored = 0
        self.last_trade_timestamp: Optional[datetime] = None
        
        # Runtime state
        self.running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._persistence_enabled = True
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data on startup (crash recovery)
        self._load_existing_data()
        
        logger.info(
            f"[sCLM] Initialized: storage={self.storage_path}, "
            f"win_threshold={self.win_threshold}%, loss_threshold={self.loss_threshold}%, "
            f"starvation_alert={self.starvation_hours}h"
        )
        logger.info(
            f"[sCLM] Recovered state: {self.total_stored} trades on disk, "
            f"last_trade={self.last_trade_timestamp}"
        )
    
    def _load_existing_data(self):
        """Load existing trade data from disk (crash recovery)"""
        if not self.storage_path.exists():
            logger.info(f"[sCLM] No existing data file: {self.storage_path}")
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                lines = f.readlines()
                self.total_stored = len(lines)
                
                # Get last trade timestamp
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        last_trade = json.loads(last_line)
                        self.last_trade_timestamp = datetime.fromisoformat(last_trade['timestamp'])
            
            logger.info(f"[sCLM] ✅ Loaded {self.total_stored} existing trades")
        
        except Exception as e:
            logger.error(f"[sCLM] ❌ Failed to load existing data: {e}", exc_info=True)
    
    async def start(self):
        """Start background monitoring tasks"""
        if self.running:
            logger.warning("[sCLM] Already running")
            return
        
        self.running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("[sCLM] ✅ Started")
    
    async def stop(self):
        """Stop background tasks"""
        self.running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("[sCLM] ✅ Stopped")
    
    def validate_trade(self, trade_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate trade data against required fields.
        
        Returns:
            (valid, error_reason)
        """
        # Check all required fields present
        missing_fields = [f for f in self.REQUIRED_FIELDS if f not in trade_data]
        if missing_fields:
            return False, f"Missing required fields: {missing_fields}"
        
        # Validate field types and values
        try:
            # Numeric validations
            if not isinstance(trade_data['entry_price'], (int, float)) or trade_data['entry_price'] <= 0:
                return False, f"Invalid entry_price: {trade_data['entry_price']}"
            
            if not isinstance(trade_data['exit_price'], (int, float)) or trade_data['exit_price'] <= 0:
                return False, f"Invalid exit_price: {trade_data['exit_price']}"
            
            if not isinstance(trade_data['pnl_percent'], (int, float)):
                return False, f"Invalid pnl_percent: {trade_data['pnl_percent']}"
            
            if not isinstance(trade_data['confidence'], (int, float)) or not (0 <= trade_data['confidence'] <= 1):
                return False, f"Invalid confidence: {trade_data['confidence']} (must be 0-1)"
            
            # String validations
            if not isinstance(trade_data['symbol'], str) or not trade_data['symbol']:
                return False, f"Invalid symbol: {trade_data['symbol']}"
            
            if trade_data['side'].upper() not in ['BUY', 'SELL', 'LONG', 'SHORT']:
                return False, f"Invalid side: {trade_data['side']} (must be BUY/SELL)"
            
            # Timestamp validation
            try:
                datetime.fromisoformat(trade_data['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return False, f"Invalid timestamp format: {trade_data['timestamp']}"
            
            return True, None
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def label_outcome(self, pnl_percent: float) -> OutcomeLabel:
        """
        Classify trade outcome based on PnL%.
        
        Rule (deterministic):
        - PnL > win_threshold → WIN
        - PnL < loss_threshold → LOSS
        - Otherwise → NEUTRAL
        """
        if pnl_percent > self.win_threshold:
            return OutcomeLabel.WIN
        elif pnl_percent < self.loss_threshold:
            return OutcomeLabel.LOSS
        else:
            return OutcomeLabel.NEUTRAL
    
    def record_trade(self, trade_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Record a closed trade.
        
        Steps:
        1. Validate input
        2. Label outcome
        3. Persist to disk
        4. Update counters
        
        Returns:
            (success, error_message)
        """
        self.total_received += 1
        
        # Step 1: Validate
        valid, error = self.validate_trade(trade_data)
        if not valid:
            self.total_rejected += 1
            logger.warning(f"[sCLM] ❌ Trade rejected: {error}")
            return False, error
        
        # Step 2: Label outcome
        try:
            outcome_label = self.label_outcome(trade_data['pnl_percent'])
            
            # Create TradeOutcome object
            trade_outcome = TradeOutcome(
                timestamp=trade_data['timestamp'],
                symbol=trade_data['symbol'],
                side=trade_data['side'].upper(),
                entry_price=float(trade_data['entry_price']),
                exit_price=float(trade_data['exit_price']),
                pnl_percent=float(trade_data['pnl_percent']),
                confidence=float(trade_data['confidence']),
                model_id=trade_data['model_id'],
                outcome_label=outcome_label.value,
                duration_seconds=trade_data.get('duration_seconds'),
                strategy_id=trade_data.get('strategy_id'),
                position_size=trade_data.get('position_size'),
                exit_reason=trade_data.get('exit_reason')
            )
            
            # Step 3: Persist
            if self._persistence_enabled:
                success = self._persist_trade(trade_outcome)
                if not success:
                    logger.critical("[sCLM] 🔥 PERSISTENCE FAILED - STOPPING NEW TRADES")
                    self._persistence_enabled = False
                    return False, "Persistence failure"
            
            # Step 4: Update counters
            self.total_stored += 1
            self.last_trade_timestamp = datetime.now(timezone.utc)
            
            logger.info(
                f"[sCLM] ✅ Trade recorded: {trade_outcome.symbol} {trade_outcome.side} "
                f"PnL={trade_outcome.pnl_percent:+.2f}% → {trade_outcome.outcome_label} "
                f"(total: {self.total_stored})"
            )
            
            return True, None
        
        except Exception as e:
            self.total_rejected += 1
            error_msg = f"Processing failed: {str(e)}"
            logger.error(f"[sCLM] ❌ {error_msg}", exc_info=True)
            return False, error_msg
    
    def _persist_trade(self, trade: TradeOutcome) -> bool:
        """
        Persist trade to JSONL file (atomic write).
        
        Returns:
            success (bool)
        """
        try:
            # Create parent directory if needed
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Direct append (simpler for Windows compatibility)
            with open(self.storage_path, 'a') as f:
                json.dump(trade.to_dict(), f)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            return True
        
        except Exception as e:
            logger.critical(f"[sCLM] 🔥 PERSIST FAILED: {e}", exc_info=True)
            return False
    
    async def _monitoring_loop(self):
        """Background task: periodic stats logging + starvation detection"""
        try:
            while self.running:
                await asyncio.sleep(self.stats_log_interval_seconds)
                
                # Log stats
                self._log_stats()
                
                # Check starvation
                self._check_starvation()
        
        except asyncio.CancelledError:
            logger.info("[sCLM] Monitoring loop cancelled")
    
    def _log_stats(self):
        """Log current statistics"""
        file_size_mb = self.storage_path.stat().st_size / 1024 / 1024 if self.storage_path.exists() else 0
        
        logger.info(
            f"[sCLM] 📊 Stats: received={self.total_received}, "
            f"stored={self.total_stored}, rejected={self.total_rejected}, "
            f"file_size={file_size_mb:.2f}MB, "
            f"last_trade={self.last_trade_timestamp}"
        )
    
    def _check_starvation(self):
        """Detect and alert on starvation (no trades for N hours)"""
        if not self.last_trade_timestamp:
            logger.warning("[sCLM] ⚠️ No trades recorded yet (waiting for first trade)")
            return
        
        idle_seconds = (datetime.now(timezone.utc) - self.last_trade_timestamp).total_seconds()
        idle_hours = idle_seconds / 3600
        
        if idle_hours > self.starvation_hours:
            logger.critical(
                f"[sCLM] 🚨 STARVATION DETECTED: No trades for {idle_hours:.1f}h "
                f"(last trade: {self.last_trade_timestamp.isoformat()})"
            )
    
    def get_status(self) -> dict:
        """Get observable health status"""
        file_size_mb = self.storage_path.stat().st_size / 1024 / 1024 if self.storage_path.exists() else 0
        
        idle_hours = None
        if self.last_trade_timestamp:
            idle_seconds = (datetime.now(timezone.utc) - self.last_trade_timestamp).total_seconds()
            idle_hours = idle_seconds / 3600
        
        return {
            "running": self.running,
            "persistence_enabled": self._persistence_enabled,
            "storage_path": str(self.storage_path),
            "file_size_mb": round(file_size_mb, 2),
            "total_trades_received": self.total_received,
            "total_trades_stored": self.total_stored,
            "total_trades_rejected": self.total_rejected,
            "last_trade_timestamp": self.last_trade_timestamp.isoformat() if self.last_trade_timestamp else None,
            "idle_hours": round(idle_hours, 2) if idle_hours else None,
            "starving": idle_hours > self.starvation_hours if idle_hours else False,
            "thresholds": {
                "win_threshold": self.win_threshold,
                "loss_threshold": self.loss_threshold,
                "starvation_hours": self.starvation_hours
            }
        }
