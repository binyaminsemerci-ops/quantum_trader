#!/usr/bin/env python3
"""
P1.5 Risk Proposal Publisher (CALC-ONLY, NO TRADING)

Reads:
  - MarketState metrics from Redis (quantum:marketstate:<symbol>)
  - Position snapshots (auto-detect: quantum:state:positions → execution stream → synthetic)

Computes:
  - Risk proposals using P1 Risk Kernel (compute_proposal)

Publishes:
  - Redis hash: quantum:risk:proposal:<symbol>
  - Optional stream: quantum:stream:risk.proposal

NO TradeIntents, NO execution calls, NO orders, NO modify/close.
"""

import os
import sys
import time
import json
import logging
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed. Install with: pip install redis")
    sys.exit(1)

from ai_engine.risk_kernel_stops import compute_proposal, PositionSnapshot


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class PositionSourceAdapter:
    """Auto-detect and fetch position snapshots from various sources"""
    
    def __init__(self, redis_client: redis.Redis, mode: str = "auto"):
        self.redis = redis_client
        self.mode = mode  # auto, redis_hash, redis_stream, synthetic
        logger.info(f"PositionSourceAdapter initialized with mode={mode}")
    
    def get_positions(self, symbols: List[str]) -> Dict[str, Optional[PositionSnapshot]]:
        """Get position snapshots for all symbols"""
        if self.mode == "synthetic":
            return self._get_synthetic_positions(symbols)
        elif self.mode == "redis_hash":
            return self._get_from_redis_hash(symbols)
        elif self.mode == "redis_stream":
            return self._get_from_redis_stream(symbols)
        else:  # auto
            # Try sources in order
            positions = self._get_from_redis_hash(symbols)
            if not positions or all(v is None for v in positions.values()):
                logger.debug("No positions in redis_hash, trying redis_stream")
                positions = self._get_from_redis_stream(symbols)
            if not positions or all(v is None for v in positions.values()):
                logger.debug("No positions in redis_stream, using synthetic")
                positions = self._get_synthetic_positions(symbols)
            return positions
    
    def _get_from_redis_hash(self, symbols: List[str]) -> Dict[str, Optional[PositionSnapshot]]:
        """Try quantum:state:positions (hash or json)"""
        positions = {}
        try:
            # Try as hash with symbol keys
            for symbol in symbols:
                key = f"quantum:state:positions:{symbol}"
                data = self.redis.hgetall(key)
                if data:
                    positions[symbol] = self._parse_position_data(symbol, data)
                else:
                    positions[symbol] = None
            
            # If no individual keys, try single hash
            if all(v is None for v in positions.values()):
                data = self.redis.hgetall("quantum:state:positions")
                if data:
                    for symbol in symbols:
                        if symbol.encode() in data or symbol in data:
                            raw = data.get(symbol.encode()) or data.get(symbol)
                            if isinstance(raw, bytes):
                                raw = raw.decode()
                            pos_data = json.loads(raw) if isinstance(raw, str) else raw
                            positions[symbol] = self._parse_position_data(symbol, pos_data)
        except Exception as e:
            logger.debug(f"Error reading from redis_hash: {e}")
        
        return positions
    
    def _get_from_redis_stream(self, symbols: List[str]) -> Dict[str, Optional[PositionSnapshot]]:
        """Try parsing quantum:stream:execution.result for latest positions"""
        positions = {sym: None for sym in symbols}
        try:
            # Read last N entries from execution stream
            entries = self.redis.xrevrange("quantum:stream:execution.result", count=100)
            for entry_id, fields in entries:
                # fields is dict with bytes keys/values
                event_type = fields.get(b"event_type", b"").decode()
                if event_type == "POSITION_OPENED":
                    symbol = fields.get(b"symbol", b"").decode()
                    if symbol in symbols and positions[symbol] is None:
                        positions[symbol] = self._parse_execution_event(symbol, fields)
        except Exception as e:
            logger.debug(f"Error reading from redis_stream: {e}")
        
        return positions
    
    def _get_synthetic_positions(self, symbols: List[str]) -> Dict[str, Optional[PositionSnapshot]]:
        """Generate synthetic positions for testing"""
        logger.debug("Using synthetic positions")
        positions = {}
        for i, symbol in enumerate(symbols):
            # Create synthetic LONG positions with some variation
            entry = 100.0 + i * 1000
            current = entry * 1.05
            peak = current * 1.02
            positions[symbol] = PositionSnapshot(
                symbol=symbol,
                side="LONG",
                entry_price=entry,
                current_price=current,
                peak_price=peak,
                trough_price=entry * 0.98,
                age_sec=3600.0 + i * 600,
                unrealized_pnl=(current - entry) * 10,
                current_sl=entry * 1.02,
                current_tp=entry * 1.10,
            )
        return positions
    
    def _parse_position_data(self, symbol: str, data: Dict) -> Optional[PositionSnapshot]:
        """Parse position data from Redis hash/json"""
        try:
            # Decode bytes keys/values if needed
            if isinstance(data, dict) and any(isinstance(k, bytes) for k in data.keys()):
                data = {k.decode() if isinstance(k, bytes) else k: 
                       v.decode() if isinstance(v, bytes) else v 
                       for k, v in data.items()}
            
            # Required fields
            return PositionSnapshot(
                symbol=data.get("symbol", symbol),
                side=data.get("side", "LONG"),
                entry_price=float(data.get("entry_price", 0)),
                current_price=float(data.get("current_price", 0)),
                peak_price=float(data.get("peak_price", data.get("current_price", 0))),
                trough_price=float(data.get("trough_price", data.get("current_price", 0))),
                age_sec=float(data.get("age_sec", 0)),
                unrealized_pnl=float(data.get("unrealized_pnl", 0)),
                current_sl=float(data.get("current_sl")) if data.get("current_sl") else None,
                current_tp=float(data.get("current_tp")) if data.get("current_tp") else None,
            )
        except Exception as e:
            logger.warning(f"Failed to parse position for {symbol}: {e}")
            return None
    
    def _parse_execution_event(self, symbol: str, fields: Dict) -> Optional[PositionSnapshot]:
        """Parse position from execution stream event"""
        try:
            return PositionSnapshot(
                symbol=symbol,
                side=fields.get(b"side", b"LONG").decode(),
                entry_price=float(fields.get(b"entry_price", 0)),
                current_price=float(fields.get(b"current_price", fields.get(b"entry_price", 0))),
                peak_price=float(fields.get(b"peak_price", fields.get(b"current_price", 0))),
                trough_price=float(fields.get(b"trough_price", fields.get(b"current_price", 0))),
                age_sec=float(fields.get(b"age_sec", 0)),
                unrealized_pnl=float(fields.get(b"unrealized_pnl", 0)),
                current_sl=float(fields.get(b"current_sl")) if fields.get(b"current_sl") else None,
                current_tp=float(fields.get(b"current_tp")) if fields.get(b"current_tp") else None,
            )
        except Exception as e:
            logger.warning(f"Failed to parse execution event for {symbol}: {e}")
            return None


class RiskProposalPublisher:
    """Main publisher service"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        symbols: List[str] = None,
        publish_interval: int = 10,
        position_source: str = "auto",
        enable_stream: bool = False,
    ):
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.publish_interval = publish_interval
        self.enable_stream = enable_stream
        self.position_adapter = PositionSourceAdapter(self.redis, mode=position_source)
        self.last_proposals = {}  # Cache for rate limiting
        
        logger.info(f"RiskProposalPublisher initialized")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Interval: {self.publish_interval}s")
        logger.info(f"  Position source: {position_source}")
        logger.info(f"  Stream enabled: {self.enable_stream}")
    
    def get_market_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Read MarketState from Redis hash (published by quantum-marketstate)"""
        try:
            key = f"quantum:marketstate:{symbol}"
            data = self.redis.hgetall(key)
            if not data:
                return None
            
            # Decode and parse
            ms = {}
            for k, v in data.items():
                k_str = k.decode() if isinstance(k, bytes) else k
                v_str = v.decode() if isinstance(v, bytes) else v
                if k_str in ["sigma", "mu", "ts", "p_trend", "p_mr", "p_chop", "dp", "vr", "spike_proxy"]:
                    ms[k_str if k_str != "p_trend" else "dummy"] = float(v_str)
            
            # Reconstruct format expected by compute_proposal
            return {
                "sigma": ms.get("sigma", 0.0),
                "mu": ms.get("mu", 0.0),
                "ts": ms.get("ts", 0.0),
                "regime_probs": {
                    "trend": ms.get("p_trend", 0.0),
                    "mr": ms.get("p_mr", 0.0),
                    "chop": ms.get("p_chop", 0.0),
                },
                "features": {
                    "dp": ms.get("dp", 0.0),
                    "vr": ms.get("vr", 0.0),
                    "spike_proxy": ms.get("spike_proxy", 0.0),
                }
            }
        except Exception as e:
            logger.warning(f"Failed to read MarketState for {symbol}: {e}")
            return None
    
    def publish_proposal(self, symbol: str, proposal: Dict[str, Any], market_state: Dict[str, Any], position: PositionSnapshot):
        """Publish proposal to Redis hash and optional stream"""
        try:
            # Build flat hash fields
            fields = {
                "proposed_sl": str(proposal["proposed_sl"]),
                "proposed_tp": str(proposal["proposed_tp"]),
                "stop_dist_pct": str(proposal["audit"]["intermediates"]["stop_dist_pct"]),
                "tp_dist_pct": str(proposal["audit"]["intermediates"]["tp_dist_pct"]),
                "trail_gap_pct": str(proposal["audit"]["intermediates"]["trail_gap_pct"]),
                "reason_codes": ",".join(proposal["reason_codes"]),
                "ts": str(market_state["ts"]),
                "sigma": str(market_state["sigma"]),
                "p_trend": str(market_state["regime_probs"]["trend"]),
                "p_mr": str(market_state["regime_probs"]["mr"]),
                "p_chop": str(market_state["regime_probs"]["chop"]),
                "computed_at_utc": datetime.utcnow().isoformat(),
                "position_side": position.side,
                "position_age_sec": str(position.age_sec),
                "position_current_price": str(position.current_price),
                "position_entry_price": str(position.entry_price),
            }
            
            # Publish to hash
            hash_key = f"quantum:risk:proposal:{symbol}"
            self.redis.hset(hash_key, mapping=fields)
            
            # Publish to stream if enabled
            if self.enable_stream:
                stream_key = "quantum:stream:risk.proposal"
                fields_with_symbol = {"symbol": symbol, **fields}
                self.redis.xadd(stream_key, fields_with_symbol, maxlen=10000)
            
            logger.info(
                f"Published proposal for {symbol}: "
                f"SL=${proposal['proposed_sl']:.4f} TP=${proposal['proposed_tp']:.4f} "
                f"reasons={','.join(proposal['reason_codes'][:2])}"
            )
            
        except Exception as e:
            logger.error(f"Failed to publish proposal for {symbol}: {e}")
    
    def should_publish(self, symbol: str, proposal: Dict[str, Any]) -> bool:
        """Rate limiting: publish only if changed meaningfully"""
        if symbol not in self.last_proposals:
            return True
        
        last = self.last_proposals[symbol]
        current_sl = proposal["proposed_sl"]
        current_tp = proposal["proposed_tp"]
        last_sl = last.get("proposed_sl", 0)
        last_tp = last.get("proposed_tp", 0)
        
        # Publish if changed by >0.1% or >60s since last
        sl_changed = abs(current_sl - last_sl) / max(last_sl, 1.0) > 0.001
        tp_changed = abs(current_tp - last_tp) / max(last_tp, 1.0) > 0.001
        time_elapsed = time.time() - last.get("timestamp", 0) > 60
        
        return sl_changed or tp_changed or time_elapsed
    
    def run_cycle(self):
        """Single publish cycle"""
        logger.info("=== Risk Proposal Publish Cycle ===")
        
        # Get positions
        positions = self.position_adapter.get_positions(self.symbols)
        
        for symbol in self.symbols:
            try:
                # Get market state
                market_state = self.get_market_state(symbol)
                if not market_state:
                    logger.warning(f"{symbol}: No MarketState data, skipping")
                    continue
                
                # Get position
                position = positions.get(symbol)
                if not position:
                    logger.debug(f"{symbol}: No position found, skipping")
                    continue
                
                # Compute proposal
                proposal = compute_proposal(symbol, market_state, position)
                
                # Rate limiting check
                if not self.should_publish(symbol, proposal):
                    logger.debug(f"{symbol}: No significant change, skipping publish")
                    continue
                
                # Publish
                self.publish_proposal(symbol, proposal, market_state, position)
                
                # Update cache
                self.last_proposals[symbol] = {
                    "proposed_sl": proposal["proposed_sl"],
                    "proposed_tp": proposal["proposed_tp"],
                    "timestamp": time.time(),
                }
                
            except Exception as e:
                logger.error(f"{symbol}: Error in cycle: {e}", exc_info=True)
    
    def run_loop(self):
        """Main publish loop"""
        logger.info("Starting publish loop")
        while True:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down (KeyboardInterrupt)")
                break
            except Exception as e:
                logger.error(f"Error in publish loop: {e}", exc_info=True)
            
            time.sleep(self.publish_interval)


def main():
    parser = argparse.ArgumentParser(description="P1.5 Risk Proposal Publisher")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--symbols", default=os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"))
    parser.add_argument("--interval", type=int, default=int(os.getenv("PUBLISH_INTERVAL_SEC", "10")))
    parser.add_argument("--position-source", default=os.getenv("POSITION_SOURCE", "auto"),
                       choices=["auto", "redis_hash", "redis_stream", "synthetic"])
    parser.add_argument("--enable-stream", action="store_true", 
                       default=os.getenv("ENABLE_STREAM", "false").lower() == "true")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    publisher = RiskProposalPublisher(
        redis_url=args.redis_url,
        symbols=symbols,
        publish_interval=args.interval,
        position_source=args.position_source,
        enable_stream=args.enable_stream,
    )
    
    if args.once:
        logger.info("Running single cycle (--once mode)")
        publisher.run_cycle()
        logger.info("Single cycle complete")
    else:
        publisher.run_loop()


if __name__ == "__main__":
    main()
