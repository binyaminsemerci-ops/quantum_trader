#!/usr/bin/env python3
"""
P2.5 Harvest Proposal Publisher (CALC-ONLY, NO TRADING)

Reads:
  - MarketState metrics from Redis (quantum:marketstate:<symbol>) - P0.5 output
  - Risk proposals from Redis (quantum:risk:proposal:<symbol>) - P1.5 output (for stop_dist_pct)
  - Position snapshots (auto-detect: quantum:state:positions → execution stream → synthetic)

Computes:
  - Harvest proposals using P2 Harvest Kernel (compute_harvest_proposal)

Publishes:
  - Redis hash: quantum:harvest:proposal:<symbol>
  - Optional stream: quantum:stream:harvest.proposal

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

# Add parent directory to path (quantum_trader root, not microservices)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed. Install with: pip install redis")
    sys.exit(1)

from ai_engine.risk_kernel_harvest import (
    compute_harvest_proposal,
    HarvestTheta,
    PositionSnapshot,
    MarketState,
    P1Proposal,
)


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
        """Try quantum:stream:execution (last entry per symbol)"""
        positions = {}
        try:
            stream_key = "quantum:stream:execution"
            # Read last 100 entries
            entries = self.redis.xrevrange(stream_key, count=100)
            
            # Find most recent entry for each symbol
            for symbol in symbols:
                for entry_id, fields in entries:
                    sym = fields.get(b"symbol", b"").decode()
                    if sym == symbol:
                        positions[symbol] = self._parse_execution_event(symbol, fields)
                        break
                else:
                    positions[symbol] = None
        except Exception as e:
            logger.debug(f"Error reading from redis_stream: {e}")
        
        return positions
    
    def _get_synthetic_positions(self, symbols: List[str]) -> Dict[str, Optional[PositionSnapshot]]:
        """Generate synthetic positions for testing"""
        positions = {}
        for symbol in symbols:
            base_price = 100.0 if "BTC" in symbol else 50.0
            positions[symbol] = PositionSnapshot(
                symbol=symbol,
                side="LONG",
                entry_price=base_price,
                current_price=base_price * 1.05,  # 5% profit
                peak_price=base_price * 1.06,
                trough_price=base_price * 0.98,
                age_sec=3600.0,  # 1 hour
                unrealized_pnl=base_price * 0.05,
                current_sl=base_price * 0.98,
                current_tp=base_price * 1.10,
            )
        logger.debug(f"Generated synthetic positions for {len(symbols)} symbols")
        return positions
    
    def _parse_position_data(self, symbol: str, data: Dict) -> Optional[PositionSnapshot]:
        """Parse position from Redis hash data"""
        try:
            # Decode bytes keys/values
            if data and isinstance(list(data.keys())[0], bytes):
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


class HarvestProposalPublisher:
    """Main publisher service"""
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        symbols: List[str] = None,
        publish_interval: int = 10,
        position_source: str = "auto",
        enable_stream: bool = False,
        fallback_stop_pct: float = 0.02,
        max_publish_age: int = 60,
        change_eps_pct: float = 0.001,
    ):
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self.publish_interval = publish_interval
        self.enable_stream = enable_stream
        self.position_adapter = PositionSourceAdapter(self.redis, mode=position_source)
        self.fallback_stop_pct = fallback_stop_pct
        self.max_publish_age = max_publish_age
        self.change_eps_pct = change_eps_pct
        self.last_proposals = {}  # Cache for rate limiting
        self.theta = HarvestTheta()  # Use default tunables
        
        logger.info(f"HarvestProposalPublisher initialized")
        logger.info(f"  Symbols: {self.symbols}")
        logger.info(f"  Interval: {self.publish_interval}s")
        logger.info(f"  Position source: {position_source}")
        logger.info(f"  Stream enabled: {self.enable_stream}")
        logger.info(f"  Fallback stop%: {self.fallback_stop_pct}")
        logger.info(f"  Max publish age: {self.max_publish_age}s")
        logger.info(f"  Change epsilon: {self.change_eps_pct}")
    
    def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Read MarketState from Redis hash (published by quantum-marketstate)"""
        try:
            key = f"quantum:marketstate:{symbol}"
            data = self.redis.hgetall(key)
            if not data:
                return None
            
            # Decode and parse
            def get_float(k: str) -> float:
                v = data.get(k.encode()) or data.get(k)
                if v is None:
                    return 0.0
                return float(v.decode() if isinstance(v, bytes) else v)
            
            return MarketState(
                sigma=get_float("sigma"),
                ts=get_float("ts"),
                p_trend=get_float("p_trend"),
                p_mr=get_float("p_mr"),
                p_chop=get_float("p_chop"),
            )
        except Exception as e:
            logger.warning(f"Failed to read MarketState for {symbol}: {e}")
            return None
    
    def get_risk_proposal(self, symbol: str) -> Optional[P1Proposal]:
        """Read P1 risk proposal from Redis hash (published by P1.5)"""
        try:
            key = f"quantum:risk:proposal:{symbol}"
            data = self.redis.hgetall(key)
            if not data:
                return None
            
            # Decode and parse
            def get_float(k: str) -> Optional[float]:
                v = data.get(k.encode()) or data.get(k)
                if v is None:
                    return None
                return float(v.decode() if isinstance(v, bytes) else v)
            
            stop_dist_pct = get_float("stop_dist_pct")
            proposed_sl = get_float("proposed_sl")
            
            if stop_dist_pct is None:
                return None
            
            return P1Proposal(
                stop_dist_pct=stop_dist_pct,
                proposed_sl=proposed_sl,
            )
        except Exception as e:
            logger.debug(f"Failed to read risk proposal for {symbol}: {e}")
            return None
    
    def _extract_k_components(self, harvest_output: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract kill score components from harvest_output audit field.
        Returns dict with regime_flip, sigma_spike, ts_drop, age_penalty.
        Defaults to 0.0 if missing or invalid (P2.6B robustness).
        """
        default_components = {
            "regime_flip": 0.0,
            "sigma_spike": 0.0,
            "ts_drop": 0.0,
            "age_penalty": 0.0,
        }
        
        try:
            audit = harvest_output.get("audit")
            if not audit:
                return default_components
            
            k_components = audit.get("k_components")
            if not k_components:
                return default_components
            
            # Extract and validate each component
            result = {}
            for key in ["regime_flip", "sigma_spike", "ts_drop", "age_penalty"]:
                try:
                    value = k_components.get(key, 0.0)
                    result[key] = float(value)
                except (ValueError, TypeError):
                    result[key] = 0.0
            
            return result
        except Exception as e:
            logger.debug(f"Failed to extract k_components: {e}")
            return default_components
    
    def publish_proposal(
        self, 
        symbol: str, 
        harvest_output: Dict[str, Any], 
        market_state: MarketState,
        position: PositionSnapshot
    ):
        """Publish harvest proposal to Redis hash and optional stream"""
        try:
            # Extract k_components safely (P2.6B observability)
            k_components = self._extract_k_components(harvest_output)
            
            # Build flat hash fields
            fields = {
                "harvest_action": harvest_output["harvest_action"],
                "new_sl_proposed": str(harvest_output.get("new_sl_proposed", "")),
                "R_net": str(harvest_output["R_net"]),
                "risk_unit": str(harvest_output["risk_unit"]),
                "cost_est": str(harvest_output["cost_est"]),
                "kill_score": str(harvest_output["kill_score"]),
                "reason_codes": ",".join(harvest_output["reason_codes"]),
                # P2.6B: Kill score component breakdown
                "k_regime_flip": str(k_components["regime_flip"]),
                "k_sigma_spike": str(k_components["sigma_spike"]),
                "k_ts_drop": str(k_components["ts_drop"]),
                "k_age_penalty": str(k_components["age_penalty"]),
                "sigma": str(market_state.sigma),
                "ts": str(market_state.ts),
                "p_trend": str(market_state.p_trend),
                "p_mr": str(market_state.p_mr),
                "p_chop": str(market_state.p_chop),
                "computed_at_utc": datetime.utcnow().isoformat(),
                # P2.7C.1: Timestamp source-of-truth for staleness detection
                "last_update_epoch": str(int(time.time())),
                "position_side": position.side,
                "position_age_sec": str(position.age_sec),
                "position_current_price": str(position.current_price),
                "position_entry_price": str(position.entry_price),
                "position_unrealized_pnl": str(position.unrealized_pnl),
            }
            
            # Publish to hash
            hash_key = f"quantum:harvest:proposal:{symbol}"
            self.redis.hset(hash_key, mapping=fields)
            
            # Publish to stream if enabled
            if self.enable_stream:
                stream_key = "quantum:stream:harvest.proposal"
                fields_with_symbol = {"symbol": symbol, **fields}
                self.redis.xadd(stream_key, fields_with_symbol, maxlen=10000)
            
            action = harvest_output["harvest_action"]
            new_sl = harvest_output.get("new_sl_proposed")
            k = harvest_output["kill_score"]
            r = harvest_output["R_net"]
            
            logger.info(
                f"Published harvest for {symbol}: "
                f"action={action} R={r:.2f} K={k:.3f} "
                f"SL={new_sl if new_sl else 'N/A'} "
                f"reasons={','.join(harvest_output['reason_codes'][:2])}"
            )
            
        except Exception as e:
            logger.error(f"Failed to publish harvest proposal for {symbol}: {e}")
    
    def should_publish(self, symbol: str, harvest_output: Dict[str, Any]) -> bool:
        """Rate limiting: publish only if changed meaningfully or max age elapsed"""
        if symbol not in self.last_proposals:
            return True
        
        last = self.last_proposals[symbol]
        current = {
            "harvest_action": harvest_output["harvest_action"],
            "new_sl_proposed": harvest_output.get("new_sl_proposed"),
            "kill_score": harvest_output["kill_score"],
            "timestamp": time.time(),
        }
        
        # Check if harvest action changed
        if current["harvest_action"] != last.get("harvest_action"):
            return True
        
        # Check if new_sl_proposed changed materially
        if current["new_sl_proposed"] and last.get("new_sl_proposed"):
            sl_change_pct = abs(current["new_sl_proposed"] - last["new_sl_proposed"]) / last["new_sl_proposed"]
            if sl_change_pct > self.change_eps_pct:
                return True
        elif current["new_sl_proposed"] != last.get("new_sl_proposed"):
            # One is None, other isn't
            return True
        
        # Check if kill_score changed materially
        k_change = abs(current["kill_score"] - last.get("kill_score", 0.0))
        if k_change > self.change_eps_pct:
            return True
        
        # Check max age
        time_elapsed = time.time() - last.get("timestamp", 0)
        if time_elapsed > self.max_publish_age:
            return True
        
        return False
    
    def run_cycle(self):
        """Single publish cycle"""
        logger.info("=== Harvest Proposal Publish Cycle ===")
        
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
                
                # Get P1 risk proposal (optional)
                p1_proposal = self.get_risk_proposal(symbol)
                if not p1_proposal:
                    logger.debug(f"{symbol}: No P1 proposal, using fallback stop_dist_pct={self.fallback_stop_pct}")
                    p1_proposal = P1Proposal(
                        stop_dist_pct=self.fallback_stop_pct,
                        proposed_sl=None,
                    )
                
                # Compute harvest proposal
                harvest_output = compute_harvest_proposal(
                    position=position,
                    market_state=market_state,
                    p1_proposal=p1_proposal,
                    theta=self.theta,
                )
                
                # Rate limiting check
                if not self.should_publish(symbol, harvest_output):
                    logger.debug(f"{symbol}: No significant change, skipping publish")
                    # P2.7C.1: Update timestamp even when rate-limited (for staleness detection)
                    self.redis.hset(
                        f"quantum:harvest:proposal:{symbol}",
                        "last_update_epoch",
                        str(int(time.time()))
                    )
                    continue
                
                # Publish
                self.publish_proposal(symbol, harvest_output, market_state, position)
                
                # Update cache
                self.last_proposals[symbol] = {
                    "harvest_action": harvest_output["harvest_action"],
                    "new_sl_proposed": harvest_output.get("new_sl_proposed"),
                    "kill_score": harvest_output["kill_score"],
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
    parser = argparse.ArgumentParser(description="P2.5 Harvest Proposal Publisher")
    parser.add_argument("--once", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--symbols", default=os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT"))
    parser.add_argument("--interval", type=int, default=int(os.getenv("PUBLISH_INTERVAL_SEC", "10")))
    parser.add_argument("--position-source", default=os.getenv("POSITION_SOURCE", "auto"),
                       choices=["auto", "redis_hash", "redis_stream", "synthetic"])
    parser.add_argument("--enable-stream", action="store_true", 
                       default=os.getenv("ENABLE_STREAM", "false").lower() == "true")
    parser.add_argument("--fallback-stop-pct", type=float, 
                       default=float(os.getenv("FALLBACK_STOP_PCT", "0.02")))
    parser.add_argument("--max-publish-age", type=int,
                       default=int(os.getenv("MAX_PUBLISH_AGE_SEC", "60")))
    parser.add_argument("--change-eps-pct", type=float,
                       default=float(os.getenv("CHANGE_EPS_PCT", "0.001")))
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    publisher = HarvestProposalPublisher(
        redis_url=args.redis_url,
        symbols=symbols,
        publish_interval=args.interval,
        position_source=args.position_source,
        enable_stream=args.enable_stream,
        fallback_stop_pct=args.fallback_stop_pct,
        max_publish_age=args.max_publish_age,
        change_eps_pct=args.change_eps_pct,
    )
    
    if args.once:
        logger.info("Running single cycle (--once mode)")
        publisher.run_cycle()
        logger.info("Single cycle complete")
    else:
        publisher.run_loop()


if __name__ == "__main__":
    main()
