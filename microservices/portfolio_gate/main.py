#!/usr/bin/env python3
"""
P2.6 Portfolio Gate - Intelligent Harvesting Portfolio-Level Controls

ROLE: Prevents panic FULL_CLOSE when portfolio is cold, accelerates risk-off when hot.

INPUT:  quantum:stream:harvest.proposal
STATE:  quantum:position:snapshot:* (from P3.3)
OUTPUT: quantum:stream:portfolio.gate (decision per plan_id)
PERMIT: quantum:permit:p26:{plan_id} = "1" TTL

Fail-closed: Missing/invalid inputs → HOLD + no permit.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed. Install with: pip install redis")
    sys.exit(1)

try:
    from prometheus_client import Counter, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("WARN: prometheus_client not installed, metrics disabled")


logging.basicConfig(
    level=os.getenv("P26_LOG_LEVEL", "INFO"),
    format="%(asctime)s [P2.6] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ========== CONFIGURATION ==========

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

ALLOWLIST = os.getenv("P26_ALLOWLIST", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
PERMIT_TTL = int(os.getenv("P26_PERMIT_TTL", 60))
COOLDOWN_SEC = int(os.getenv("P26_COOLDOWN_SEC", 60))
POLL_INTERVAL = int(os.getenv("P26_POLL_INTERVAL", 5))

# Portfolio stress config
H_MIN = float(os.getenv("P26_H_MIN", 0.05))
H_MAX = float(os.getenv("P26_H_MAX", 0.35))
STRESS_FULLCLOSE_MIN = float(os.getenv("P26_STRESS_FULLCLOSE_MIN", 0.55))
STRESS_RISKOFF = float(os.getenv("P26_STRESS_RISKOFF", 0.80))
KS_HARD = float(os.getenv("P26_KS_HARD", 0.85))
A_COEF = float(os.getenv("P26_A", 0.60))
B_COEF = float(os.getenv("P26_B", 0.25))
C_COEF = float(os.getenv("P26_C", 0.15))

METRICS_PORT = int(os.getenv("P26_METRICS_PORT", 8047))

STREAM_HARVEST_PROPOSAL = "quantum:stream:harvest.proposal"
STREAM_PORTFOLIO_GATE = "quantum:stream:portfolio.gate"
CONSUMER_GROUP = "p26_portfolio_gate"
CONSUMER_NAME = f"p26_{os.getpid()}"

EPSILON = 1e-9


# ========== PROMETHEUS METRICS ==========

if PROMETHEUS_AVAILABLE:
    from prometheus_client import Histogram
    
    p26_stress = Gauge('p26_stress', 'Portfolio stress level (0-1)')
    p26_heat = Gauge('p26_heat', 'Portfolio heat metric (0-1 normalized)')
    p26_concentration = Gauge('p26_concentration', 'Directional concentration (0-1)')
    p26_corr_proxy = Gauge('p26_corr_proxy', 'Correlation proxy (0-1)')
    
    # Observability: portfolio state visibility
    p26_symbols_with_snapshot = Gauge('p26_symbols_with_snapshot', 'Number of symbols with valid snapshots')
    p26_total_abs_notional = Gauge('p26_total_abs_notional', 'Total absolute notional across portfolio (USD)')
    p26_snapshot_age_seconds = Gauge('p26_snapshot_age_seconds', 'Age of oldest snapshot in seconds (staleness detector)', ['symbol'])
    
    p26_stream_reads = Counter('p26_stream_reads_total', 'Total stream read attempts')
    p26_plans_seen = Counter('p26_plans_seen_total', 'Total proposals seen', ['action_proposed'])
    p26_gate_writes = Counter('p26_gate_writes_total', 'Gate decisions written to stream')
    p26_actions_downgraded = Counter('p26_actions_downgraded_total', 'Actions downgraded', ['from_action', 'to_action', 'reason'])
    p26_permit_issued = Counter('p26_permit_issued_total', 'Permits issued')
    p26_fail_closed = Counter('p26_fail_closed_total', 'Fail-closed events', ['reason'])
    p26_cluster_fallback = Counter('p26_cluster_fallback_total', 'Cluster stress fallback to proxy (P2.7 unavailable)')
    p26_cluster_stress_used = Gauge('p26_cluster_stress_used', 'Cluster stress from P2.7 (0=proxy, 1=cluster)')

    
    # Latency histogram: harvest.proposal → permit:p26:{plan_id}
    p26_time_to_permit_seconds = Histogram(
        'p26_time_to_permit_seconds',
        'Time from proposal read to permit issuance',
        buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0)
    )


# ========== DATA STRUCTURES ==========

@dataclass
class HarvestProposal:
    plan_id: str
    symbol: str
    action_proposed: str  # HOLD|PARTIAL_25|PARTIAL_50|PARTIAL_75|FULL_CLOSE_PROPOSED
    kill_score: float
    r_net: Optional[float]
    ts_epoch: int


@dataclass
class PositionSnapshot:
    symbol: str
    position_amt: float
    notional_usd: float
    mark_price: Optional[float]
    ts_epoch: int
    stale: bool = False


@dataclass
class PortfolioMetrics:
    heat: float  # 0-1 normalized
    concentration: float  # 0-1
    corr_proxy: float  # 0-1
    stress: float  # 0-1
    total_notional: float


@dataclass
class GateDecision:
    plan_id: str
    symbol: str
    action_proposed: str
    final_action: str
    gate_reason: str
    stress: float
    heat: float
    concentration: float
    corr_proxy: float
    kill_score: float
    ts_epoch: int


# ========== PORTFOLIO GATE ==========

class PortfolioGate:
    def __init__(self):
        self.redis = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True
        )
        logger.info(f"Connected to Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
        logger.info(f"Allowlist: {ALLOWLIST}")
        logger.info(f"Stress config: H_MIN={H_MIN}, H_MAX={H_MAX}, STRESS_FC={STRESS_FULLCLOSE_MIN}, STRESS_RISKOFF={STRESS_RISKOFF}")
        
        # Create consumer group (idempotent)
        try:
            self.redis.xgroup_create(STREAM_HARVEST_PROPOSAL, CONSUMER_GROUP, id='0', mkstream=True)
            logger.info(f"Consumer group '{CONSUMER_GROUP}' created")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{CONSUMER_GROUP}' already exists")
            else:
                raise
    
    def get_cluster_stress(self) -> Optional[float]:
        """Read cluster stress from P2.7 (with freshness check and fallback)"""
        try:
            data = self.redis.hgetall('quantum:portfolio:cluster_state')
            if not data:
                return None
            
            cluster_stress = float(data.get('cluster_stress', 0))
            updated_ts = int(data.get('updated_ts', 0))
            now = int(time.time())
            age = now - updated_ts
            
            # Freshness check: P2.7 should update every UPDATE_SEC seconds
            # Allow 2x grace period
            p27_update_sec = int(os.getenv("P27_UPDATE_SEC", 60))
            if age > 2 * p27_update_sec:
                logger.debug(f"Cluster stress stale (age={age}s), using fallback")
                return None
            
            logger.debug(f"Using cluster stress from P2.7: {cluster_stress:.3f} (age={age}s)")
            return cluster_stress
        
        except Exception as e:
            logger.debug(f"Error reading cluster stress: {e}")
            return None
    
    def get_position_snapshots(self) -> Dict[str, PositionSnapshot]:
        """Fetch position snapshots for all allowlist symbols"""
        snapshots = {}
        now = time.time()
        stale_threshold = 300  # 5 minutes
        
        for symbol in ALLOWLIST:
            key = f"quantum:position:snapshot:{symbol}"
            data = self.redis.hgetall(key)
            
            if not data:
                logger.debug(f"{symbol}: No position snapshot found")
                continue
            
            try:
                position_amt = float(data.get('position_amt', 0))
                ts_epoch = int(data.get('ts_epoch', 0))
                age = now - ts_epoch
                stale = age > stale_threshold
                
                # Try to get notional_usd directly
                notional_usd = data.get('notional_usd')
                mark_price = data.get('mark_price')
                
                if notional_usd:
                    notional_usd = float(notional_usd)
                elif mark_price:
                    # Derive notional if mark_price available
                    # position_amt=0 is VALID (flat state), so notional=0
                    mark_price = float(mark_price)
                    notional_usd = abs(position_amt * mark_price)
                else:
                    logger.warning(f"{symbol}: Cannot determine notional (missing mark_price)")
                    continue
                
                snapshots[symbol] = PositionSnapshot(
                    symbol=symbol,
                    position_amt=position_amt,
                    notional_usd=abs(notional_usd),
                    mark_price=float(mark_price) if mark_price else None,
                    ts_epoch=ts_epoch,
                    stale=stale
                )
                
                # Track snapshot age for staleness detection
                if PROMETHEUS_AVAILABLE:
                    p26_snapshot_age_seconds.labels(symbol=symbol).set(age)
                
                if stale:
                    logger.warning(f"{symbol}: Snapshot is stale (age={age:.0f}s)")
            
            except Exception as e:
                logger.error(f"{symbol}: Error parsing snapshot: {e}")
                continue
        
        return snapshots
    
    def compute_portfolio_metrics(self, snapshots: Dict[str, PositionSnapshot]) -> Optional[PortfolioMetrics]:
        """
        Compute portfolio-level metrics:
        - Heat (volatility-weighted notional)
        - Concentration (directional bias)
        - Correlation proxy (alignment of positions)
        - Stress (weighted composite)
        """
        if not snapshots:
            logger.warning("No valid snapshots, fail-closed")
            if PROMETHEUS_AVAILABLE:
                p26_fail_closed.labels(reason='no_snapshots').inc()
            return None
        
        # Filter stale
        fresh = {s: snap for s, snap in snapshots.items() if not snap.stale}
        if not fresh:
            logger.warning("All snapshots stale, fail-closed")
            if PROMETHEUS_AVAILABLE:
                p26_fail_closed.labels(reason='all_stale').inc()
            return None
        
        # Total notional
        total_notional = sum(snap.notional_usd for snap in fresh.values())
        if total_notional < EPSILON:
            logger.info("Total notional near zero, portfolio cold (stress=0)")
            return PortfolioMetrics(heat=0.0, concentration=0.0, corr_proxy=0.0, stress=0.0, total_notional=0.0)
        
        # Weights
        weights = {s: snap.notional_usd / total_notional for s, snap in fresh.items()}
        
        # Sigma proxy (simplified: use 1.0 neutral if no sigma store)
        # TODO: Integrate with MarketState sigma if available
        sigmas = {s: 1.0 for s in fresh.keys()}
        logger.debug("Using neutral sigma=1.0 for all symbols (sigma store not integrated)")
        
        # Portfolio heat
        heat_raw = sum(weights[s] * sigmas[s] for s in fresh.keys())
        heat_normalized = max(0.0, min(1.0, (heat_raw - H_MIN) / max(H_MAX - H_MIN, EPSILON)))
        
        # Directional concentration
        net_exposure = sum(
            (1 if snap.position_amt > 0 else -1 if snap.position_amt < 0 else 0) * snap.notional_usd
            for snap in fresh.values()
        )
        gross_exposure = sum(snap.notional_usd for snap in fresh.values())
        concentration = abs(net_exposure) / max(gross_exposure, EPSILON)
        
        # Correlation proxy (cheap: fraction aligned with net direction)
        net_sign = 1 if net_exposure > 0 else -1 if net_exposure < 0 else 0
        if net_sign == 0:
            corr_proxy = 0.0
        else:
            corr_proxy = sum(
                weights[s] for s, snap in fresh.items()
                if ((snap.position_amt > 0 and net_sign > 0) or (snap.position_amt < 0 and net_sign < 0))
            )
        
        # Correlation/clustering: Use P2.7 cluster stress if available, else fallback to proxy
        cluster_stress = self.get_cluster_stress()
        if cluster_stress is not None:
            K = cluster_stress
            cluster_used = True
            logger.debug(f"Using cluster stress from P2.7: K={K:.3f}")
        else:
            K = corr_proxy
            cluster_used = False
            logger.debug(f"P2.7 unavailable, using proxy correlation: K={K:.3f}")
            if PROMETHEUS_AVAILABLE:
                p26_cluster_fallback.inc()
        
        # Stress composite (with cluster stress)
        stress = max(0.0, min(1.0, A_COEF * heat_normalized + B_COEF * concentration + C_COEF * K))
        
        logger.info(f"Portfolio metrics: heat={heat_normalized:.3f}, conc={concentration:.3f}, corr_proxy={corr_proxy:.3f}, K={K:.3f} ({'cluster' if cluster_used else 'proxy'}), stress={stress:.3f}, notional=${total_notional:.2f}")
        
        # Update Prometheus
        if PROMETHEUS_AVAILABLE:
            p26_heat.set(heat_normalized)
            p26_concentration.set(concentration)
            p26_corr_proxy.set(corr_proxy)
            p26_stress.set(stress)
            p26_symbols_with_snapshot.set(len(fresh))
            p26_total_abs_notional.set(total_notional)
            p26_cluster_stress_used.set(1 if cluster_used else 0)
        
        return PortfolioMetrics(
            heat=heat_normalized,
            concentration=concentration,
            corr_proxy=corr_proxy,
            stress=stress,
            total_notional=total_notional
        )
    
    def apply_policy(
        self,
        proposal: HarvestProposal,
        metrics: Optional[PortfolioMetrics]
    ) -> GateDecision:
        """
        Apply portfolio gate policy to harvest proposal.
        
        Policy:
        1. Allowlist check
        2. Fail-closed if portfolio state missing
        3. Cooldown downgrade
        4. Anti-panic (forbid FULL_CLOSE if stress < threshold)
        5. Risk-off accelerator (promote partials if stress high)
        6. Pass-through otherwise
        """
        symbol = proposal.symbol
        action_proposed = proposal.action_proposed
        kill_score = proposal.kill_score
        
        # Default: fail-closed
        final_action = "HOLD"
        gate_reason = "init"
        
        stress = metrics.stress if metrics else 0.0
        heat = metrics.heat if metrics else 0.0
        conc = metrics.concentration if metrics else 0.0
        corr = metrics.corr_proxy if metrics else 0.0
        
        # Policy 1: Allowlist
        if symbol not in ALLOWLIST:
            final_action = "HOLD"
            gate_reason = "not_allowed"
            logger.warning(f"{symbol}: HOLD - Not in allowlist (allowed: {ALLOWLIST})")
            if PROMETHEUS_AVAILABLE:
                p26_fail_closed.labels(reason='not_allowed').inc()
            
            return GateDecision(
                plan_id=proposal.plan_id,
                symbol=symbol,
                action_proposed=action_proposed,
                final_action=final_action,
                gate_reason=gate_reason,
                stress=stress,
                heat=heat,
                concentration=conc,
                corr_proxy=corr,
                kill_score=kill_score,
                ts_epoch=int(time.time())
            )
        
        # Policy 2: Fail-closed if portfolio state missing
        if metrics is None:
            final_action = "HOLD"
            gate_reason = "fail_closed_portfolio_state"
            logger.error(f"{symbol}: HOLD - Fail-closed due to missing/invalid portfolio state (no valid snapshots or all stale)")
            if PROMETHEUS_AVAILABLE:
                p26_fail_closed.labels(reason='portfolio_state').inc()
            
            return GateDecision(
                plan_id=proposal.plan_id,
                symbol=symbol,
                action_proposed=action_proposed,
                final_action=final_action,
                gate_reason=gate_reason,
                stress=stress,
                heat=heat,
                concentration=conc,
                corr_proxy=corr,
                kill_score=kill_score,
                ts_epoch=int(time.time())
            )
        
        # Start with proposed action
        final_action = action_proposed
        gate_reason = "pass"
        
        # Policy 3: Cooldown downgrade
        cooldown_key = f"quantum:p26:cooldown:{symbol}"
        if self.redis.exists(cooldown_key):
            downgrade_map = {
                "FULL_CLOSE_PROPOSED": "PARTIAL_75",
                "PARTIAL_75": "PARTIAL_50",
                "PARTIAL_50": "PARTIAL_25",
                "PARTIAL_25": "HOLD"
            }
            if final_action in downgrade_map:
                old_action = final_action
                final_action = downgrade_map[final_action]
                gate_reason = "cooldown_downgrade"
                logger.info(f"{symbol}: Cooldown active, downgrade {old_action} → {final_action}")
                if PROMETHEUS_AVAILABLE:
                    p26_actions_downgraded.labels(from_action=old_action, to_action=final_action, reason='cooldown').inc()
        
        # Policy 4: Anti-panic (forbid FULL_CLOSE if portfolio cold)
        if final_action == "FULL_CLOSE_PROPOSED" and stress < STRESS_FULLCLOSE_MIN:
            if kill_score >= KS_HARD:
                final_action = "PARTIAL_75"
                gate_reason = "forbid_full_close_portfolio_cold_ks_hard"
                logger.info(f"{symbol}: Portfolio cold (stress={stress:.3f}), but kill_score={kill_score:.3f} >= {KS_HARD}, downgrade to PARTIAL_75")
            else:
                final_action = "PARTIAL_50"
                gate_reason = "forbid_full_close_portfolio_cold"
                logger.info(f"{symbol}: Portfolio cold (stress={stress:.3f}), forbid FULL_CLOSE → PARTIAL_50")
            
            if PROMETHEUS_AVAILABLE:
                p26_actions_downgraded.labels(from_action='FULL_CLOSE_PROPOSED', to_action=final_action, reason='anti_panic').inc()
        
        # Policy 5: Risk-off accelerator (promote partials if stress high)
        elif stress >= STRESS_RISKOFF:
            upgrade_map = {
                "PARTIAL_25": "PARTIAL_75",
                "PARTIAL_50": "PARTIAL_75",
                "PARTIAL_75": "PARTIAL_75"  # already max for accelerator
            }
            if final_action in upgrade_map and final_action != upgrade_map[final_action]:
                old_action = final_action
                final_action = upgrade_map[final_action]
                gate_reason = "riskoff_accelerate"
                logger.info(f"{symbol}: High stress (stress={stress:.3f}), accelerate {old_action} → {final_action}")
                if PROMETHEUS_AVAILABLE:
                    p26_actions_downgraded.labels(from_action=old_action, to_action=final_action, reason='riskoff').inc()
        
        return GateDecision(
            plan_id=proposal.plan_id,
            symbol=symbol,
            action_proposed=action_proposed,
            final_action=final_action,
            gate_reason=gate_reason,
            stress=stress,
            heat=heat,
            concentration=conc,
            corr_proxy=corr,
            kill_score=kill_score,
            ts_epoch=int(time.time())
        )
    
    def set_cooldown(self, symbol: str):
        """Set cooldown key to prevent rapid-fire actions"""
        key = f"quantum:p26:cooldown:{symbol}"
        self.redis.setex(key, COOLDOWN_SEC, 1)
        logger.debug(f"{symbol}: Cooldown set for {COOLDOWN_SEC}s")
    
    def issue_permit(self, plan_id: str):
        """Issue P2.6 permit for Apply Layer"""
        key = f"quantum:permit:p26:{plan_id}"
        self.redis.setex(key, PERMIT_TTL, "1")
        logger.info(f"Permit issued: {plan_id} (TTL={PERMIT_TTL}s)")
        if PROMETHEUS_AVAILABLE:
            p26_permit_issued.inc()
    
    def publish_decision(self, decision: GateDecision):
        """Publish gate decision to stream"""
        fields = asdict(decision)
        # Convert all values to strings for Redis
        fields = {k: str(v) for k, v in fields.items()}
        
        self.redis.xadd(STREAM_PORTFOLIO_GATE, fields)
        logger.debug(f"Published decision: {decision.symbol} {decision.final_action} (reason: {decision.gate_reason})")
        
        # Track gate writes for "plans flowing but no output" alarm
        if PROMETHEUS_AVAILABLE:
            p26_gate_writes.inc()
    
    def process_proposals(self):
        """Process batch of harvest proposals"""
        if PROMETHEUS_AVAILABLE:
            p26_stream_reads.inc()
        
        try:
            # Read from stream (blocking with timeout)
            messages = self.redis.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {STREAM_HARVEST_PROPOSAL: '>'},
                count=10,
                block=5000  # 5s block
            )
            
            if not messages:
                logger.debug("No new proposals")
                return
            
            # Get portfolio state once for this batch
            snapshots = self.get_position_snapshots()
            metrics = self.compute_portfolio_metrics(snapshots)
            
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    t0 = time.time()  # Start latency timer
                    try:
                        # Parse proposal (support legacy 'action' field for backwards compat)
                        proposal = HarvestProposal(
                            plan_id=fields['plan_id'],
                            symbol=fields['symbol'],
                            action_proposed=fields.get('action_proposed', fields.get('harvest_action', fields.get('action', 'HOLD'))),
                            kill_score=float(fields.get('kill_score', 0)),
                            r_net=float(fields['r_net']) if 'r_net' in fields and fields['r_net'] else None,
                            ts_epoch=int(fields.get('ts_epoch', time.time()))
                        )
                        
                        logger.info(f"Processing: {proposal.symbol} {proposal.action_proposed} (ks={proposal.kill_score:.3f})")
                        
                        if PROMETHEUS_AVAILABLE:
                            p26_plans_seen.labels(action_proposed=proposal.action_proposed).inc()
                        
                        # Apply policy
                        decision = self.apply_policy(proposal, metrics)
                        
                        # Publish decision
                        self.publish_decision(decision)
                        
                        # Issue permit if action is not HOLD
                        if decision.final_action != "HOLD":
                            self.issue_permit(decision.plan_id)
                            self.set_cooldown(decision.symbol)
                            
                            # Record latency for successful permit issuance
                            if PROMETHEUS_AVAILABLE:
                                p26_time_to_permit_seconds.observe(time.time() - t0)
                        else:
                            logger.info(f"{decision.symbol}: No permit issued (final_action=HOLD, reason={decision.gate_reason})")
                        
                        # ACK message
                        self.redis.xack(STREAM_HARVEST_PROPOSAL, CONSUMER_GROUP, msg_id)
                    
                    except Exception as e:
                        logger.error(f"Error processing message {msg_id}: {e}", exc_info=True)
                        # Still ACK to avoid reprocessing bad messages
                        self.redis.xack(STREAM_HARVEST_PROPOSAL, CONSUMER_GROUP, msg_id)
        
        except Exception as e:
            logger.error(f"Error reading stream: {e}", exc_info=True)
    
    def run(self):
        """Main loop"""
        logger.info("=== P2.6 Portfolio Gate Started ===")
        logger.info(f"Metrics: http://localhost:{METRICS_PORT}/metrics")
        
        if PROMETHEUS_AVAILABLE:
            start_http_server(METRICS_PORT)
        
        while True:
            try:
                self.process_proposals()
            except KeyboardInterrupt:
                logger.info("Shutting down (KeyboardInterrupt)")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(POLL_INTERVAL)


def main():
    gate = PortfolioGate()
    gate.run()


if __name__ == "__main__":
    main()
