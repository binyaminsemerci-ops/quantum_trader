#!/usr/bin/env python3
"""
P2.7 Portfolio Clusters - Correlation Matrix + Capital Clustering

ROLE: Computes real correlation clusters and cluster stress for portfolio-aware exits.

INPUT:  quantum:position:snapshot:* (mark_price time series)
OUTPUT: quantum:portfolio:cluster_state (cluster metrics, ClusterStress)
        quantum:portfolio:clusters (cluster_id → [symbols])
        quantum:stream:portfolio.cluster_state (audit trail)

INTEGRATION: P2.6 Portfolio Gate uses ClusterStress instead of proxy correlation.

Fail-safe: If insufficient data, writes p27_corr_ready=0 and P2.6 falls back to proxy.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import math

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
    level=os.getenv("P27_LOG_LEVEL", "INFO"),
    format="%(asctime)s [P2.7] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ========== CONFIGURATION ==========

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

ALLOWLIST = os.getenv("P27_ALLOWLIST", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
LOOKBACK = int(os.getenv("P27_LOOKBACK", 360))  # minutes
UPDATE_SEC = int(os.getenv("P27_UPDATE_SEC", 60))
CLUSTER_CORR_MIN = float(os.getenv("P27_CLUSTER_CORR_MIN", 0.70))
METRICS_PORT = int(os.getenv("P27_METRICS_PORT", 8048))

EPSILON = 1e-9


# ========== PROMETHEUS METRICS ==========

if PROMETHEUS_AVAILABLE:
    p27_corr_ready = Gauge('p27_corr_ready', 'Correlation matrix ready (0/1)')
    p27_symbols_in_matrix = Gauge('p27_symbols_in_matrix', 'Symbols with sufficient data')
    p27_clusters_count = Gauge('p27_clusters_count', 'Number of clusters detected')
    p27_cluster_stress_max = Gauge('p27_cluster_stress_max', 'Maximum cluster stress')
    p27_cluster_stress_sum = Gauge('p27_cluster_stress_sum', 'Sum of cluster stress (ClusterStress)')
    p27_updates_total = Counter('p27_updates_total', 'Total cluster state updates')
    p27_fail_closed_total = Counter('p27_fail_closed_total', 'Fail-closed events', ['reason'])
    
    # Warmup observability
    p27_min_points_per_symbol = Gauge('p27_min_points_per_symbol', 'Minimum data points across all symbols (warmup tracker)')
    p27_points_per_symbol = Gauge('p27_points_per_symbol', 'Data points collected per symbol', ['symbol'])



# ========== DATA STRUCTURES ==========

@dataclass
class PricePoint:
    price: float
    ts_epoch: int


@dataclass
class ClusterMetrics:
    cluster_id: str
    symbols: List[str]
    notional: float
    weight: float
    concentration: float
    correlation: float
    stress: float
    ts_epoch: int


# ========== PORTFOLIO CLUSTERS ENGINE ==========

class PortfolioClusters:
    def __init__(self):
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        logger.info(f"Connected to Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
        logger.info(f"Allowlist: {ALLOWLIST}")
        logger.info(f"Lookback: {LOOKBACK} minutes, Update: {UPDATE_SEC}s")
        logger.info(f"Cluster correlation threshold: {CLUSTER_CORR_MIN}")
        
        # Ring buffers: symbol → deque of PricePoints
        self.price_buffers: Dict[str, deque] = {symbol: deque(maxlen=LOOKBACK + 1) for symbol in ALLOWLIST}
        self.last_update = 0
        
        if PROMETHEUS_AVAILABLE:
            p27_corr_ready.set(0)
    
    def fetch_snapshots(self) -> Dict[str, Dict]:
        """Fetch position snapshots for all symbols"""
        snapshots = {}
        for symbol in ALLOWLIST:
            key = f"quantum:position:snapshot:{symbol}"
            data = self.redis.hgetall(key)
            if data:
                snapshots[symbol] = data
        return snapshots
    
    def update_price_buffers(self, snapshots: Dict[str, Dict]):
        """Update price ring buffers from snapshots"""
        now = int(time.time())
        
        for symbol, data in snapshots.items():
            try:
                mark_price = float(data.get('mark_price', 0))
                ts_epoch = int(data.get('ts_epoch', now))
                
                if mark_price <= 0:
                    continue
                
                # Avoid duplicates (same timestamp)
                if self.price_buffers[symbol] and self.price_buffers[symbol][-1].ts_epoch == ts_epoch:
                    continue
                
                self.price_buffers[symbol].append(PricePoint(price=mark_price, ts_epoch=ts_epoch))
                logger.debug(f"{symbol}: Added price {mark_price:.2f} @ {ts_epoch} (buffer size: {len(self.price_buffers[symbol])})")
            
            except Exception as e:
                logger.error(f"{symbol}: Error parsing price: {e}")
    
    def compute_returns(self, symbol: str) -> Optional[List[float]]:
        """Compute log returns for symbol: r(t) = ln(p_t / p_{t-1})"""
        buffer = self.price_buffers[symbol]
        
        if len(buffer) < 2:
            return None
        
        prices = [p.price for p in buffer]
        returns = []
        
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                r = math.log(prices[i] / prices[i-1])
                returns.append(r)
        
        return returns if returns else None
    
    def standardize(self, returns: List[float]) -> List[float]:
        """Z-score normalization: (x - mean) / std"""
        if not returns:
            return []
        
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / n
        std = math.sqrt(variance) + EPSILON
        
        return [(r - mean) / std for r in returns]
    
    def compute_correlation(self, z1: List[float], z2: List[float]) -> float:
        """Pearson correlation between two standardized series"""
        if len(z1) != len(z2) or len(z1) == 0:
            return 0.0
        
        n = len(z1)
        corr = sum(z1[i] * z2[i] for i in range(n)) / n
        return max(-1.0, min(1.0, corr))  # Clamp to [-1, 1]
    
    def build_correlation_matrix(self, symbols: List[str]) -> Dict[Tuple[str, str], float]:
        """Build correlation matrix for symbols with sufficient data"""
        corr_matrix = {}
        returns_map = {}
        
        # Compute returns and standardize
        for symbol in symbols:
            returns = self.compute_returns(symbol)
            if returns and len(returns) >= 10:  # Minimum 10 data points
                returns_map[symbol] = self.standardize(returns)
        
        # Compute pairwise correlations
        valid_symbols = list(returns_map.keys())
        for i, s1 in enumerate(valid_symbols):
            for s2 in valid_symbols[i:]:
                if s1 == s2:
                    corr_matrix[(s1, s2)] = 1.0
                else:
                    # Align series length (use min length)
                    min_len = min(len(returns_map[s1]), len(returns_map[s2]))
                    z1 = returns_map[s1][-min_len:]
                    z2 = returns_map[s2][-min_len:]
                    corr = self.compute_correlation(z1, z2)
                    corr_matrix[(s1, s2)] = corr
                    corr_matrix[(s2, s1)] = corr
        
        return corr_matrix, valid_symbols
    
    def find_clusters(self, symbols: List[str], corr_matrix: Dict[Tuple[str, str], float]) -> List[Set[str]]:
        """Threshold clustering: connected components where corr >= CLUSTER_CORR_MIN"""
        # Build adjacency graph
        graph = defaultdict(set)
        for s1 in symbols:
            for s2 in symbols:
                if s1 != s2 and corr_matrix.get((s1, s2), 0) >= CLUSTER_CORR_MIN:
                    graph[s1].add(s2)
                    graph[s2].add(s1)
        
        # Find connected components (DFS)
        visited = set()
        clusters = []
        
        def dfs(node, cluster):
            visited.add(node)
            cluster.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)
        
        for symbol in symbols:
            if symbol not in visited:
                cluster = set()
                dfs(symbol, cluster)
                clusters.append(cluster)
        
        logger.info(f"Found {len(clusters)} clusters from {len(symbols)} symbols")
        return clusters
    
    def compute_cluster_metrics(self, cluster: Set[str], cluster_id: str, 
                                snapshots: Dict[str, Dict], 
                                corr_matrix: Dict[Tuple[str, str], float],
                                total_notional: float) -> Optional[ClusterMetrics]:
        """Compute metrics for a single cluster"""
        try:
            # Notional per symbol
            notionals = {}
            sides = {}
            
            for symbol in cluster:
                if symbol not in snapshots:
                    continue
                
                data = snapshots[symbol]
                position_amt = float(data.get('position_amt', 0))
                mark_price = float(data.get('mark_price', 0))
                
                notional_usd = data.get('notional_usd')
                if notional_usd:
                    notional = abs(float(notional_usd))
                elif mark_price > 0:
                    notional = abs(position_amt * mark_price)
                else:
                    notional = 0
                
                if notional > EPSILON:
                    notionals[symbol] = notional
                    sides[symbol] = 1 if position_amt > 0 else -1
            
            if not notionals:
                return None
            
            # Cluster notional
            cluster_notional = sum(notionals.values())
            
            # Cluster weight
            weight = cluster_notional / max(total_notional, EPSILON)
            
            # Cluster concentration (net directional bias)
            net_notional = sum(sides[s] * notionals[s] for s in notionals)
            concentration = abs(net_notional) / max(cluster_notional, EPSILON)
            
            # Cluster correlation (mean pairwise corr within cluster)
            cluster_list = list(notionals.keys())
            if len(cluster_list) < 2:
                cluster_corr = 0.0
            else:
                corr_sum = 0
                count = 0
                for i, s1 in enumerate(cluster_list):
                    for s2 in cluster_list[i+1:]:
                        corr_sum += corr_matrix.get((s1, s2), 0)
                        count += 1
                cluster_corr = corr_sum / max(count, 1)
            
            # Cluster stress
            stress = max(0.0, min(1.0, weight * concentration * cluster_corr))
            
            return ClusterMetrics(
                cluster_id=cluster_id,
                symbols=sorted(list(cluster)),
                notional=cluster_notional,
                weight=weight,
                concentration=concentration,
                correlation=cluster_corr,
                stress=stress,
                ts_epoch=int(time.time())
            )
        
        except Exception as e:
            logger.error(f"Error computing metrics for cluster {cluster_id}: {e}")
            return None
    
    def write_cluster_state(self, cluster_metrics: List[ClusterMetrics], cluster_stress: float):
        """Write cluster state to Redis"""
        try:
            now = int(time.time())
            
            # Global cluster state
            self.redis.hset('quantum:portfolio:cluster_state', mapping={
                'updated_ts': now,
                'cluster_stress': cluster_stress,
                'clusters_count': len(cluster_metrics)
            })
            
            # Individual cluster data
            cluster_mapping = {}
            for cm in cluster_metrics:
                cluster_mapping[cm.cluster_id] = json.dumps(cm.symbols)
                
                # Cluster details
                self.redis.hset(f'quantum:portfolio:cluster_state:{cm.cluster_id}', mapping={
                    'notional': cm.notional,
                    'weight': cm.weight,
                    'concentration': cm.concentration,
                    'correlation': cm.correlation,
                    'stress': cm.stress,
                    'updated_ts': cm.ts_epoch,
                    'symbols': json.dumps(cm.symbols)
                })
            
            if cluster_mapping:
                self.redis.hset('quantum:portfolio:clusters', mapping=cluster_mapping)
            
            # Optional stream audit
            audit_payload = {
                'cluster_stress': cluster_stress,
                'clusters_count': len(cluster_metrics),
                'clusters': json.dumps([asdict(cm) for cm in cluster_metrics]),
                'ts_epoch': now
            }
            self.redis.xadd('quantum:stream:portfolio.cluster_state', audit_payload)
            
            logger.info(f"Wrote cluster state: {len(cluster_metrics)} clusters, ClusterStress={cluster_stress:.3f}")
        
        except Exception as e:
            logger.error(f"Error writing cluster state: {e}", exc_info=True)
    
    def update_clusters(self):
        """Main cluster computation and update"""
        try:
            # Fetch snapshots
            snapshots = self.fetch_snapshots()
            if not snapshots:
                logger.warning("No snapshots available, fail-closed")
                if PROMETHEUS_AVAILABLE:
                    p27_fail_closed_total.labels(reason='no_snapshots').inc()
                    p27_corr_ready.set(0)
                return
            
            # Update price buffers
            self.update_price_buffers(snapshots)
            
            # Track warmup progress
            if PROMETHEUS_AVAILABLE:
                for symbol in ALLOWLIST:
                    buf_len = len(self.price_buffers.get(symbol, []))
                    p27_points_per_symbol.labels(symbol=symbol).set(buf_len)
                
                all_lengths = [len(self.price_buffers[s]) for s in ALLOWLIST]
                p27_min_points_per_symbol.set(min(all_lengths) if all_lengths else 0)
            
            # Check if we have enough data
            ready_symbols = [s for s in ALLOWLIST if len(self.price_buffers[s]) >= 10]
            if len(ready_symbols) < 2:
                logger.warning(f"Insufficient price data (need 10+ points per symbol, have {len(ready_symbols)} ready)")
                if PROMETHEUS_AVAILABLE:
                    p27_fail_closed_total.labels(reason='insufficient_data').inc()
                    p27_corr_ready.set(0)
                return
            
            # Build correlation matrix
            corr_matrix, valid_symbols = self.build_correlation_matrix(ready_symbols)
            
            if len(valid_symbols) < 2:
                logger.warning("Correlation matrix failed (not enough valid symbols)")
                if PROMETHEUS_AVAILABLE:
                    p27_fail_closed_total.labels(reason='corr_failed').inc()
                    p27_corr_ready.set(0)
                return
            
            # Find clusters
            clusters = self.find_clusters(valid_symbols, corr_matrix)
            
            # Compute total notional
            total_notional = 0
            for symbol, data in snapshots.items():
                try:
                    notional_usd = data.get('notional_usd')
                    if notional_usd:
                        total_notional += abs(float(notional_usd))
                    else:
                        position_amt = float(data.get('position_amt', 0))
                        mark_price = float(data.get('mark_price', 0))
                        if mark_price > 0:
                            total_notional += abs(position_amt * mark_price)
                except:
                    pass
            
            # Compute cluster metrics
            cluster_metrics = []
            for i, cluster in enumerate(clusters):
                cluster_id = f"cluster_{i}"
                cm = self.compute_cluster_metrics(cluster, cluster_id, snapshots, corr_matrix, total_notional)
                if cm:
                    cluster_metrics.append(cm)
                    logger.info(f"{cluster_id}: {cm.symbols} - notional=${cm.notional:.2f}, stress={cm.stress:.3f}")
            
            # Compute ClusterStress
            cluster_stress = max(0.0, min(1.0, sum(cm.stress for cm in cluster_metrics)))
            
            # Write to Redis
            self.write_cluster_state(cluster_metrics, cluster_stress)
            
            # Update metrics
            if PROMETHEUS_AVAILABLE:
                p27_corr_ready.set(1)
                p27_symbols_in_matrix.set(len(valid_symbols))
                p27_clusters_count.set(len(cluster_metrics))
                p27_cluster_stress_max.set(max([cm.stress for cm in cluster_metrics], default=0))
                p27_cluster_stress_sum.set(cluster_stress)
                p27_updates_total.inc()
            
        except Exception as e:
            logger.error(f"Error in cluster update: {e}", exc_info=True)
            if PROMETHEUS_AVAILABLE:
                p27_fail_closed_total.labels(reason='exception').inc()
    
    def run(self):
        """Main loop"""
        logger.info("=== P2.7 Portfolio Clusters Started ===")
        logger.info(f"Metrics: http://localhost:{METRICS_PORT}/metrics")
        
        if PROMETHEUS_AVAILABLE:
            start_http_server(METRICS_PORT)
        
        while True:
            try:
                self.update_clusters()
                time.sleep(UPDATE_SEC)
            except KeyboardInterrupt:
                logger.info("Shutting down (KeyboardInterrupt)")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(UPDATE_SEC)


def main():
    engine = PortfolioClusters()
    engine.run()


if __name__ == "__main__":
    main()
