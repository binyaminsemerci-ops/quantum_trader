"""
P0 MarketState Module - LOCKED SPEC v1.0 (calc-only)

Implements the P0 MarketState specification exactly:
- Robust volatility (sigma): winsorized EWMA + MAD fallback + spike-proxy blend
- Robust trend (mu): multi-window Theil-Sen/Huber on log prices
- Trend Strength (TS): abs(mu)/(sigma+eps)
- Regime probabilities: softmax over TREND/MR/CHOP using dp/vr/ts features

NO trading actions. NO intents. Pure calculation only.
"""

import logging
from typing import Dict, List, Optional, Union
import numpy as np
from scipy import stats
from scipy.special import softmax
import time

logger = logging.getLogger(__name__)


# Default theta configuration (all overridable)
DEFAULT_THETA = {
    'eps': 1e-12,
    'vol': {
        'window': 256,
        'winsor_q_low': 0.01,
        'winsor_q_high': 0.99,
        'ewma_lambda': 0.94,
        'sigma_floor': 1e-6,
        'spike_proxy_cap': 5.0,
        'spike_center': 1.2,
        'spike_scale': 0.3
    },
    'trend': {
        'windows': [64, 128, 256],
        'weights': [0.5, 0.3, 0.2],
        'method': 'theil_sen'  # or 'huber', 'ols'
    },
    'regime': {
        'window': 256,
        'vr_k': 5,
        'softmax_temp': 1.0,
        'score': {
            'a1': 1.0, 'a2': 1.0, 'a3': 1.0,  # trend coeffs
            'b1': 1.0, 'b2': 1.0, 'b3': 1.0,  # mr coeffs
            'c1': 1.0, 'c2': 1.0, 'c3': 1.0   # chop coeffs
        }
    }
}


def sigmoid(x: float) -> float:
    """Sigmoid function: 1 / (1 + exp(-x))"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class MarketState:
    """
    P0 MarketState calculator - SPEC v1.0
    
    Usage:
        ms = MarketState(theta=custom_theta)
        state = ms.get_state(symbol, prices)
        # → {'sigma': ..., 'mu': ..., 'ts': ..., 'regime_probs': {...}}
    """
    
    def __init__(self, theta: Optional[Dict] = None):
        """
        Initialize with theta configuration
        
        Args:
            theta: Optional config dict. Defaults to DEFAULT_THETA if None.
        """
        self.theta = {**DEFAULT_THETA, **(theta or {})}
        
        # Deep merge nested dicts
        for key in ['vol', 'trend', 'regime']:
            if key in DEFAULT_THETA:
                self.theta[key] = {**DEFAULT_THETA[key], **self.theta.get(key, {})}
        
        # Deep merge regime.score
        if 'regime' in self.theta and 'score' in DEFAULT_THETA['regime']:
            self.theta['regime']['score'] = {
                **DEFAULT_THETA['regime']['score'],
                **self.theta.get('regime', {}).get('score', {})
            }
        
        logger.info(f"MarketState initialized with theta.trend.method={self.theta['trend']['method']}")
    
    def get_state(self, symbol: str, prices: Union[List[float], np.ndarray]) -> Optional[Dict]:
        """
        Compute market state for a symbol given price series
        
        Args:
            symbol: Symbol name (for logging)
            prices: Close prices (list or array)
        
        Returns:
            Dict with MUST fields: sigma, mu, ts, regime_probs
            SHOULD fields: features, windows, ts_components (for debug)
            None if insufficient data
        """
        prices = np.array(prices, dtype=float)
        
        # Check minimum data
        min_required = max(
            self.theta['vol']['window'],
            max(self.theta['trend']['windows']),
            self.theta['regime']['window']
        ) + 1
        
        if len(prices) < min_required:
            logger.warning(f"{symbol}: Insufficient data ({len(prices)} < {min_required})")
            return None
        
        try:
            # Compute log returns
            log_prices = np.log(prices + self.theta['eps'])
            returns = np.diff(log_prices)
            
            if len(returns) < 10:
                return None
            
            # === STEP 1: SIGMA (robust volatility) ===
            sigma, sigma_components = self._compute_sigma(returns)
            
            # === STEP 2: MU (robust trend) ===
            mu = self._compute_mu(log_prices)
            
            # === STEP 3: TS (trend strength) ===
            ts = abs(mu) / (sigma + self.theta['eps'])
            
            # === STEP 4: REGIME PROBS ===
            dp = self._compute_dp(returns)
            vr = self._compute_vr(returns)
            regime_probs = self._compute_regime_probs(ts, dp, vr)
            
            # Build output
            state = {
                # MUST fields
                'sigma': float(sigma),
                'mu': float(mu),
                'ts': float(ts),
                'regime_probs': regime_probs,
                
                # SHOULD fields (debug/proof)
                'features': {
                    'dp': float(dp),
                    'vr': float(vr),
                    'spike_proxy': float(sigma_components['spike_proxy'])
                },
                'windows': {
                    'returns_n': len(returns),
                    'trend_windows': self.theta['trend']['windows'],
                    'regime_window': self.theta['regime']['window']
                },
                'ts_components': {
                    'mu_abs': float(abs(mu)),
                    'sigma_used': float(sigma)
                }
            }
            
            # Optional rate-limited logging
            self._log_state(symbol, state)
            
            return state
            
        except Exception as e:
            logger.error(f"{symbol}: Failed to compute state: {e}", exc_info=True)
            return None
    
    def _compute_sigma(self, returns: np.ndarray) -> tuple:
        """
        Compute robust sigma per SPEC:
        A) Winsorize returns
        B) EWMA variance
        C) MAD fallback
        D) Spike proxy + blend
        
        Returns:
            (sigma, components_dict)
        """
        eps = self.theta['eps']
        vol_cfg = self.theta['vol']
        
        # A) Winsorize
        window = min(vol_cfg['window'], len(returns))
        returns_window = returns[-window:]
        
        q_low = np.quantile(returns_window, vol_cfg['winsor_q_low'])
        q_high = np.quantile(returns_window, vol_cfg['winsor_q_high'])
        r_w = np.clip(returns_window, q_low, q_high)
        
        # B) EWMA variance
        lam = vol_cfg['ewma_lambda']
        v = np.var(r_w[:min(10, len(r_w))])  # Initialize
        for r in r_w:
            v = lam * v + (1 - lam) * r ** 2
        sigma_ewma = np.sqrt(v)
        
        # C) MAD fallback
        median = np.median(r_w)
        mad = np.median(np.abs(r_w - median))
        sigma_mad = 1.4826 * mad
        
        # D) Spike proxy + blend
        spike_proxy = np.clip(
            sigma_mad / (sigma_ewma + eps),
            0,
            vol_cfg['spike_proxy_cap']
        )
        
        w = sigmoid(
            (spike_proxy - vol_cfg['spike_center']) / vol_cfg['spike_scale']
        )
        
        sigma_raw = (1 - w) * sigma_ewma + w * sigma_mad
        sigma = max(vol_cfg['sigma_floor'], sigma_raw)
        
        components = {
            'sigma_ewma': sigma_ewma,
            'sigma_mad': sigma_mad,
            'spike_proxy': spike_proxy,
            'blend_weight': w,
            'sigma_raw': sigma_raw
        }
        
        return sigma, components
    
    def _compute_mu(self, log_prices: np.ndarray) -> float:
        """
        Compute robust trend mu per SPEC:
        - Multi-window robust slope on log prices
        - Weighted combination
        
        Returns:
            mu (trend slope)
        """
        trend_cfg = self.theta['trend']
        windows = trend_cfg['windows']
        weights = trend_cfg['weights']
        method = trend_cfg['method']
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        mu_values = []
        for window in windows:
            if len(log_prices) < window + 1:
                window = len(log_prices) - 1
            
            x_window = log_prices[-window:]
            x_indices = np.arange(len(x_window))
            
            # Compute robust slope
            if method == 'theil_sen':
                mu_w = self._theil_sen_slope(x_indices, x_window)
            elif method == 'huber':
                mu_w = self._huber_slope(x_indices, x_window)
            else:  # ols fallback
                mu_w = self._ols_slope(x_indices, x_window)
            
            mu_values.append(mu_w)
        
        # Weighted combination
        mu = np.dot(weights[:len(mu_values)], mu_values)
        return mu
    
    def _theil_sen_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Theil-Sen slope estimator"""
        try:
            result = stats.theilslopes(y, x)
            return result.slope
        except Exception:
            return self._ols_slope(x, y)
    
    def _huber_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """Huber regression slope (simple IRLS)"""
        try:
            from scipy.optimize import minimize
            
            def huber_loss(params, x, y, delta=1.35):
                slope, intercept = params
                residuals = y - (slope * x + intercept)
                loss = np.where(
                    np.abs(residuals) <= delta,
                    0.5 * residuals ** 2,
                    delta * (np.abs(residuals) - 0.5 * delta)
                )
                return np.sum(loss)
            
            # Initial guess from OLS
            initial = self._ols_fit(x, y)
            result = minimize(huber_loss, initial, args=(x, y), method='Nelder-Mead')
            return result.x[0]
        except Exception:
            return self._ols_slope(x, y)
    
    def _ols_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """OLS slope"""
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]
    
    def _ols_fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """OLS fit returning [slope, intercept]"""
        return np.polyfit(x, y, 1)
    
    def _compute_dp(self, returns: np.ndarray) -> float:
        """
        Compute directional persistence (dp) per SPEC:
        dp = correlation of sign(r_t) with sign(r_{t-1}) over regime window
        
        Returns:
            dp in [0, 1] where 0.5 is neutral
        """
        window = min(self.theta['regime']['window'], len(returns))
        r_window = returns[-window:]
        
        if len(r_window) < 2:
            return 0.5
        
        signs = np.sign(r_window)
        # Count consecutive same signs
        same_sign = np.sum(signs[:-1] == signs[1:])
        dp = same_sign / (len(signs) - 1)
        
        return dp
    
    def _compute_vr(self, returns: np.ndarray) -> float:
        """
        Compute variance ratio (vr) per SPEC:
        VR(k) = Var(R_k) / (k * Var(R_1))
        where R_k is k-period return
        
        Returns:
            vr (variance ratio)
        """
        k = self.theta['regime']['vr_k']
        window = min(self.theta['regime']['window'], len(returns))
        r_window = returns[-window:]
        
        if len(r_window) < k + 1:
            return 1.0
        
        # 1-period variance
        var_1 = np.var(r_window)
        
        # k-period returns
        r_k = np.array([
            np.sum(r_window[i:i+k])
            for i in range(len(r_window) - k + 1)
        ])
        var_k = np.var(r_k)
        
        # Variance ratio
        vr = var_k / (k * var_1 + self.theta['eps'])
        
        return vr
    
    def _compute_regime_probs(self, ts: float, dp: float, vr: float) -> Dict[str, float]:
        """
        Compute regime probabilities per SPEC:
        - Score functions with tunable coefficients
        - Softmax over [s_trend, s_mr, s_chop]
        
        Returns:
            Dict with {'trend': p1, 'mr': p2, 'chop': p3} where sum=1
        """
        score_cfg = self.theta['regime']['score']
        temp = self.theta['regime']['softmax_temp']
        
        # Score functions
        s_trend = (
            score_cfg['a1'] * ts +
            score_cfg['a2'] * dp -
            score_cfg['a3'] * abs(vr - 1)
        )
        
        s_mr = (
            score_cfg['b1'] * abs(vr - 1) -
            score_cfg['b2'] * ts -
            score_cfg['b3'] * abs(dp)
        )
        
        s_chop = (
            score_cfg['c1'] * (1 - abs(dp)) +
            score_cfg['c2'] * (1 - min(ts, 1)) -
            score_cfg['c3'] * abs(vr - 1)
        )
        
        # Softmax
        logits = np.array([s_trend, s_mr, s_chop]) / temp
        probs = softmax(logits)
        
        return {
            'trend': float(probs[0]),
            'mr': float(probs[1]),
            'chop': float(probs[2])
        }
    
    def _log_state(self, symbol: str, state: Dict) -> None:
        """Optional rate-limited logging (no spam)"""
        # Simple rate limiting: log every ~100 calls per symbol
        if not hasattr(self, '_log_counter'):
            self._log_counter = {}
        
        self._log_counter[symbol] = self._log_counter.get(symbol, 0) + 1
        
        if self._log_counter[symbol] % 100 == 1:
            regime = max(state['regime_probs'].items(), key=lambda x: x[1])
            logger.info(
                f"MarketState {symbol}: "
                f"σ={state['sigma']:.6f} μ={state['mu']:.6f} TS={state['ts']:.4f} | "
                f"Regime={regime[0].upper()}({regime[1]:.1%}) | "
                f"dp={state['features']['dp']:.3f} vr={state['features']['vr']:.3f} "
                f"spike={state['features']['spike_proxy']:.3f}"
            )


if __name__ == "__main__":
    # Quick smoke test
    ms = MarketState()
    
    # Generate trending data
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.randn(300) * 0.01 + 0.001))
    
    state = ms.get_state("TEST", prices)
    if state:
        print(f"\n=== P0 MarketState TEST ===")
        print(f"Sigma: {state['sigma']:.6f}")
        print(f"Mu:    {state['mu']:.6f}")
        print(f"TS:    {state['ts']:.4f}")
        print(f"\nRegime Probabilities:")
        for regime, prob in state['regime_probs'].items():
            print(f"  {regime:5s}: {prob:.2%}")
        print(f"\nFeatures: dp={state['features']['dp']:.3f} "
              f"vr={state['features']['vr']:.3f} "
              f"spike_proxy={state['features']['spike_proxy']:.3f}")

