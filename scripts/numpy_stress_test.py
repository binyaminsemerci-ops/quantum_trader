#!/usr/bin/env python3
"""
NumPy/Math Intensive Stress Test - Tester matematiske beregninger under last
"""

import time
import numpy as np
import threading
import statistics
from datetime import datetime
from typing import List, Dict
import concurrent.futures
import math

class MathStressTest:
    def __init__(self):
        print("🔢 Initialiserer matematisk stress test...")
        self.results = []
        
    def generate_crypto_features(self, n_samples: int = 1000):
        """Generer crypto price data for beregninger"""
        # Realistic crypto price patterns
        base_price = np.random.uniform(30000, 70000, n_samples)
        volatility = np.random.exponential(0.02, n_samples)
        volume = np.random.exponential(1000000, n_samples)
        
        # Add price movements with trend
        price_changes = np.random.normal(0, volatility * base_price)
        prices = base_price + np.cumsum(price_changes)
        
        return {
            'prices': prices,
            'volumes': volume,
            'volatility': volatility,
            'returns': np.diff(prices) / prices[:-1]
        }
    
    def calculate_technical_indicators(self, data: Dict) -> Dict:
        """Beregn tekniske indikatorer - math-intensive"""
        prices = data['prices']
        volumes = data['volumes']
        
        # Moving averages (different windows)
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='valid')
        sma_50 = np.convolve(prices, np.ones(50)/50, mode='valid')
        
        # Exponential moving average
        alpha = 2 / (12 + 1)
        ema = []
        ema.append(prices[0])
        for i in range(1, len(prices)):
            ema.append(alpha * prices[i] + (1 - alpha) * ema[-1])
        ema = np.array(ema)
        
        # RSI calculation
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(14)/14, mode='valid')
        avg_losses = np.convolve(losses, np.ones(14)/14, mode='valid')
        
        rs = avg_gains / np.maximum(avg_losses, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = np.convolve(prices, np.ones(20)/20, mode='valid')
        rolling_std = []
        for i in range(19, len(prices)):
            rolling_std.append(np.std(prices[i-19:i+1]))
        rolling_std = np.array(rolling_std)
        
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        
        # MACD
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        macd_line = ema_12 - ema_26
        signal_line = self._calculate_ema(macd_line, 9)
        
        # Volume indicators
        obv = np.cumsum(np.where(np.diff(prices) > 0, volumes[1:], -volumes[1:]))
        
        return {
            'sma_20': sma_20,
            'sma_50': sma_50,
            'ema': ema,
            'rsi': rsi,
            'bollinger_upper': upper_band,
            'bollinger_lower': lower_band,
            'macd': macd_line,
            'signal': signal_line,
            'obv': obv
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def monte_carlo_simulation(self, n_simulations: int = 1000) -> Dict:
        """Monte Carlo simulering for portfolio risk"""
        # Portfolio parameters
        initial_value = 100000
        n_days = 252  # Trading days in a year
        
        # Asset parameters (BTC, ETH, etc.)
        expected_returns = np.array([0.0008, 0.0012, 0.0005])  # Daily
        volatilities = np.array([0.04, 0.05, 0.03])           # Daily
        weights = np.array([0.5, 0.3, 0.2])
        
        # Correlation matrix
        correlation_matrix = np.array([
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.6],
            [0.5, 0.6, 1.0]
        ])
        
        # Cholesky decomposition for correlated random numbers
        chol_matrix = np.linalg.cholesky(correlation_matrix)
        
        portfolio_values = []
        
        for _ in range(n_simulations):
            # Generate correlated random returns
            random_returns = np.random.normal(0, 1, (n_days, 3))
            correlated_returns = random_returns @ chol_matrix.T
            
            # Scale by expected returns and volatilities
            daily_returns = (expected_returns + 
                           correlated_returns * volatilities)
            
            # Portfolio returns
            portfolio_returns = np.sum(daily_returns * weights, axis=1)
            
            # Calculate portfolio value evolution
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            final_value = initial_value * cumulative_returns[-1]
            
            portfolio_values.append(final_value)
        
        portfolio_values = np.array(portfolio_values)
        
        return {
            'mean_value': np.mean(portfolio_values),
            'std_value': np.std(portfolio_values),
            'var_95': np.percentile(portfolio_values, 5),   # 95% VaR
            'var_99': np.percentile(portfolio_values, 1),   # 99% VaR
            'max_value': np.max(portfolio_values),
            'min_value': np.min(portfolio_values)
        }
    
    def option_pricing_black_scholes(self, n_calculations: int = 1000) -> Dict:
        """Black-Scholes option pricing - math intensive"""
        results = []
        
        for _ in range(n_calculations):
            # Random option parameters
            S = np.random.uniform(30000, 70000)  # Current price
            K = S * np.random.uniform(0.8, 1.2)  # Strike price
            T = np.random.uniform(0.1, 1.0)      # Time to expiration
            r = np.random.uniform(0.01, 0.05)    # Risk-free rate
            sigma = np.random.uniform(0.2, 0.8)  # Volatility
            
            # Black-Scholes calculation
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Normal CDF approximation
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / np.sqrt(2)))
            
            # Call option price
            call_price = S * norm_cdf(d1) - K * np.exp(-r*T) * norm_cdf(d2)
            
            # Greeks
            delta = norm_cdf(d1)
            gamma = (1/np.sqrt(2*np.pi)) * np.exp(-0.5*d1**2) / (S*sigma*np.sqrt(T))
            theta = (-(S * (1/np.sqrt(2*np.pi)) * np.exp(-0.5*d1**2) * sigma) / (2*np.sqrt(T)) -
                    r*K*np.exp(-r*T)*norm_cdf(d2))
            
            results.append({
                'call_price': call_price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta
            })
        
        return {
            'avg_call_price': np.mean([r['call_price'] for r in results]),
            'avg_delta': np.mean([r['delta'] for r in results]),
            'avg_gamma': np.mean([r['gamma'] for r in results]),
            'avg_theta': np.mean([r['theta'] for r in results])
        }
    
    def math_worker(self, thread_id: int, duration_seconds: int):
        """Worker thread som kjører matematiske beregninger"""
        thread_start = time.time()
        thread_results = []
        
        while time.time() - thread_start < duration_seconds:
            # Velg tilfeldig matematisk operasjon
            operation = np.random.choice([
                'technical_indicators',
                'monte_carlo',
                'option_pricing',
                'matrix_operations'
            ], p=[0.4, 0.3, 0.2, 0.1])
            
            start_time = time.time()
            
            try:
                if operation == 'technical_indicators':
                    data = self.generate_crypto_features(2000)
                    result = self.calculate_technical_indicators(data)
                    operation_result = len(result)  # Number of indicators calculated
                    
                elif operation == 'monte_carlo':
                    result = self.monte_carlo_simulation(500)
                    operation_result = result['mean_value']
                    
                elif operation == 'option_pricing':
                    result = self.option_pricing_black_scholes(200)
                    operation_result = result['avg_call_price']
                    
                else:  # matrix_operations
                    # Large matrix multiplication and eigenvalue decomposition
                    A = np.random.randn(500, 500)
                    B = np.random.randn(500, 500)
                    C = A @ B
                    eigenvals, _ = np.linalg.eig(C)
                    operation_result = np.real(eigenvals[0])
                
                end_time = time.time()
                
                thread_results.append({
                    'thread_id': thread_id,
                    'operation': operation,
                    'latency_ms': (end_time - start_time) * 1000,
                    'result_value': operation_result,
                    'success': True,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                thread_results.append({
                    'thread_id': thread_id,
                    'operation': operation,
                    'latency_ms': 0,
                    'result_value': 0,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                })
            
            # Short pause
            time.sleep(np.random.uniform(0.001, 0.01))
        
        # Thread-safe append
        with threading.Lock():
            self.results.extend(thread_results)
    
    def run_math_stress_test(self, n_threads: int = 6, duration_seconds: int = 30):
        """Kjør matematisk stress test"""
        print(f"[ROCKET] Starter Math/NumPy Stress Test...")
        print(f"[CHART] Tråder: {n_threads} | Varighet: {duration_seconds}s")
        print(f"🔢 Operasjoner: Technical Indicators, Monte Carlo, Option Pricing, Matrix Ops")
        print("-" * 70)
        
        self.results = []
        threads = []
        
        # Start worker threads
        start_time = time.time()
        for i in range(n_threads):
            thread = threading.Thread(
                target=self.math_worker,
                args=(i, duration_seconds)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in self.results if r['success']]
        failed_results = [r for r in self.results if not r['success']]
        
        total_operations = len(self.results)
        success_rate = len(successful_results) / total_operations * 100 if total_operations > 0 else 0
        
        if successful_results:
            latencies = [r['latency_ms'] for r in successful_results]
            
            print(f"\n🔢 MATH STRESS TEST RESULTATER")
            print("=" * 70)
            print(f"Totale operasjoner:     {total_operations:,}")
            print(f"Suksessrate:           {success_rate:.2f}%")
            print(f"Operasjoner/sekund:    {total_operations/total_time:.1f}")
            print(f"Gjennomsnittlig latens: {np.mean(latencies):.1f}ms")
            print(f"Median latens:         {np.median(latencies):.1f}ms")
            print(f"95th percentil:        {np.percentile(latencies, 95):.1f}ms")
            print(f"99th percentil:        {np.percentile(latencies, 99):.1f}ms")
            print(f"Maks latens:           {np.max(latencies):.1f}ms")
            
            # Operation breakdown
            print(f"\n[CHART] OPERASJON BREAKDOWN:")
            operations = ['technical_indicators', 'monte_carlo', 'option_pricing', 'matrix_operations']
            for op in operations:
                op_results = [r for r in successful_results if r['operation'] == op]
                if op_results:
                    op_latencies = [r['latency_ms'] for r in op_results]
                    count = len(op_results)
                    avg_latency = np.mean(op_latencies)
                    
                    print(f"  {op:20} | {count:4d} ops | {avg_latency:6.1f}ms avg")
            
            # Performance grading
            avg_latency = np.mean(latencies)
            ops_rate = total_operations / total_time
            
            print(f"\n[TARGET] MATH/NUMPY PERFORMANCE VURDERING:")
            
            if (success_rate >= 99.5 and avg_latency <= 200 and ops_rate >= 10):
                grade = "A+"
                verdict = "Fantastisk - NumPy/Math operasjoner optimal under stress"
            elif (success_rate >= 99 and avg_latency <= 500 and ops_rate >= 5):
                grade = "A"
                verdict = "Utmerket - Robust matematisk performance"
            elif (success_rate >= 98 and avg_latency <= 1000 and ops_rate >= 3):
                grade = "B"
                verdict = "Bra - Akseptabel math performance"
            else:
                grade = "C"
                verdict = "Trenger optimalisering - Math performance issues"
            
            print(f"  Karakter: {grade}")
            print(f"  Vurdering: {verdict}")
            
            if failed_results:
                print(f"\n❌ FEIL: {len(failed_results)} failed operasjoner")
                for error_result in failed_results[:3]:
                    print(f"  - {error_result['operation']}: {error_result.get('error', 'Unknown error')}")
            
            return {
                'total_operations': total_operations,
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
                'operations_per_second': ops_rate,
                'grade': grade
            }
        else:
            print("❌ Ingen suksessfulle operasjoner!")
            return None

if __name__ == "__main__":
    print("🔢 NUMPY/MATH INTENSIVE STRESS TEST")
    print("=" * 70)
    print(f"Tidspunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Tester NumPy og matematiske beregninger under concurrent last...")
    print()
    
    # Initialize og kjør test
    tester = MathStressTest()
    results = tester.run_math_stress_test(n_threads=6, duration_seconds=30)
    
    if results:
        print(f"\n[OK] MATH STRESS TEST FULLFØRT")
        print(f"Math System presterte med karakter: {results['grade']}")
        print("Alle NumPy/Math komponenter validert under real last!")