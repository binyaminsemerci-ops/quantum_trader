#!/usr/bin/env python3
"""
Real Scikit-Learn Stress Test - Tester faktiske ML modeller under last
"""

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import threading
import statistics
from datetime import datetime
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class RealMLStressTest:
    def __init__(self):
        print("🔄 Initialiserer ML modeller...")
        self.scaler = StandardScaler()
        self.models = {}
        self.results = []
        
        # Generer syntetisk crypto trading dataset
        self.X, self.y = self._generate_crypto_dataset(10000)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Tren modeller
        print("🏋️  Trener ML modeller...")
        
        self.models['RandomForest'] = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=1
        )
        self.models['RandomForest'].fit(X_train_scaled, y_train)
        
        self.models['LogisticRegression'] = LogisticRegression(
            random_state=42, max_iter=1000
        )
        self.models['LogisticRegression'].fit(X_train_scaled, y_train)
        
        self.models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=50, max_depth=5, random_state=42
        )
        self.models['GradientBoosting'].fit(X_train_scaled, y_train)
        
        # Lagre test data for inference
        self.X_test_scaled = X_test_scaled
        self.y_test = y_test
        
        print(f"[OK] {len(self.models)} modeller trent og klare!")
        
    def _generate_crypto_dataset(self, n_samples: int = 10000):
        """Generer realistisk crypto trading dataset"""
        np.random.seed(42)
        
        # Price features
        price = np.random.uniform(30000, 70000, n_samples)
        volume = np.random.exponential(1000000, n_samples)
        
        # Technical indicators
        rsi = np.random.uniform(0, 100, n_samples)
        macd = np.random.normal(0, 100, n_samples)
        sma_ratio = np.random.uniform(0.8, 1.2, n_samples)
        volatility = np.random.exponential(0.02, n_samples)
        
        # Market features
        market_cap = price * np.random.uniform(19e6, 21e6, n_samples)
        price_change_24h = np.random.normal(0, 0.05, n_samples)
        volume_change_24h = np.random.normal(0, 0.3, n_samples)
        
        # Sentiment features (mock)
        social_sentiment = np.random.uniform(-1, 1, n_samples)
        news_sentiment = np.random.uniform(-1, 1, n_samples)
        
        # Stack features
        X = np.column_stack([
            price, volume, rsi, macd, sma_ratio, volatility,
            market_cap, price_change_24h, volume_change_24h,
            social_sentiment, news_sentiment
        ])
        
        # Generate targets (BUY=1, SELL=0) based on complex rules
        buy_signals = (
            (rsi < 30) & (macd > 0) & (price_change_24h > -0.02) |
            (sma_ratio > 1.05) & (social_sentiment > 0.3) & (volatility < 0.03) |
            (volume_change_24h > 0.2) & (news_sentiment > 0.4)
        )
        
        y = buy_signals.astype(int)
        
        return X, y
    
    def inference_worker(self, thread_id: int, duration_seconds: int):
        """Worker som kjører ML inference kontinuerlig"""
        thread_start = time.time()
        thread_results = []
        
        while time.time() - thread_start < duration_seconds:
            # Velg random model og data batch
            model_name = np.random.choice(list(self.models.keys()))
            model = self.models[model_name]
            
            # Random batch size
            batch_size = np.random.randint(1, 50)
            indices = np.random.choice(len(self.X_test_scaled), batch_size, replace=False)
            X_batch = self.X_test_scaled[indices]
            y_true = self.y_test[indices]
            
            # Time inference
            start_time = time.time()
            
            try:
                # Predict probabilities
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_batch)
                    predictions = (proba[:, 1] > 0.5).astype(int)
                    confidence = np.max(proba, axis=1).mean()
                else:
                    predictions = model.predict(X_batch)
                    confidence = 0.8  # Mock confidence for models without proba
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Calculate accuracy for this batch
                accuracy = accuracy_score(y_true, predictions)
                
                result = {
                    'thread_id': thread_id,
                    'model': model_name,
                    'batch_size': batch_size,
                    'latency_ms': latency_ms,
                    'accuracy': accuracy,
                    'confidence': confidence,
                    'success': True,
                    'timestamp': time.time()
                }
                
            except Exception as e:
                result = {
                    'thread_id': thread_id,
                    'model': model_name,
                    'batch_size': batch_size,
                    'latency_ms': 0,
                    'accuracy': 0,
                    'confidence': 0,
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
            
            thread_results.append(result)
            
            # Short pause
            time.sleep(np.random.uniform(0.001, 0.01))
        
        # Thread-safe append
        with threading.Lock():
            self.results.extend(thread_results)
    
    def run_stress_test(self, n_threads: int = 8, duration_seconds: int = 30):
        """Kjør real ML stress test"""
        print(f"[ROCKET] Starter Real ML Stress Test...")
        print(f"[CHART] Tråder: {n_threads} | Varighet: {duration_seconds}s")
        print(f"🤖 Modeller: {list(self.models.keys())}")
        print("-" * 70)
        
        self.results = []
        threads = []
        
        # Start worker threads
        start_time = time.time()
        for i in range(n_threads):
            thread = threading.Thread(
                target=self.inference_worker,
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
        
        total_inferences = len(self.results)
        success_rate = len(successful_results) / total_inferences * 100 if total_inferences > 0 else 0
        
        if successful_results:
            latencies = [r['latency_ms'] for r in successful_results]
            accuracies = [r['accuracy'] for r in successful_results]
            confidences = [r['confidence'] for r in successful_results]
            batch_sizes = [r['batch_size'] for r in successful_results]
            
            print(f"\n🧠 REAL ML STRESS TEST RESULTATER")
            print("=" * 70)
            print(f"Totale ML inferences:   {total_inferences:,}")
            print(f"Suksessrate:           {success_rate:.2f}%")
            print(f"Inferences/sekund:     {total_inferences/total_time:.1f}")
            print(f"Gjennomsnittlig latens: {np.mean(latencies):.1f}ms")
            print(f"Median latens:         {np.median(latencies):.1f}ms")
            print(f"95th percentil:        {np.percentile(latencies, 95):.1f}ms")
            print(f"99th percentil:        {np.percentile(latencies, 99):.1f}ms")
            print(f"Maks latens:           {np.max(latencies):.1f}ms")
            print(f"Gjennomsnitt batch:    {np.mean(batch_sizes):.1f}")
            print(f"Gjennomsnitt accuracy: {np.mean(accuracies):.3f}")
            print(f"Gjennomsnitt confidence: {np.mean(confidences):.3f}")
            
            # Model breakdown
            print(f"\n🤖 MODEL PERFORMANCE BREAKDOWN:")
            for model_name in self.models.keys():
                model_results = [r for r in successful_results if r['model'] == model_name]
                if model_results:
                    model_latencies = [r['latency_ms'] for r in model_results]
                    model_accuracies = [r['accuracy'] for r in model_results]
                    count = len(model_results)
                    avg_latency = np.mean(model_latencies)
                    avg_accuracy = np.mean(model_accuracies)
                    
                    print(f"  {model_name:20} | {count:4d} runs | {avg_latency:6.1f}ms | acc: {avg_accuracy:.3f}")
            
            # Performance grading
            avg_latency = np.mean(latencies)
            inference_rate = total_inferences / total_time
            avg_accuracy = np.mean(accuracies)
            
            print(f"\n[TARGET] SCIKIT-LEARN PERFORMANCE VURDERING:")
            
            if (success_rate >= 99.5 and avg_latency <= 50 and 
                inference_rate >= 100 and avg_accuracy >= 0.7):
                grade = "A+"
                verdict = "Fantastisk - Scikit-learn håndterer høy last perfekt"
            elif (success_rate >= 99 and avg_latency <= 100 and 
                  inference_rate >= 50 and avg_accuracy >= 0.6):
                grade = "A"
                verdict = "Utmerket - Robust ML performance under stress"
            elif (success_rate >= 98 and avg_latency <= 200 and 
                  inference_rate >= 25 and avg_accuracy >= 0.5):
                grade = "B"
                verdict = "Bra - Akseptabel ML performance"
            else:
                grade = "C"
                verdict = "Trenger optimalisering - ML performance issues"
            
            print(f"  Karakter: {grade}")
            print(f"  Vurdering: {verdict}")
            
            if failed_results:
                print(f"\n❌ FEIL: {len(failed_results)} failed inferences")
                for error_result in failed_results[:3]:  # Show first 3 errors
                    print(f"  - {error_result['model']}: {error_result.get('error', 'Unknown error')}")
            
            return {
                'total_inferences': total_inferences,
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
                'inferences_per_second': inference_rate,
                'avg_accuracy': avg_accuracy,
                'grade': grade
            }
        else:
            print("❌ Ingen suksessfulle inferences!")
            return None

if __name__ == "__main__":
    print("🧠 REAL SCIKIT-LEARN STRESS TEST")
    print("=" * 70)
    print(f"Tidspunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Tester faktiske scikit-learn modeller under concurrent last...")
    print()
    
    # Initialize og kjør test
    tester = RealMLStressTest()
    results = tester.run_stress_test(n_threads=8, duration_seconds=30)
    
    if results:
        print(f"\n[OK] SCIKIT-LEARN STRESS TEST FULLFØRT")
        print(f"ML System presterte med karakter: {results['grade']}")
        print("Alle scikit-learn komponenter validert under real last!")