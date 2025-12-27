#!/usr/bin/env python3
"""
Show the 4 AI models that predict price movements
"""
from datetime import datetime

def main():
    print("\n" + "="*80)
    print(f"  🤖 DE 4 AI-MODELLENE SOM PREDIKERER PRISBEVEGLSER")
    print("="*80 + "\n")
    
    models = [
        {
            "name": "XGBoost (XGB)",
            "type": "Gradient Boosting",
            "emoji": "🎯",
            "role": "Primær prediksjonsmotor",
            "features": [
                "Analyserer 100+ tekniske indikatorer",
                "RSI, MACD, Bollinger Bands, EMA crossovers",
                "Volume patterns, momentum indicators",
                "Historical price patterns"
            ],
            "output": "BUY / SELL / HOLD + Confidence (0-100%)",
            "strength": "Ekstremt nøyaktig på trendidentifikasjon",
            "weakness": "Kan være sen på sudden reversals",
            "confidence_example": "XGB:SELL/0.96 = 96% sikker på SELL"
        },
        {
            "name": "LightGBM (LGBM)",
            "type": "Light Gradient Boosting",
            "emoji": "⚡",
            "role": "Rask prediksjonsmotor",
            "features": [
                "Samme features som XGBoost",
                "Men optimalisert for hastighet",
                "Leaf-wise tree growth",
                "Håndterer store datasett bedre"
            ],
            "output": "BUY / SELL / HOLD + Confidence (0-100%)",
            "strength": "Veldig rask, god på volume patterns",
            "weakness": "Kan overfit på små bevegelser",
            "confidence_example": "LGBM:SELL/0.89 = 89% sikker på SELL"
        },
        {
            "name": "N-HiTS",
            "type": "Neural Hierarchical Interpolation for Time Series",
            "emoji": "🧠",
            "role": "Time series forecasting",
            "features": [
                "Predikerer faktisk pris 12 steps fremover",
                "Multi-horizon forecasting",
                "Hierarkisk interpolering",
                "Fanger seasonality og cycles"
            ],
            "output": "Price prediction array + Confidence",
            "strength": "Ser langsiktige patterns og cycles",
            "weakness": "Krever mye data, kan feile på volatilitet",
            "confidence_example": "NH:SELL/0.60 = 60% sikker (fallback mode)",
            "fallback": "Bruker RSI + EMA hvis prediction feiler"
        },
        {
            "name": "PatchTST",
            "type": "Patch Time Series Transformer",
            "emoji": "🔮",
            "role": "Advanced pattern recognition",
            "features": [
                "Transformer-based arkitektur",
                "Self-attention på price patches",
                "Fanger komplekse patterns",
                "Multi-variate time series"
            ],
            "output": "Price prediction array + Confidence",
            "strength": "Ekstremt god på komplekse patterns",
            "weakness": "Ressurskrevende, krever mye data",
            "confidence_example": "PT:SELL/0.60 = 60% sikker (fallback mode)",
            "fallback": "Bruker RSI + EMA hvis prediction feiler"
        }
    ]
    
    for i, model in enumerate(models, 1):
        print(f"{'─'*80}")
        print(f"{model['emoji']} MODELL #{i}: {model['name']}")
        print(f"{'─'*80}\n")
        
        print(f"   📊 TYPE: {model['type']}")
        print(f"   🎯 ROLLE: {model['role']}\n")
        
        print(f"   🔍 HVA DEN ANALYSERER:")
        for feature in model['features']:
            print(f"      • {feature}")
        
        print(f"\n   📤 OUTPUT: {model['output']}")
        print(f"   ✅ STYRKE: {model['strength']}")
        print(f"   ⚠️ SVAKHET: {model['weakness']}")
        print(f"\n   📈 EKSEMPEL: {model['confidence_example']}")
        
        if 'fallback' in model:
            print(f"   🔄 FALLBACK: {model['fallback']}")
        
        print()
    
    print("="*80)
    print("  🎯 ENSEMBLE MANAGER - KOMBINERER ALLE 4 MODELLER")
    print("="*80 + "\n")
    
    print("   📊 HVORDAN ENSEMBLE FUNGERER:\n")
    print("   1️⃣ Alle 4 modeller analyserer samme coin samtidig")
    print("      → XGBoost: Analyserer tekniske indikatorer")
    print("      → LightGBM: Bekrefter med rask analyse")
    print("      → N-HiTS: Predikerer fremtidig pris")
    print("      → PatchTST: Finner komplekse patterns\n")
    
    print("   2️⃣ Hver modell gir sin predikasjon + confidence:")
    print("      → XGB: HOLD (96% confident)")
    print("      → LGBM: SELL (89% confident)")
    print("      → N-HiTS: SELL (60% confident)")
    print("      → PatchTST: SELL (60% confident)\n")
    
    print("   3️⃣ Ensemble Manager kombinerer:")
    print("      → Vekter basert på confidence")
    print("      → Beregner konsensus")
    print("      → Output: SELL 76.68%\n")
    
    print("   4️⃣ FINAL BESLUTNING:")
    print("      → Hvis ensemble >70% confident → TRADE")
    print("      → Hvis ensemble 50-70% → HOLD")
    print("      → Hvis ensemble <50% → SKIP\n")
    
    print("="*80)
    print("  📊 SANNTIDS EKSEMPEL FRA DINE POSISJONER")
    print("="*80 + "\n")
    
    # Real example from logs
    print("   🎯 ZECUSDT ANALYSE (15:52:27):\n")
    print("   ├─ XGBoost:   HOLD (confidence: 0.96) ✅")
    print("   ├─ LightGBM:  SELL (confidence: 0.89) 🔴")
    print("   ├─ N-HiTS:    SELL (confidence: 0.60) 🔴 [fallback mode]")
    print("   └─ PatchTST:  SELL (confidence: 0.60) 🔴 [fallback mode]\n")
    
    print("   📊 ENSEMBLE BESLUTNING:")
    print("   └─ Final: SELL 76.68% (3 av 4 sier SELL)\n")
    
    print("   🎯 DYNAMIC TP/SL BEREGNING:")
    print("   └─ Confidence 0.77 → TP=6.4% SL=6.9% Trail=2.0%\n")
    
    print("   ✅ RESULTAT:")
    print("   └─ SHORT posisjon åpnet på ZECUSDT ved $556.48\n")
    
    print("="*80)
    print("  🔄 KONTINUERLIG ANALYSE SYKLUS")
    print("="*80 + "\n")
    
    print("   ⏰ HVERT 5. MINUTT (00, 05, 10, 15, etc.):")
    print("   ├─ 1. Henter latest price data for alle coins")
    print("   ├─ 2. Beregner 100+ tekniske indikatorer")
    print("   ├─ 3. Alle 4 modeller analyserer samtidig")
    print("   ├─ 4. Ensemble Manager kombinerer resultater")
    print("   ├─ 5. Safety Governor validerer trades")
    print("   ├─ 6. Dynamic TP/SL beregner exit-nivåer")
    print("   └─ 7. Event-Driven Executor plasserer ordrer\n")
    
    print("   📊 ANALYSE PR COIN:")
    print("   ├─ Total tid: ~200-500ms per coin")
    print("   ├─ XGBoost: ~50ms")
    print("   ├─ LightGBM: ~30ms")
    print("   ├─ N-HiTS: ~100ms")
    print("   └─ PatchTST: ~100ms\n")
    
    print("   🎯 TOTAL ANALYSERT:")
    print("   └─ 3 coins × 4 modeller = 12 prediksjoner per syklus\n")
    
    print("="*80)
    print("  🧠 HVA MODELLENE SER PÅ FOR Å BESTEMME OPP/NED")
    print("="*80 + "\n")
    
    indicators = [
        ("📈 Trend Indicators", [
            "EMA (7, 25, 99) crossovers",
            "SMA (20, 50, 200)",
            "MACD line vs signal line",
            "ADX (trend strength)",
            "Parabolic SAR"
        ]),
        ("💪 Momentum Indicators", [
            "RSI (overbought/oversold)",
            "Stochastic oscillator",
            "ROC (Rate of Change)",
            "Williams %R",
            "CCI (Commodity Channel Index)"
        ]),
        ("📊 Volume Indicators", [
            "Volume moving averages",
            "OBV (On-Balance Volume)",
            "Volume-Price Trend",
            "Chaikin Money Flow",
            "Volume Rate of Change"
        ]),
        ("🎯 Volatility Indicators", [
            "Bollinger Bands (upper/lower)",
            "ATR (Average True Range)",
            "Keltner Channels",
            "Standard Deviation",
            "Historical Volatility"
        ]),
        ("🔮 Pattern Recognition", [
            "Support/Resistance levels",
            "Fibonacci retracements",
            "Price action patterns",
            "Candlestick patterns",
            "Historical correlations"
        ])
    ]
    
    for category, items in indicators:
        print(f"   {category}:")
        for item in items:
            print(f"      • {item}")
        print()
    
    print("="*80)
    print("  💡 BESLUTNINGSLOGIKK")
    print("="*80 + "\n")
    
    print("   🟢 BUY SIGNAL (opp):")
    print("      • RSI < 30 (oversold)")
    print("      • Price krysser over EMA 25")
    print("      • MACD crossover (bullish)")
    print("      • Volume øker")
    print("      • Bollinger Band bounce fra lower band")
    print("      → Ensemble: Hvis 3/4 modeller enige → BUY\n")
    
    print("   🔴 SELL SIGNAL (ned):")
    print("      • RSI > 70 (overbought)")
    print("      • Price krysser under EMA 25")
    print("      • MACD crossover (bearish)")
    print("      • Volume divergence")
    print("      • Price hits Bollinger upper band")
    print("      → Ensemble: Hvis 3/4 modeller enige → SELL\n")
    
    print("   ⚪ HOLD SIGNAL (usikker):")
    print("      • Mixed signals fra indikatorer")
    print("      • Low confidence (<70%)")
    print("      • Sideways market")
    print("      → Ensemble: Hvis modeller uenige → HOLD\n")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
