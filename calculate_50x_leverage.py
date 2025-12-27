#!/usr/bin/env python3
"""
Calculate impact of increasing leverage to 50x after AI is proven
"""
from datetime import datetime, timedelta

def main():
    print("\n" + "="*80)
    print(f"  🚀 50X LEVERAGE: NÅR AI ER MODENT OG SIKKER")
    print("="*80 + "\n")
    
    print("🎯 SCENARIO:")
    print("   ├─ AI har kjørt i 30+ dager")
    print("   ├─ 75-80% win rate established")
    print("   ├─ Proven track record")
    print("   ├─ Øker leverage fra 30x → 50x")
    print("   └─ Forventet høyere ROI per trade\n")
    
    print("="*80)
    print("  📊 SAMMENLIGNING: 30X VS 50X LEVERAGE")
    print("="*80 + "\n")
    
    initial = 10000
    
    # 30x leverage (current)
    daily_roi_30x = 0.04  # 4% per dag
    
    # 50x leverage (increased)
    # Med 50x leverage øker potential returns, men også risk
    # Realistisk: ~6-7% daglig ROI hvis AI er proven
    daily_roi_50x = 0.065  # 6.5% per dag
    
    print("   📈 30X LEVERAGE (NÅVÆRENDE):\n")
    print(f"   ├─ Daglig ROI:        4.0%")
    print(f"   ├─ Win rate:          75%")
    print(f"   ├─ Risk level:        MODERAT")
    print(f"   ├─ Position size:     Normal")
    print(f"   └─ Best for:          Learning phase\n")
    
    print("   🚀 50X LEVERAGE (OPPGRADERT):\n")
    print(f"   ├─ Daglig ROI:        6.5%")
    print(f"   ├─ Win rate:          75-80%")
    print(f"   ├─ Risk level:        HØY")
    print(f"   ├─ Position size:     67% større per trade")
    print(f"   └─ Best for:          Når AI proven (30+ dager)\n")
    
    print("="*80)
    print("  💰 $10,000 → $100,000 MED 50X LEVERAGE")
    print("="*80 + "\n")
    
    balance_50x = initial
    days_to_100k_50x = 0
    
    print(f"   Start: ${initial:,}")
    print(f"   Daglig ROI: 6.5%")
    print(f"   Leverage: 50x\n")
    print("   📊 MILEPÆLER:\n")
    
    milestones = [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
    start_date = datetime(2025, 12, 24)
    
    for milestone in milestones:
        while balance_50x < milestone:
            balance_50x *= (1 + daily_roi_50x)
            days_to_100k_50x += 1
        
        milestone_date = start_date + timedelta(days=days_to_100k_50x)
        print(f"   ${milestone:>6,} → Dag {days_to_100k_50x:>2} ({milestone_date.strftime('%d.%m.%Y')})")
    
    end_date_50x = milestone_date
    
    print(f"\n   🎯 TOTAL TID MED 50X: {days_to_100k_50x} dager")
    print(f"   📅 SLUTTDATO: {end_date_50x.strftime('%d. %B %Y')}\n")
    
    # Calculate 30x for comparison
    balance_30x = initial
    days_to_100k_30x = 0
    while balance_30x < 100000:
        balance_30x *= (1 + daily_roi_30x)
        days_to_100k_30x += 1
    
    time_saved = days_to_100k_30x - days_to_100k_50x
    
    print("="*80)
    print("  📊 30X VS 50X SAMMENLIGNING")
    print("="*80 + "\n")
    
    print("   ┌─────────────┬──────────┬───────────┬──────────────┬──────────┐")
    print("   │ Leverage    │ Daglig   │ Dager til │ Sluttdato    │ Profitt  │")
    print("   │             │ ROI      │ $100k     │              │          │")
    print("   ├─────────────┼──────────┼───────────┼──────────────┼──────────┤")
    print(f"   │ 30x         │ 4.0%     │ {days_to_100k_30x:>9} │ {(start_date + timedelta(days=days_to_100k_30x)).strftime('%d.%m.%Y'):>12} │ $90,000  │")
    print(f"   │ 50x 🚀      │ 6.5%     │ {days_to_100k_50x:>9} │ {end_date_50x.strftime('%d.%m.%Y'):>12} │ $90,000  │")
    print("   └─────────────┴──────────┴───────────┴──────────────┴──────────┘\n")
    
    print(f"   ⚡ TIDSBESPARELSE: {time_saved} dager raskere! ({(time_saved/days_to_100k_30x)*100:.1f}% raskere)\n")
    
    print("="*80)
    print("  🚀 DAG-FOR-DAG MED 50X LEVERAGE")
    print("="*80 + "\n")
    
    balance = initial
    print("   📈 FØRSTE 20 DAGER:\n")
    
    for day in range(1, 21):
        old_balance = balance
        balance *= (1 + daily_roi_50x)
        daily_gain = balance - old_balance
        date = start_date + timedelta(days=day-1)
        print(f"   Dag {day:>2} ({date.strftime('%d.%m')}): ${balance:>12,.2f} (+${daily_gain:>8,.2f})")
    
    print(f"\n   💰 ETTER 20 DAGER: ${balance:,.2f}")
    print(f"   📈 PROFITT: ${balance - initial:,.2f} (+{((balance-initial)/initial*100):.1f}%)\n")
    
    print("="*80)
    print("  ⚡ COMPOUND EFFECT MED 50X")
    print("="*80 + "\n")
    
    periods = [
        ("Uke 1", 7),
        ("Uke 2", 14),
        ("Uke 3", 21),
        ("Måned 1", 30),
        ("10x mål", days_to_100k_50x)
    ]
    
    print("   💡 EKSPONENTIELL VEKST:\n")
    
    for period_name, period_days in periods:
        period_balance = initial * (1 + daily_roi_50x) ** period_days
        period_profit = period_balance - initial
        period_roi = (period_profit / initial) * 100
        
        print(f"   {period_name:10} ({period_days:>2} dager):")
        print(f"   ├─ Balance:  ${period_balance:>12,.2f}")
        print(f"   ├─ Profitt:  ${period_profit:>12,.2f}")
        print(f"   └─ ROI:      +{period_roi:>6.1f}%\n")
    
    print("="*80)
    print("  ⚠️ ØKTE RISIKO MED 50X LEVERAGE")
    print("="*80 + "\n")
    
    print("   🔴 HØYERE RISIKO:\n")
    
    print("   1️⃣ STØRRE LIQUIDATION RISK:")
    print("      • Med 30x: Liquidation ved ~3.3% mot deg")
    print("      • Med 50x: Liquidation ved ~2.0% mot deg")
    print("      • Mindre margin for error")
    print("      • En 2% spike kan wipe out position\n")
    
    print("   2️⃣ STØRRE TAP PER LOSING TRADE:")
    print("      • Med 30x: -$200 på en loss")
    print("      • Med 50x: -$333 på en loss")
    print("      • 67% større losses")
    print("      • Krever høyere win rate for profitt\n")
    
    print("   3️⃣ ØKTE MARGIN CALLS:")
    print("      • Mindre buffer før forced liquidation")
    print("      • Volatilitet kan trigger liquidations")
    print("      • Trenger større account balance som buffer")
    print("      • Exchange kan øke margin requirements\n")
    
    print("   4️⃣ PSYKOLOGISK STRESS:")
    print("      • Se $10k → $7k på EN trade")
    print("      • Høyere swings = mer emotions")
    print("      • Fristende å panic-close")
    print("      • Krever sterkere mental game\n")
    
    print("="*80)
    print("  ✅ NÅR ER 50X LEVERAGE TRYGT?")
    print("="*80 + "\n")
    
    print("   📋 KRAV FØR DU ØKER TIL 50X:\n")
    
    print("   1️⃣ PROVEN TRACK RECORD:")
    print("      ✅ Minimum 30 dagers trading på 30x")
    print("      ✅ 75%+ win rate konsistent")
    print("      ✅ Profitable hver uke siste 3 uker")
    print("      ✅ Max drawdown < 15%\n")
    
    print("   2️⃣ STØRRE ACCOUNT BALANCE:")
    print("      ✅ Minimum $10,000 (ikke $1,000)")
    print("      ✅ Bedre buffer mot margin calls")
    print("      ✅ Kan absorbere større swings")
    print("      ✅ Position sizing mer flexibel\n")
    
    print("   3️⃣ OPTIMAL AI PERFORMANCE:")
    print("      ✅ AI-modeller fullstendig trent")
    print("      ✅ 80%+ confidence på predictions")
    print("      ✅ Dynamic TP/SL fungerer perfekt")
    print("      ✅ Self-Healing har prevented alle anomalies\n")
    
    print("   4️⃣ STABLE MARKET CONDITIONS:")
    print("      ✅ Ikke under extreme volatilitet")
    print("      ✅ Ikke under major news events")
    print("      ✅ High liquidity periods")
    print("      ✅ Normal trading volume\n")
    
    print("="*80)
    print("  💡 OPTIMAL STRATEGI: GRADVIS ØKNING")
    print("="*80 + "\n")
    
    print("   🎯 SMART LEVERAGE ESCALATION:\n")
    
    phases = [
        ("Fase 1", "Dag 1-30", "20-30x", "Learning & proving"),
        ("Fase 2", "Dag 31-60", "30-35x", "Small increase, test waters"),
        ("Fase 3", "Dag 61-90", "35-40x", "Gradual increase"),
        ("Fase 4", "Dag 91+", "40-50x", "Full power (if proven)")
    ]
    
    print("   ┌────────┬─────────────┬──────────┬───────────────────────┐")
    print("   │ Fase   │ Tidsperiode │ Leverage │ Status                │")
    print("   ├────────┼─────────────┼──────────┼───────────────────────┤")
    for phase, period, leverage, status in phases:
        print(f"   │ {phase:6} │ {period:11} │ {leverage:8} │ {status:21} │")
    print("   └────────┴─────────────┴──────────┴───────────────────────┘\n")
    
    print("   ✅ FORDELER MED GRADVIS ØKNING:")
    print("   ├─ Redusert risk ved hver økning")
    print("   ├─ Test AI performance på høyere leverage")
    print("   ├─ Lettere å justere ned hvis issues")
    print("   └─ Mer sustainable long-term\n")
    
    print("="*80)
    print("  💰 SAMMENLIGNING: ULIKE LEVERAGES")
    print("="*80 + "\n")
    
    leverage_scenarios = [
        (20, 0.03, "Konservativ & trygg"),
        (30, 0.04, "Balansert (anbefalt)"),
        (40, 0.055, "Aggressiv"),
        (50, 0.065, "Ekstrem høy risk"),
        (75, 0.08, "FARLIG - ikke anbefalt"),
        (100, 0.10, "EKSTREM FARLIG - unngå")
    ]
    
    print("   📊 DAGLIG ROI & DAGER TIL $100K:\n")
    print("   ┌──────────┬───────────┬────────────┬────────────────────┐")
    print("   │ Leverage │ Daglig    │ Dager til  │ Risk Level         │")
    print("   │          │ ROI       │ $100k      │                    │")
    print("   ├──────────┼───────────┼────────────┼────────────────────┤")
    
    for lev, roi, risk in leverage_scenarios:
        days = 0
        bal = 10000
        while bal < 100000:
            bal *= (1 + roi)
            days += 1
        print(f"   │ {lev:>3}x     │ {roi*100:>5.1f}%    │ {days:>10} │ {risk:18} │")
    
    print("   └──────────┴───────────┴────────────┴────────────────────┘\n")
    
    print("="*80)
    print("  🎯 REALISTISK PLAN: $10K → $100K MED 50X")
    print("="*80 + "\n")
    
    print("   📅 TIDSLINJE:\n")
    
    # Hybrid approach: start 30x, increase to 50x
    balance = 10000
    day = 0
    
    print("   FASE 1: 30x LEVERAGE (Dag 1-30)")
    for _ in range(30):
        balance *= (1 + 0.04)
        day += 1
    
    phase1_date = start_date + timedelta(days=30)
    print(f"   └─ Dag 30 ({phase1_date.strftime('%d.%m')}): ${balance:,.2f}\n")
    
    print("   FASE 2: 50x LEVERAGE (Dag 31+)")
    while balance < 100000:
        balance *= (1 + 0.065)
        day += 1
    
    final_date = start_date + timedelta(days=day)
    print(f"   └─ Dag {day} ({final_date.strftime('%d.%m')}): ${balance:,.2f}\n")
    
    print(f"   🎯 HYBRID RESULTAT:")
    print(f"   ├─ Total tid: {day} dager ({day/7:.1f} uker)")
    print(f"   ├─ Start: $10,000")
    print(f"   ├─ Slutt: ${balance:,.2f}")
    print(f"   ├─ Profitt: ${balance - 10000:,.2f}")
    print(f"   └─ Sluttdato: {final_date.strftime('%d. %B %Y')}\n")
    
    print("="*80)
    print("  ⚠️ VIKTIG ADVARSEL")
    print("="*80 + "\n")
    
    print("   🚨 50X LEVERAGE ER EKSTREMT RISIKABELT:\n")
    
    print("   ❌ WORST CASE SCENARIOS:")
    print("   ├─ Én 2% spike mot deg = Liquidation")
    print("   ├─ Flash crash = Total loss mulig")
    print("   ├─ Exchange outage = Kan ikke stenge position")
    print("   ├─ News event = 5-10% swing i sekunder")
    print("   └─ AI bug = Kan åpne farlige positions\n")
    
    print("   💰 ANBEFALT APPROACH:")
    print("   ├─ Start med $1,000 på 30x (test)")
    print("   ├─ Øk til $10,000 når proven")
    print("   ├─ Hold 30x til $30-50k profitt")
    print("   ├─ Test 40x med en del av profits")
    print("   ├─ Hvis success → gradvis til 50x")
    print("   └─ Aldri ALL IN på 50x!\n")
    
    print("="*80)
    print("  ✅ KONKLUSJON")
    print("="*80 + "\n")
    
    print("   ❓ HVA SKJER MED 50X LEVERAGE?\n")
    
    print(f"   ✅ RESULTAT:\n")
    print(f"   ├─ $10,000 → $100,000 på {days_to_100k_50x} dager")
    print(f"   ├─ {time_saved} dager raskere enn 30x")
    print(f"   ├─ 6.5% daglig ROI (vs 4% på 30x)")
    print(f"   └─ Sluttdato: {end_date_50x.strftime('%d. %B %Y')}\n")
    
    print("   📊 SAMMENLIGNING:")
    print("   ├─ 30x leverage: 59 dager til $100k")
    print("   ├─ 50x leverage: 37 dager til $100k")
    print("   └─ Tidsbesparelse: 22 dager (37% raskere)\n")
    
    print("   ⚠️ MEN:")
    print("   ├─ 67% høyere risk per trade")
    print("   ├─ Liquidation ved 2% (vs 3.3% på 30x)")
    print("   ├─ Krever perfekt AI performance")
    print("   └─ Én mistake kan koste $3-5k\n")
    
    print("   🎯 ANBEFALING:")
    print("   ├─ 1. Bevise AI på 30x først (30+ dager)")
    print("   ├─ 2. Oppnå 75%+ win rate konsistent")
    print("   ├─ 3. Bygg account til $30-50k")
    print("   ├─ 4. Test 40x på 20% av account")
    print("   ├─ 5. Hvis success → gradvis øk til 50x")
    print("   └─ 6. Aldri full account på 50x!\n")
    
    print("   💡 BEST PRAKSIS:")
    print("   └─ Bruk 30x på 80% av account")
    print("      Bruk 50x på 20% av account (high confidence trades)")
    print("      Dette gir høyere returns med kontrollerbar risk! 🎯\n")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
