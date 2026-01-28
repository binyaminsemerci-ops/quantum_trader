#!/usr/bin/env python3
"""
Trade Path Tracer - Diagnose where trading signals die in the pipeline
Analyzes UTF stream to count BUY/SELL/HOLD per source (AI → Strategy → Risk → Execution)
"""
import os
import sys
import json
import subprocess
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, Any, List

LOG_FILE = '/var/log/quantum/utf_trade_path_report.log'
SAMPLE_SIZE = int(os.getenv('TRADE_PATH_SAMPLE_SIZE', '5000'))

def log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"{ts} | {msg}\n"
    print(line, end='', flush=True)
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(line)
    except Exception as e:
        print(f"Failed to write log: {e}", file=sys.stderr)

def get_utf_events(count: int = 5000) -> List[Dict[str, Any]]:
    """Fetch recent UTF events from Redis Stream"""
    try:
        result = subprocess.run(
            ['redis-cli', '--raw', 'XREVRANGE', 'quantum:stream:utf', '+', '-', 'COUNT', str(count)],
            capture_output=True, timeout=30, text=True
        )
        
        if result.returncode != 0:
            log(f"ERROR: Failed to fetch UTF stream: {result.stderr}")
            return []
        
        lines = result.stdout.strip().split('\n')
        events = []
        
        i = 0
        while i < len(lines):
            # Skip event ID
            if i + 1 < len(lines) and lines[i+1] == 'event':
                # Next line is the JSON event
                if i + 2 < len(lines):
                    try:
                        event = json.loads(lines[i+2])
                        events.append(event)
                        i += 3
                    except json.JSONDecodeError:
                        i += 1
            else:
                i += 1
        
        return events
    
    except Exception as e:
        log(f"ERROR: Exception fetching events: {e}")
        return []

def analyze_trade_path(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze trade path: where do signals get blocked?"""
    
    # Counters per source
    decisions_per_source = defaultdict(Counter)  # source -> {BUY: count, SELL: count, HOLD: count}
    total_per_source = Counter()  # source -> total_events
    errors_per_source = Counter()  # source -> error_count
    
    # Block reasons (extract from risk-brain)
    block_reasons = Counter()
    
    # Symbol activity per source
    symbols_per_source = defaultdict(set)  # source -> {symbols}
    
    for event in events:
        source = event.get('source', 'unknown')
        decision = event.get('decision')
        symbol = event.get('symbol')
        message = event.get('message', '')
        level = int(event.get('level', 6))
        
        total_per_source[source] += 1
        
        # Count decisions
        if decision:
            decisions_per_source[source][decision] += 1
        
        # Count errors
        if level <= 4 or 'ERROR' in message.upper():
            errors_per_source[source] += 1
        
        # Extract block reasons from risk-brain
        if source == 'risk' and any(keyword in message.lower() for keyword in ['deny', 'reject', 'block', 'fail', 'refuse']):
            # Try to extract reason
            if 'insufficient' in message.lower():
                block_reasons['insufficient_margin'] += 1
            elif 'leverage' in message.lower():
                block_reasons['leverage_limit'] += 1
            elif 'position' in message.lower():
                block_reasons['position_limit'] += 1
            elif 'confidence' in message.lower():
                block_reasons['low_confidence'] += 1
            elif 'volatility' in message.lower():
                block_reasons['high_volatility'] += 1
            elif 'drawdown' in message.lower():
                block_reasons['max_drawdown'] += 1
            elif 'cooldown' in message.lower():
                block_reasons['cooldown_active'] += 1
            elif 'symbol' in message.lower():
                block_reasons['symbol_restricted'] += 1
            else:
                block_reasons['other'] += 1
        
        # Track symbols
        if symbol:
            symbols_per_source[source].add(symbol)
    
    return {
        'decisions_per_source': dict(decisions_per_source),
        'total_per_source': dict(total_per_source),
        'errors_per_source': dict(errors_per_source),
        'block_reasons': dict(block_reasons),
        'symbols_per_source': {k: len(v) for k, v in symbols_per_source.items()}
    }

def generate_report(analysis: Dict[str, Any]):
    """Generate human-readable report"""
    
    log("=" * 80)
    log(f"TRADE PATH REPORT (last {SAMPLE_SIZE} UTF events)")
    log("=" * 80)
    
    # Event volume per source
    log("\n📊 EVENT VOLUME PER SOURCE:")
    total = analysis['total_per_source']
    for source in ['ai_engine', 'strategy', 'risk', 'execution']:
        count = total.get(source, 0)
        pct = (count / sum(total.values()) * 100) if sum(total.values()) > 0 else 0
        log(f"  {source:12} : {count:5} events ({pct:5.1f}%)")
    
    # Decision breakdown
    log("\n🎯 DECISIONS PER SOURCE:")
    decisions = analysis['decisions_per_source']
    for source in ['ai_engine', 'strategy', 'risk', 'execution']:
        if source in decisions:
            d = decisions[source]
            buy = d.get('BUY', 0)
            sell = d.get('SELL', 0)
            hold = d.get('HOLD', 0)
            total_decisions = buy + sell + hold
            
            if total_decisions > 0:
                log(f"  {source:12} : BUY={buy:4} ({buy/total_decisions*100:5.1f}%) | SELL={sell:4} ({sell/total_decisions*100:5.1f}%) | HOLD={hold:4} ({hold/total_decisions*100:5.1f}%)")
        else:
            log(f"  {source:12} : No decisions detected")
    
    # Error rates
    log("\n⚠️  ERROR RATES:")
    errors = analysis['errors_per_source']
    for source in ['ai_engine', 'strategy', 'risk', 'execution']:
        error_count = errors.get(source, 0)
        total_events = total.get(source, 0)
        error_rate = (error_count / total_events * 100) if total_events > 0 else 0
        log(f"  {source:12} : {error_count:4} errors ({error_rate:5.1f}% error rate)")
    
    # Block reasons
    if analysis['block_reasons']:
        log("\n🚫 BLOCK REASONS (from risk-brain):")
        for reason, count in sorted(analysis['block_reasons'].items(), key=lambda x: x[1], reverse=True):
            log(f"  {reason:25} : {count:4} times")
    
    # Symbol diversity
    log("\n📈 UNIQUE SYMBOLS PER SOURCE:")
    symbols = analysis['symbols_per_source']
    for source in ['ai_engine', 'strategy', 'risk', 'execution']:
        count = symbols.get(source, 0)
        log(f"  {source:12} : {count:3} unique symbols")
    
    # Drop-off analysis
    log("\n🔻 PIPELINE DROP-OFF ANALYSIS:")
    ai_decisions = sum((decisions.get('ai_engine', {}).get(d, 0) for d in ['BUY', 'SELL']))
    strategy_decisions = sum((decisions.get('strategy', {}).get(d, 0) for d in ['BUY', 'SELL']))
    risk_decisions = sum((decisions.get('risk', {}).get(d, 0) for d in ['BUY', 'SELL']))
    execution_decisions = sum((decisions.get('execution', {}).get(d, 0) for d in ['BUY', 'SELL']))
    
    log(f"  AI Engine  → Strategy  : {ai_decisions:4} → {strategy_decisions:4} ({(strategy_decisions/ai_decisions*100) if ai_decisions > 0 else 0:5.1f}% pass-through)")
    log(f"  Strategy   → Risk      : {strategy_decisions:4} → {risk_decisions:4} ({(risk_decisions/strategy_decisions*100) if strategy_decisions > 0 else 0:5.1f}% pass-through)")
    log(f"  Risk       → Execution : {risk_decisions:4} → {execution_decisions:4} ({(execution_decisions/risk_decisions*100) if risk_decisions > 0 else 0:5.1f}% pass-through)")
    
    # Diagnosis
    log("\n💡 DIAGNOSIS:")
    if ai_decisions > 0 and execution_decisions == 0:
        log("  ⚠️  CRITICAL: AI Engine generating signals but ZERO reaching execution!")
        if strategy_decisions == 0:
            log("     → Block location: Between AI Engine and Strategy")
            log("     → Check: Strategy Brain service status, event routing")
        elif risk_decisions == 0:
            log("     → Block location: Between Strategy and Risk")
            log("     → Check: Risk Brain service status, risk filters")
        else:
            log("     → Block location: Between Risk and Execution")
            log("     → Check: Execution service status (likely crashed)")
    elif execution_decisions < ai_decisions * 0.1:
        log("  ⚠️  WARNING: <10% of AI signals reaching execution (high drop-off)")
        log("     → Check block reasons above for details")
    else:
        log("  ✅ OK: Signal flow appears healthy")
    
    log("\n" + "=" * 80)

def main():
    log("Trade Path Tracer starting...")
    
    # Fetch events
    log(f"Fetching last {SAMPLE_SIZE} UTF events...")
    events = get_utf_events(SAMPLE_SIZE)
    
    if not events:
        log("ERROR: No events fetched. Check UTF stream.")
        return
    
    log(f"Fetched {len(events)} events")
    
    # Analyze
    analysis = analyze_trade_path(events)
    
    # Generate report
    generate_report(analysis)
    
    log("Trade Path Tracer complete")

if __name__ == '__main__':
    main()
