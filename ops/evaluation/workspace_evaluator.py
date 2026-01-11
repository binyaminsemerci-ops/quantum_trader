#!/usr/bin/env python3
"""
Workspace Evaluator - Comprehensive Model and System Evaluation

Provides a holistic evaluation framework for the current workspace,
combining quality gates, model validation, and metrics aggregation.

Usage:
    ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode full
    ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --mode models-only
    ops/run.sh ai-engine ops/evaluation/workspace_evaluator.py --after 2026-01-10T05:43:15Z
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import redis
import numpy as np
from collections import defaultdict

# Import quality gate functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ops.model_safety.quality_gate import (
    read_redis_stream,
    extract_model_predictions,
    analyze_predictions,
    check_quality_gate,
    normalize_confidence,
    timestamp_to_stream_id
)


class WorkspaceEvaluator:
    """
    Comprehensive workspace evaluation orchestrator.
    
    Features:
    - Model quality validation (XGB, PatchTST, CatBoost, RandomForest)
    - Cutover-aware analysis (pre/post comparison)
    - Degeneracy detection
    - Ensemble health metrics
    - Event count validation
    - Comprehensive reporting
    """
    
    def __init__(
        self,
        stream_key: str = 'quantum:stream:trade.intent',
        min_events: int = 200,
        cutover_ts: Optional[str] = None
    ):
        self.stream_key = stream_key
        self.min_events = min_events
        self.cutover_ts = cutover_ts
        self.redis_client = self._connect_redis()
        
    def _connect_redis(self) -> redis.Redis:
        """Connect to Redis (fail-closed)"""
        try:
            r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
            r.ping()
            return r
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")
    
    def evaluate_workspace(self) -> Dict[str, Any]:
        """
        Perform comprehensive workspace evaluation.
        
        Returns:
            dict: Complete evaluation results including:
                - model_results: Per-model quality analysis
                - ensemble_health: Ensemble agreement metrics
                - event_metrics: Event count and coverage
                - quality_status: Overall PASS/FAIL status
                - recommendations: Action items
        """
        print("="*80)
        print("WORKSPACE EVALUATOR - Comprehensive Analysis")
        print("="*80)
        print()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'cutover_mode': bool(self.cutover_ts),
            'cutover_ts': self.cutover_ts,
            'status': 'UNKNOWN'
        }
        
        # Step 1: Validate event count
        print("Step 1: Event Count Validation")
        print("-" * 80)
        event_metrics = self._validate_events()
        results['event_metrics'] = event_metrics
        
        if event_metrics['sufficient']:
            print(f"✅ Sufficient events: {event_metrics['count']}/{self.min_events}")
        else:
            print(f"❌ Insufficient events: {event_metrics['count']}/{self.min_events}")
            results['status'] = 'FAIL_INSUFFICIENT_DATA'
            return results
        print()
        
        # Step 2: Model quality analysis
        print("Step 2: Per-Model Quality Analysis")
        print("-" * 80)
        model_results = self._analyze_models(event_metrics['events'])
        results['model_results'] = model_results
        
        # Step 3: Ensemble health
        print()
        print("Step 3: Ensemble Health Analysis")
        print("-" * 80)
        ensemble_health = self._analyze_ensemble_health(model_results)
        results['ensemble_health'] = ensemble_health
        
        # Step 4: Degeneracy detection
        print()
        print("Step 4: Degeneracy Detection")
        print("-" * 80)
        degeneracy_check = self._check_degeneracy(model_results)
        results['degeneracy_check'] = degeneracy_check
        
        # Step 5: Overall status determination
        print()
        print("Step 5: Overall Status Determination")
        print("-" * 80)
        overall_status = self._determine_status(model_results, ensemble_health, degeneracy_check)
        results['status'] = overall_status['status']
        results['blockers'] = overall_status['blockers']
        results['warnings'] = overall_status['warnings']
        results['recommendations'] = overall_status['recommendations']
        
        return results
    
    def _validate_events(self) -> Dict[str, Any]:
        """Validate event count and coverage"""
        events = read_redis_stream(
            self.stream_key,
            count=2000,
            after_ts=self.cutover_ts
        )
        
        return {
            'count': len(events),
            'sufficient': len(events) >= self.min_events,
            'min_required': self.min_events,
            'events': events
        }
    
    def _analyze_models(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze individual model quality"""
        model_data = extract_model_predictions(events)
        model_results = {}
        
        for model_name, model_info in model_data.items():
            predictions = model_info['predictions']
            analysis = analyze_predictions(predictions)
            failures = check_quality_gate(analysis)
            
            model_results[model_name] = {
                'analysis': analysis,
                'failures': failures,
                'status': 'PASS' if not failures else 'FAIL',
                'prediction_count': len(predictions)
            }
            
            status_icon = "✅" if not failures else "❌"
            print(f"{status_icon} {model_name}: {len(predictions)} predictions - {model_results[model_name]['status']}")
            if failures:
                for failure in failures[:3]:  # Show first 3
                    print(f"   - {failure}")
        
        return model_results
    
    def _analyze_ensemble_health(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze ensemble agreement and health"""
        active_models = [
            name for name, result in model_results.items()
            if result['status'] == 'PASS'
        ]
        
        if len(active_models) < 2:
            print("⚠️  Less than 2 passing models - ensemble degraded")
            return {
                'status': 'DEGRADED',
                'active_model_count': len(active_models),
                'agreement_metrics': None
            }
        
        # Calculate agreement metrics across active models
        agreement_data = self._calculate_agreement(model_results, active_models)
        
        agreement_pct = agreement_data.get('agreement_pct', 0)
        hard_disagree_pct = agreement_data.get('hard_disagree_pct', 0)
        
        # Determine health status
        if 55 <= agreement_pct <= 80 and hard_disagree_pct < 20:
            status = 'HEALTHY'
            icon = "✅"
        elif 40 <= agreement_pct <= 90 and hard_disagree_pct < 30:
            status = 'WARNING'
            icon = "⚠️ "
        else:
            status = 'UNHEALTHY'
            icon = "❌"
        
        print(f"{icon} Ensemble status: {status}")
        print(f"   Active models: {len(active_models)}")
        print(f"   Agreement: {agreement_pct:.1f}%")
        print(f"   Hard disagree: {hard_disagree_pct:.1f}%")
        
        return {
            'status': status,
            'active_model_count': len(active_models),
            'active_models': active_models,
            'agreement_metrics': agreement_data
        }
    
    def _calculate_agreement(
        self,
        model_results: Dict[str, Any],
        active_models: List[str]
    ) -> Dict[str, Any]:
        """Calculate ensemble agreement metrics"""
        if len(active_models) < 2:
            return {'agreement_pct': 0, 'hard_disagree_pct': 0}
        
        # Simplified agreement calculation based on action distributions
        action_distributions = {}
        for model_name in active_models:
            analysis = model_results[model_name]['analysis']
            if analysis:
                action_distributions[model_name] = analysis['action_pcts']
        
        if not action_distributions:
            return {'agreement_pct': 0, 'hard_disagree_pct': 0}
        
        # Calculate average divergence
        # Agreement = 100 - avg_divergence
        # This is a simplified metric; real implementation would compare predictions directly
        avg_buy = np.mean([d['BUY'] for d in action_distributions.values()])
        avg_sell = np.mean([d['SELL'] for d in action_distributions.values()])
        avg_hold = np.mean([d['HOLD'] for d in action_distributions.values()])
        
        # Calculate variance
        buy_var = np.var([d['BUY'] for d in action_distributions.values()])
        sell_var = np.var([d['SELL'] for d in action_distributions.values()])
        
        # Agreement heuristic: lower variance = higher agreement
        avg_variance = (buy_var + sell_var) / 2
        agreement_pct = max(0, 100 - (avg_variance / 2))  # Normalize
        
        # Hard disagree: significant BUY/SELL conflict
        hard_disagree_pct = min(avg_buy, avg_sell)  # Overlapping BUY/SELL
        
        return {
            'agreement_pct': agreement_pct,
            'hard_disagree_pct': hard_disagree_pct,
            'avg_action_dist': {'BUY': avg_buy, 'SELL': avg_sell, 'HOLD': avg_hold}
        }
    
    def _check_degeneracy(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for model degeneracy (constant outputs, stuck models).
        
        Degeneracy indicators:
        - Constant confidence (std < 0.01)
        - Single-action dominance (>95%)
        - HOLD collapse (>90%)
        - Confidence violations
        """
        degenerate_models = []
        
        for model_name, result in model_results.items():
            analysis = result.get('analysis')
            if not analysis:
                continue
            
            is_degenerate = False
            reasons = []
            
            # Check 1: Constant confidence
            conf_std = analysis['confidence']['std']
            if conf_std < 0.01:
                is_degenerate = True
                reasons.append(f"Constant confidence (std={conf_std:.4f})")
            
            # Check 2: Single action dominance
            for action, pct in analysis['action_pcts'].items():
                if pct > 95:
                    is_degenerate = True
                    reasons.append(f"{action} dominance ({pct:.1f}%)")
            
            # Check 3: HOLD collapse
            if analysis['action_pcts']['HOLD'] > 90:
                is_degenerate = True
                reasons.append(f"HOLD collapse ({analysis['action_pcts']['HOLD']:.1f}%)")
            
            # Check 4: Confidence violations
            if analysis.get('confidence_violations'):
                is_degenerate = True
                reasons.append(f"{len(analysis['confidence_violations'])} confidence violations")
            
            if is_degenerate:
                degenerate_models.append({
                    'model': model_name,
                    'reasons': reasons
                })
                print(f"❌ DEGENERATE: {model_name}")
                for reason in reasons:
                    print(f"   - {reason}")
        
        if not degenerate_models:
            print("✅ No degeneracy detected")
        
        return {
            'has_degeneracy': len(degenerate_models) > 0,
            'degenerate_models': degenerate_models,
            'count': len(degenerate_models)
        }
    
    def _determine_status(
        self,
        model_results: Dict[str, Any],
        ensemble_health: Dict[str, Any],
        degeneracy_check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine overall workspace status"""
        blockers = []
        warnings = []
        recommendations = []
        
        # Check for blockers
        failing_models = [
            name for name, result in model_results.items()
            if result['status'] == 'FAIL'
        ]
        
        if failing_models:
            blockers.append(f"{len(failing_models)} model(s) failing quality gate: {', '.join(failing_models)}")
        
        if degeneracy_check['has_degeneracy']:
            blockers.append(f"{degeneracy_check['count']} degenerate model(s) detected")
        
        if ensemble_health['status'] == 'UNHEALTHY':
            blockers.append("Ensemble health is UNHEALTHY")
        
        # Check for warnings
        if ensemble_health['status'] == 'WARNING':
            warnings.append("Ensemble health is in WARNING state")
        
        if ensemble_health['active_model_count'] < 3:
            warnings.append(f"Only {ensemble_health['active_model_count']} active models (recommended: ≥3)")
        
        # Generate recommendations
        if blockers:
            recommendations.append("DO NOT activate or deploy - blockers present")
            recommendations.append("Review failing models and degeneracy issues")
            recommendations.append("Run model diagnostics and retraining if needed")
        elif warnings:
            recommendations.append("WAIT - monitor warnings before activation")
            recommendations.append("Consider increasing ensemble diversity")
        else:
            recommendations.append("SAFE TO PROCEED - all checks passed")
            recommendations.append("Monitor ensemble agreement during canary deployment")
        
        # Determine final status
        if blockers:
            status = 'FAIL_BLOCKERS'
            icon = "❌"
        elif warnings:
            status = 'PASS_WITH_WARNINGS'
            icon = "⚠️ "
        else:
            status = 'PASS'
            icon = "✅"
        
        print(f"{icon} Overall Status: {status}")
        if blockers:
            print("\nBlockers:")
            for blocker in blockers:
                print(f"  ❌ {blocker}")
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"  → {rec}")
        
        return {
            'status': status,
            'blockers': blockers,
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def generate_report(self, results: Dict[str, Any], output_path: Path):
        """Generate comprehensive markdown report"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        
        lines = [
            "# Workspace Evaluation Report",
            "",
            f"**Generated:** {timestamp}",
            f"**Evaluator Version:** 1.0.0",
            ""
        ]
        
        # Executive summary
        status = results['status']
        status_icons = {
            'PASS': '✅',
            'PASS_WITH_WARNINGS': '⚠️ ',
            'FAIL_BLOCKERS': '❌',
            'FAIL_INSUFFICIENT_DATA': '❌'
        }
        icon = status_icons.get(status, '❓')
        
        lines.extend([
            "## Executive Summary",
            "",
            f"**Overall Status:** {icon} {status}",
            ""
        ])
        
        if results.get('cutover_mode'):
            lines.append(f"**Mode:** Post-cutover analysis (after {results['cutover_ts']})")
            lines.append("")
        
        # Blockers
        if results.get('blockers'):
            lines.append("### 🚫 Blockers")
            lines.append("")
            for blocker in results['blockers']:
                lines.append(f"- {blocker}")
            lines.append("")
        
        # Warnings
        if results.get('warnings'):
            lines.append("### ⚠️  Warnings")
            lines.append("")
            for warning in results['warnings']:
                lines.append(f"- {warning}")
            lines.append("")
        
        # Recommendations
        if results.get('recommendations'):
            lines.append("### 📋 Recommendations")
            lines.append("")
            for rec in results['recommendations']:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Event metrics
        event_metrics = results.get('event_metrics', {})
        lines.extend([
            "## Event Metrics",
            "",
            f"- **Events analyzed:** {event_metrics.get('count', 0)}",
            f"- **Minimum required:** {event_metrics.get('min_required', 0)}",
            f"- **Status:** {'✅ Sufficient' if event_metrics.get('sufficient') else '❌ Insufficient'}",
            ""
        ])
        
        # Model results
        model_results = results.get('model_results', {})
        if model_results:
            lines.extend([
                "## Per-Model Analysis",
                "",
                "| Model | Status | Predictions | Failures |",
                "|-------|--------|-------------|----------|"
            ])
            
            for model_name in sorted(model_results.keys()):
                result = model_results[model_name]
                status = result['status']
                status_icon = "✅" if status == 'PASS' else "❌"
                pred_count = result['prediction_count']
                failure_count = len(result['failures'])
                
                lines.append(f"| {model_name} | {status_icon} {status} | {pred_count} | {failure_count} |")
            
            lines.append("")
            
            # Detailed model breakdown
            for model_name in sorted(model_results.keys()):
                result = model_results[model_name]
                analysis = result.get('analysis')
                
                status = result['status']
                status_icon = "✅" if status == 'PASS' else "❌"
                
                lines.append(f"### {model_name} {status_icon}")
                lines.append("")
                
                if analysis:
                    lines.append("**Action Distribution:**")
                    for action in ['BUY', 'SELL', 'HOLD']:
                        pct = analysis['action_pcts'][action]
                        count = analysis['action_counts'][action]
                        lines.append(f"- {action}: {pct:.1f}% ({count})")
                    
                    lines.append("")
                    lines.append("**Confidence Stats:**")
                    conf = analysis['confidence']
                    lines.append(f"- Mean: {conf['mean']:.4f}")
                    lines.append(f"- Std: {conf['std']:.4f}")
                    lines.append(f"- P10-P90 Range: {conf['p10_p90_range']:.4f}")
                    
                    if result['failures']:
                        lines.append("")
                        lines.append("**Quality Gate Failures:**")
                        for failure in result['failures']:
                            lines.append(f"- {failure}")
                    
                    lines.append("")
        
        # Ensemble health
        ensemble_health = results.get('ensemble_health', {})
        if ensemble_health:
            ens_status = ensemble_health['status']
            ens_icon = {'HEALTHY': '✅', 'WARNING': '⚠️ ', 'UNHEALTHY': '❌', 'DEGRADED': '❌'}.get(ens_status, '❓')
            
            lines.extend([
                f"## Ensemble Health {ens_icon}",
                "",
                f"**Status:** {ens_status}",
                f"**Active Models:** {ensemble_health.get('active_model_count', 0)}",
                ""
            ])
            
            if ensemble_health.get('agreement_metrics'):
                metrics = ensemble_health['agreement_metrics']
                lines.append(f"**Agreement:** {metrics.get('agreement_pct', 0):.1f}%")
                lines.append(f"**Hard Disagree:** {metrics.get('hard_disagree_pct', 0):.1f}%")
                lines.append("")
        
        # Degeneracy check
        degeneracy = results.get('degeneracy_check', {})
        if degeneracy:
            deg_icon = "❌" if degeneracy['has_degeneracy'] else "✅"
            lines.extend([
                f"## Degeneracy Check {deg_icon}",
                "",
                f"**Degenerate Models:** {degeneracy['count']}",
                ""
            ])
            
            if degeneracy['degenerate_models']:
                for deg_model in degeneracy['degenerate_models']:
                    lines.append(f"### {deg_model['model']} (DEGENERATE)")
                    for reason in deg_model['reasons']:
                        lines.append(f"- {reason}")
                    lines.append("")
        
        # Write report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('\n'.join(lines))
        print()
        print(f"📄 Report saved: {output_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Workspace Evaluator - Comprehensive Model and System Evaluation'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'models-only', 'ensemble-only'],
        help='Evaluation mode (default: full)'
    )
    parser.add_argument(
        '--after',
        type=str,
        default=None,
        help='Analyze only events after timestamp (ISO 8601 format)'
    )
    parser.add_argument(
        '--min-events',
        type=int,
        default=200,
        help='Minimum required events (default: 200)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output report path (default: reports/evaluation/workspace_eval_<timestamp>.md)'
    )
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = '_post_cutover' if args.after else ''
        output_path = Path(f'reports/evaluation/workspace_eval_{timestamp_str}{suffix}.md')
    
    # Create evaluator
    evaluator = WorkspaceEvaluator(
        min_events=args.min_events,
        cutover_ts=args.after
    )
    
    # Run evaluation
    try:
        results = evaluator.evaluate_workspace()
        
        # Generate report
        evaluator.generate_report(results, output_path)
        
        # Return exit code based on status
        status = results['status']
        if status == 'PASS':
            return 0
        elif status == 'PASS_WITH_WARNINGS':
            return 0  # Still pass, but with warnings logged
        else:
            return 2  # FAIL
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
