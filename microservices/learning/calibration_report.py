"""
Calibration Report Generator

Generates human-readable reports for calibration analysis.
Reports include:
- Confidence calibration tables
- Ensemble weight changes
- Safety check results
- Approval workflow instructions
"""
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from microservices.learning.cadence_policy import Trade
from microservices.learning.calibration_types import (
    Calibration,
    ValidationReport,
    ValidationSeverity
)

logger = logging.getLogger(__name__)


class CalibrationReportGenerator:
    """
    Generates comprehensive calibration reports for human review.
    
    Reports are saved as both:
    - Markdown (.md) for readability
    - JSON (.json) for programmatic access
    """
    
    def __init__(self, output_dir: str = "/tmp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(
        self,
        calibration: Calibration,
        trades: List[Trade],
        validation: ValidationReport
    ) -> str:
        """
        Generate complete calibration report.
        
        Args:
            calibration: Calibration configuration
            trades: Trade data used for calibration
            validation: Validation report
        
        Returns:
            Path to generated markdown report
        """
        logger.info(f"📝 Generating calibration report: {calibration.version}")
        
        # Build markdown report
        md = self._render_header(calibration, trades)
        md += self._render_confidence_section(calibration, trades)
        md += self._render_weights_section(calibration)
        md += self._render_validation_section(validation)
        md += self._render_approval_section(calibration, validation)
        md += self._render_deployment_instructions(calibration)
        
        # Save markdown
        md_path = self.output_dir / f"calibration_{calibration.version}.md"
        try:
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md)
        except UnicodeEncodeError:
            # Fallback: strip emojis for Windows compatibility
            md_ascii = md.encode('ascii', errors='ignore').decode('ascii')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_ascii)
        
        logger.info(f"  Report saved: {md_path}")
        
        # Save JSON for programmatic access
        json_path = self.output_dir / f"calibration_{calibration.version}.json"
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(calibration.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"  JSON saved: {json_path}")
        
        return str(md_path)
    
    def _render_header(self, cal: Calibration, trades: List[Trade]) -> str:
        """Render report header"""
        win_count = sum(1 for t in trades if t.outcome_label == "WIN")
        loss_count = sum(1 for t in trades if t.outcome_label == "LOSS")
        neutral_count = len(trades) - win_count - loss_count
        
        win_rate = (win_count / len(trades) * 100) if trades else 0
        
        # Time span
        if trades:
            first_trade = min(t.timestamp for t in trades)
            last_trade = max(t.timestamp for t in trades)
            span_days = (last_trade - first_trade).total_seconds() / 86400
        else:
            span_days = 0
        
        return f"""# 🎯 CALIBRATION REPORT

**Version**: `{cal.version}`  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Status**: {cal.metadata.validation_status.upper()}

## 📊 Data Summary

| Metric | Value |
|--------|-------|
| Total Trades | {len(trades)} |
| Time Span | {span_days:.1f} days |
| Win Rate | {win_rate:.1%} ({win_count} wins) |
| Loss Rate | {(loss_count/len(trades)*100):.1%} ({loss_count} losses) |
| Neutral | {neutral_count} |
| Risk Score | {cal.metadata.risk_score:.1%} |

---

"""
    
    def _render_confidence_section(self, cal: Calibration, trades: List[Trade]) -> str:
        """Render confidence calibration section"""
        conf = cal.confidence
        
        if not conf.enabled:
            return """## 1. 🎯 Confidence Calibration

**Status**: ❌ DISABLED (insufficient improvement)

Confidence calibration was not applied because the improvement was below the 5% threshold.

**MSE Before**: {:.4f}  
**MSE After**: {:.4f}  
**Improvement**: {:+.1f}%

---

""".format(conf.mse_before, conf.mse_after, conf.improvement_pct)
        
        md = f"""## 1. 🎯 Confidence Calibration

**Status**: ✅ ENABLED  
**Method**: {conf.method}  
**Improvement**: {conf.improvement_pct:+.1f}%

### Calibration Curve

This table shows how predicted confidence is mapped to calibrated (actual) win rate:

| Predicted Confidence | Calibrated (Actual Win Rate) | Adjustment |
|---------------------|------------------------------|------------|
"""
        
        # Add calibration table rows
        for pred_conf in sorted(conf.mapping.keys()):
            calibrated = conf.mapping[pred_conf]
            adjustment = calibrated - pred_conf
            adjustment_str = f"{adjustment:+.3f}"
            
            arrow = ""
            if adjustment > 0.05:
                arrow = " ⬆️"
            elif adjustment < -0.05:
                arrow = " ⬇️"
            
            md += f"| {pred_conf:.2f} | {calibrated:.3f} | {adjustment_str}{arrow} |\n"
        
        md += f"""
### Metrics

- **MSE Before**: {conf.mse_before:.4f}
- **MSE After**: {conf.mse_after:.4f}
- **Improvement**: {conf.improvement_pct:+.1f}%
- **Sample Size**: {conf.sample_size} trades

### Interpretation

"""
        
        if conf.improvement_pct > 15:
            md += "✅ **Excellent improvement** - confidence calibration will significantly improve decision quality.\n"
        elif conf.improvement_pct > 10:
            md += "✅ **Good improvement** - confidence calibration will noticeably improve decision quality.\n"
        elif conf.improvement_pct > 5:
            md += "✅ **Moderate improvement** - confidence calibration will help improve decision quality.\n"
        else:
            md += "⚠️  **Marginal improvement** - consider waiting for more data.\n"
        
        md += "\n---\n\n"
        
        return md
    
    def _render_weights_section(self, cal: Calibration) -> str:
        """Render ensemble weights section"""
        weights = cal.weights
        
        if not weights.enabled:
            return """## 2. ⚖️  Ensemble Weight Adjustments

**Status**: ❌ DISABLED (changes too small or insufficient data)

Ensemble weights were not adjusted.

---

"""
        
        md = f"""## 2. ⚖️  Ensemble Weight Adjustments

**Status**: ✅ ENABLED  
**Total Delta**: {weights.total_delta:.4f}

### Weight Changes

| Model | Before | After | Delta | Change % | Reason |
|-------|--------|-------|-------|----------|--------|
"""
        
        for change in weights.changes:
            delta_str = f"{change.delta:+.4f}"
            delta_pct_str = f"{change.delta_pct:+.1f}%"
            
            arrow = ""
            if change.delta > 0.01:
                arrow = " ⬆️"
            elif change.delta < -0.01:
                arrow = " ⬇️"
            
            md += (
                f"| {change.model:8s} | {change.before:.3f} | {change.after:.3f} | "
                f"{delta_str}{arrow} | {delta_pct_str} | {change.reason} |\n"
            )
        
        # Verification
        weight_sum = sum(weights.weights.values())
        
        md += f"""
### Verification

- **Sum of weights**: {weight_sum:.6f} (must be 1.0)
- **Total adjustment**: {weights.total_delta:.4f}
- **Sample size**: {weights.sample_size} trades

### Interpretation

"""
        
        # Find largest change
        largest_change = max(weights.changes, key=lambda c: abs(c.delta))
        
        if abs(largest_change.delta) > 0.05:
            md += f"⚠️  **Significant change** - {largest_change.model} weight changed by {largest_change.delta_pct:+.1f}%. Monitor closely after deployment.\n"
        elif weights.total_delta > 0.10:
            md += f"⚠️  **Multiple adjustments** - total weight change is {weights.total_delta:.4f}. Verify impact after deployment.\n"
        else:
            md += "✅ **Conservative adjustments** - weight changes are within safe bounds.\n"
        
        md += "\n---\n\n"
        
        return md
    
    def _render_validation_section(self, validation: ValidationReport) -> str:
        """Render validation results"""
        status_emoji = "✅" if validation.passed else "❌"
        
        md = f"""## 3. 🔍 Safety Validation

**Status**: {status_emoji} {'PASSED' if validation.passed else 'FAILED'}  
**Risk Score**: {validation.risk_score:.1%}

### Validation Checks

| Check | Status | Severity | Result |
|-------|--------|----------|--------|
"""
        
        for check in validation.checks:
            status = "✅ PASS" if check.passed else "❌ FAIL"
            severity_emoji = {
                ValidationSeverity.INFO: "ℹ️",
                ValidationSeverity.WARNING: "⚠️",
                ValidationSeverity.ERROR: "❌",
                ValidationSeverity.CRITICAL: "🚨"
            }.get(check.severity, "")
            
            md += (
                f"| {check.name} | {status} | {severity_emoji} {check.severity.value} | "
                f"{check.result} |\n"
            )
        
        # Errors and warnings
        if validation.errors:
            md += "\n### ❌ Errors\n\n"
            for error in validation.errors:
                md += f"- {error}\n"
        
        if validation.warnings:
            md += "\n### ⚠️  Warnings\n\n"
            for warning in validation.warnings:
                md += f"- {warning}\n"
        
        md += "\n---\n\n"
        
        return md
    
    def _render_approval_section(self, cal: Calibration, validation: ValidationReport) -> str:
        """Render approval requirements"""
        if not validation.passed:
            return """## 4. ⛔ APPROVAL NOT POSSIBLE

**This calibration FAILED validation and cannot be deployed.**

Please review the validation errors above and:
1. Wait for more data
2. Adjust calibration parameters
3. Re-run calibration analysis

---

"""
        
        risk_level = "LOW" if validation.risk_score < 0.3 else "MEDIUM" if validation.risk_score < 0.6 else "HIGH"
        risk_emoji = "🟢" if risk_level == "LOW" else "🟡" if risk_level == "MEDIUM" else "🔴"
        
        return f"""## 4. ✅ APPROVAL REQUIRED

**Risk Level**: {risk_emoji} {risk_level} ({validation.risk_score:.1%})

### Pre-Deployment Checklist

Before approving this calibration, verify:

- [ ] Confidence calibration improves MSE by at least 5%
- [ ] Ensemble weights sum to exactly 1.0
- [ ] No weight exceeds bounds (0.15 - 0.40)
- [ ] Total weight change is reasonable (< 0.15)
- [ ] Validation passed with no critical errors
- [ ] Trade data is sufficient (50+ trades, 3+ days)

### Review Recommendations

"""
        
        if validation.risk_score < 0.3:
            return f"""✅ **LOW RISK** - This calibration is safe to deploy.

All metrics indicate conservative, well-validated adjustments.

---

"""
        elif validation.risk_score < 0.6:
            return f"""⚠️  **MEDIUM RISK** - Review carefully before deploying.

Some metrics indicate larger-than-usual adjustments. Monitor closely after deployment.

**Recommended**: Deploy during low-volume trading hours.

---

"""
        else:
            return f"""🔴 **HIGH RISK** - Consider waiting for more data or reducing adjustments.

Multiple warnings or large adjustments detected. Deployment is possible but requires extra caution.

**Recommended**: 
- Wait for more trade data
- Deploy as A/B test first
- Have rollback plan ready

---

"""
    
    def _render_deployment_instructions(self, cal: Calibration) -> str:
        """Render deployment instructions"""
        return f"""## 5. 🚀 Deployment Instructions

### To Approve and Deploy

```bash
# Run this command to deploy calibration
python microservices/learning/calibration_cli.py approve {cal.version}
```

This will:
1. Backup current configuration
2. Deploy new calibration to `/home/qt/quantum_trader/config/calibration.json`
3. Signal AI Engine to reload config
4. Mark calibration as completed in Learning Cadence

### Monitoring (24-48 hours)

After deployment, monitor:

1. **Win Rate**: Should not decrease
2. **Confidence Alignment**: Check META-V2 logs for calibrated confidence
3. **Drawdown**: Should not increase
4. **HOLD Rate**: Should remain 50-70%
5. **Meta Statistics**: Check override rate stability

### To Rollback

If any issues arise:

```bash
# Immediate rollback to previous version
python microservices/learning/calibration_cli.py rollback
```

Rollback time: < 2 minutes

---

## 📚 Additional Information

**Calibration Config Path**: `/home/qt/quantum_trader/config/calibration.json`  
**Archive Path**: `/home/qt/quantum_trader/config/calibration_archive/`  
**Learning Cadence API**: `http://127.0.0.1:8003`

**Generated by**: Calibration-Only Learning System  
**Report Version**: 1.0.0
"""
