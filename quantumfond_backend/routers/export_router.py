"""
Export Router - Report Generation
Provides JSON, CSV, and PDF export functionality
"""
from fastapi import APIRouter, Response
from db.connection import SessionLocal
from db.models.trade_journal import TradeJournal
import pandas as pd
import io
import json

router = APIRouter(prefix="/reports", tags=["Reports & Export"])

@router.get("/export/{fmt}")
def export_report(fmt: str):
    """Export trade journal data in multiple formats"""
    db = SessionLocal()
    try:
        rows = db.query(TradeJournal).all()
        
        if not rows:
            return {"error": "No data available"}
        
        # Convert to DataFrame
        data = [{
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "symbol": r.symbol,
            "direction": r.direction,
            "entry_price": r.entry_price,
            "exit_price": r.exit_price,
            "pnl": r.pnl,
            "tp": r.tp,
            "sl": r.sl,
            "confidence": r.confidence,
            "model": r.model,
            "exit_reason": r.exit_reason
        } for r in rows]
        
        df = pd.DataFrame(data)
        
        if fmt == "json":
            json_data = df.to_json(orient="records", date_format="iso")
            return Response(
                content=json_data,
                media_type="application/json",
                headers={"Content-Disposition": f"attachment; filename=quantumfond_trades.json"}
            )
        
        if fmt == "csv":
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            return Response(
                content=csv_bytes,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=quantumfond_trades.csv"}
            )
        
        if fmt == "pdf":
            # Simple HTML table for PDF
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>QuantumFond Trading Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #22c55e; }}
                    table {{ border-collapse: collapse; width: 100%; font-size: 10px; }}
                    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
                    th {{ background-color: #1f2937; color: white; }}
                    .positive {{ color: #22c55e; }}
                    .negative {{ color: #ef4444; }}
                </style>
            </head>
            <body>
                <h1>QuantumFond Trading Report</h1>
                <p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Trades: {len(df)}</p>
                {df.head(100).to_html(index=False, classes='table', escape=False)}
            </body>
            </html>
            """
            
            try:
                from weasyprint import HTML
                pdf_bytes = HTML(string=html_content).write_pdf()
                return Response(
                    content=pdf_bytes,
                    media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=quantumfond_report.pdf"}
                )
            except ImportError:
                return {"error": "PDF export requires weasyprint. Install with: pip install weasyprint"}
        
        return {"error": f"Unsupported format: {fmt}. Use json, csv, or pdf"}
    
    finally:
        db.close()

@router.get("/summary/json")
def export_summary_json():
    """Export performance summary as JSON"""
    from .performance_router_enhanced import compute_metrics
    
    db = SessionLocal()
    try:
        rows = db.query(TradeJournal).filter(TradeJournal.pnl.isnot(None)).all()
        
        if not rows:
            return {"metrics": {}, "trades": []}
        
        data = [{"timestamp": r.timestamp, "pnl": r.pnl, "symbol": r.symbol} for r in rows]
        df = pd.DataFrame(data).sort_values("timestamp")
        
        metrics, curve = compute_metrics(df)
        
        report = {
            "generated_at": pd.Timestamp.now().isoformat(),
            "metrics": metrics,
            "equity_curve": curve[:100],  # Limit for JSON size
            "trade_count": len(rows)
        }
        
        return report
    finally:
        db.close()
