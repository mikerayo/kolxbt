#!/usr/bin/env python3
"""
Backtesting Report Generator

Genera reportes automáticos de backtesting:
- Model validation metrics
- Strategy performance (Follow KOLs, Buy & Hold)
- Comparison vs benchmarks
- Top/Worst performers

Usage:
    python analytics/generate_backtesting_report.py

Output:
    data/backtesting_report.json
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.backtesting import StrategyBacktester
from core.model_validation import ModelValidator
from core.database import db, KOL, ClosedPosition


def generate_backtesting_report(
    top_n: int = 10,
    period_days: int = 90,
    output_path: str = "data/backtesting_report.json"
) -> dict:
    """
    Genera reporte completo de backtesting

    Args:
        top_n: Número de KOLs top a analizar
        period_days: Días de backtesting histórico
        output_path: Path donde guardar el reporte

    Returns:
        Dict con reporte completo
    """
    print("=" * 70)
    print("KOL TRACKER ML - BACKTESTING REPORT GENERATOR")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Period: Last {period_days} days")
    print(f"Top N KOLs: {top_n}")
    print("=" * 70)
    print()

    # Initialize components
    backtester = StrategyBacktester()
    validator = ModelValidator()

    report = {
        "generated_at": datetime.now().isoformat(),
        "period_days": period_days,
        "top_n_kols": top_n,
        "model_validation": None,
        "follow_kols_strategy": None,
        "buy_hold_comparison": None,
        "top_performers": [],
        "recommendations": ""
    }

    # 1. Model Validation
    print("[1/4] Validating model predictions...")
    try:
        model_metrics = validator.validate_predictions(
            start_date=datetime.now() - timedelta(days=period_days)
        )

        report["model_validation"] = {
            "accuracy": float(model_metrics.accuracy),
            "precision": float(model_metrics.precision),
            "recall": float(model_metrics.recall),
            "f1_score": float(model_metrics.f1_score),
            "roc_auc": float(model_metrics.roc_auc),
            "true_positives": model_metrics.true_positives,
            "true_negatives": model_metrics.true_negatives,
            "false_positives": model_metrics.false_positives,
            "false_negatives": model_metrics.false_negatives,
            "calibration_error": float(model_metrics.calibration_error),
            "classification_report": model_metrics.classification_report
        }

        print(f"  [OK] Accuracy: {model_metrics.accuracy:.1%}")
        print(f"  [OK] Precision: {model_metrics.precision:.1%}")
        print(f"  [OK] Recall: {model_metrics.recall:.1%}")
        print(f"  [OK] F1 Score: {model_metrics.f1_score:.2f}")
        print()

    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        print()

    # 2. Follow KOLs Strategy
    print(f"[2/4] Backtesting 'Follow Top {top_n} KOLs' strategy...")
    try:
        follow_results = backtester.backtest_follow_kols(
            top_n=top_n,
            start_date=datetime.now() - timedelta(days=period_days)
        )

        report["follow_kols_strategy"] = {
            "total_trades": follow_results.total_trades,
            "win_rate": float(follow_results.win_rate),
            "total_return": float(follow_results.total_return),
            "cagr": float(follow_results.cagr),
            "sharpe_ratio": float(follow_results.sharpe_ratio),
            "sortino_ratio": float(follow_results.sortino_ratio),
            "max_drawdown": float(follow_results.max_drawdown),
            "best_trade": float(follow_results.best_trade),
            "worst_trade": float(follow_results.worst_trade),
            "profit_factor": float(follow_results.profit_factor),
            "expectancy": float(follow_results.expectancy)
        }

        print(f"  [OK] Total Return: {follow_results.total_return:.1f}%")
        print(f"  [OK] Sharpe Ratio: {follow_results.sharpe_ratio:.2f}")
        print(f"  [OK] Max Drawdown: {follow_results.max_drawdown:.1f}%")
        print(f"  [OK] Win Rate: {follow_results.win_rate:.1%}")
        print()

    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        print()

    # 3. Buy & Hold Comparison
    print("[3/4] Backtesting Buy & Hold strategies...")
    try:
        buy_hold_results = backtester.backtest_buy_and_hold(
            top_n=top_n,
            start_date=datetime.now() - timedelta(days=period_days)
        )

        # Encontrar mejor período
        best_period = max(buy_hold_results.items(), key=lambda x: x[1].sharpe_ratio)

        report["buy_hold_comparison"] = {
            "best_hold_period_hours": best_period[0].total_seconds() / 3600,
            "total_return": float(best_period[1].total_return),
            "sharpe_ratio": float(best_period[1].sharpe_ratio),
            "max_drawdown": float(best_period[1].max_drawdown),
            "win_rate": float(best_period[1].win_rate)
        }

        print(f"  [OK] Best Hold Period: {best_period[0].total_seconds() / 3600:.1f}h")
        print(f"  [OK] Total Return: {best_period[1].total_return:.1f}%")
        print(f"  [OK] Sharpe Ratio: {best_period[1].sharpe_ratio:.2f}")
        print()

    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        print()

    # 4. Top Performers Analysis
    print("[4/4] Analyzing top performers...")
    try:
        session = db.get_session()

        # Obtener KOLs con mejor performance
        kols = session.query(KOL).all()

        kol_performance = []

        for kol in kols:
            # Obtener posiciones cerradas
            positions = session.query(ClosedPosition).filter(
                ClosedPosition.kol_id == kol.id
            ).all()

            if len(positions) >= 5:  # Mínimo 5 trades
                total_pnl = sum([p.pnl_sol for p in positions])
                win_rate = sum([1 for p in positions if p.is_profitable]) / len(positions)
                avg_multiple = sum([p.pnl_multiple for p in positions if p.pnl_multiple]) / len(positions)

                kol_performance.append({
                    "kol_id": kol.id,
                    "name": kol.name,
                    "wallet_address": kol.wallet_address,
                    "total_trades": len(positions),
                    "total_pnl_sol": float(total_pnl),
                    "win_rate": float(win_rate),
                    "avg_multiple": float(avg_multiple)
                })

        # Ordenar por PnL
        kol_performance.sort(key=lambda x: x['total_pnl_sol'], reverse=True)

        # Top 10
        report["top_performers"] = kol_performance[:10]

        print(f"  [OK] Analyzed {len(kol_performance)} KOLs with 5+ trades")
        print(f"  [OK] Top performer: {report['top_performers'][0]['name'] if report['top_performers'] else 'N/A'}")
        print()

    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        print()

    # 5. Generate Recommendations
    print("[[OK]] Generating recommendations...")
    report["recommendations"] = _generate_recommendations(report)

    # Save report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n[[OK]] Report saved to: {output_file}")
    print()
    print("=" * 70)
    print("RECOMMENDATIONS:")
    print("=" * 70)
    print(report["recommendations"])
    print("=" * 70)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    return report


def _generate_recommendations(report: dict) -> str:
    """Genera recomendaciones basadas en resultados del backtesting"""
    recommendations = []

    # Model validation
    if report.get("model_validation"):
        acc = report["model_validation"]["accuracy"]

        if acc >= 0.75:
            recommendations.append("✅ Model is highly accurate (>75%) - predictions are reliable")
        elif acc >= 0.65:
            recommendations.append("⚠️ Model has moderate accuracy (65-75%) - use predictions with caution")
        else:
            recommendations.append("❌ Model accuracy is low (<65%) - model needs retraining")

    # Follow KOLs strategy
    if report.get("follow_kols_strategy"):
        sharpe = report["follow_kols_strategy"]["sharpe_ratio"]
        total_return = report["follow_kols_strategy"]["total_return"]

        if sharpe > 1.5 and total_return > 100:
            recommendations.append("✅ Follow KOLs strategy shows excellent risk-adjusted returns")
            recommendations.append("   Recommendation: Increase position sizes when top KOLs trade")
        elif sharpe > 1.0 and total_return > 50:
            recommendations.append("✅ Follow KOLs strategy shows good returns")
            recommendations.append("   Recommendation: Continue following top KOLs")
        elif sharpe > 0.5:
            recommendations.append("⚠️ Follow KOLs strategy shows moderate returns")
            recommendations.append("   Recommendation: Reduce position sizes, consider buy & hold")
        else:
            recommendations.append("❌ Follow KOLs strategy underperforms")
            recommendations.append("   Recommendation: Do NOT follow KOLs, use buy & hold instead")

    # Drawdown analysis
    if report.get("follow_kols_strategy"):
        max_dd = abs(report["follow_kols_strategy"]["max_drawdown"])

        if max_dd < 20:
            recommendations.append("✅ Maximum drawdown is well controlled (<20%)")
        elif max_dd < 40:
            recommendations.append("⚠️ Maximum drawdown is moderate (20-40%)")
            recommendations.append("   Recommendation: Use stop-loss to limit downside")
        else:
            recommendations.append("❌ Maximum drawdown is high (>40%)")
            recommendations.append("   Recommendation: Reduce risk per trade significantly")

    # Comparison vs Buy & Hold
    if report.get("follow_kols_strategy") and report.get("buy_hold_comparison"):
        follow_return = report["follow_kols_strategy"]["total_return"]
        buy_hold_return = report["buy_hold_comparison"]["total_return"]

        if follow_return > buy_hold_return * 1.2:
            recommendations.append("✅ Follow KOLs significantly outperforms buy & hold (>20% alpha)")
            recommendations.append("   Recommendation: Active trading adds value vs passive holding")
        elif follow_return > buy_hold_return:
            recommendations.append("✅ Follow KOLs slightly outperforms buy & hold")
            recommendations.append("   Recommendation: Active trading provides slight edge")
        else:
            recommendations.append("❌ Buy & hold outperforms following KOLs")
            recommendations.append("   Recommendation: Hold positions longer, reduce trading frequency")

    if not recommendations:
        recommendations.append("⏳ Not enough data for recommendations yet")

    return "\n".join(recommendations)


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate backtesting report')

    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top KOLs to analyze (default: 10)'
    )

    parser.add_argument(
        '--period-days',
        type=int,
        default=90,
        help='Backtesting period in days (default: 90)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/backtesting_report.json',
        help='Output file path (default: data/backtesting_report.json)'
    )

    args = parser.parse_args()

    # Generate report
    generate_backtesting_report(
        top_n=args.top_n,
        period_days=args.period_days,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
