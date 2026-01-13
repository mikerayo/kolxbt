import sys
from pathlib import Path
# Add parent directory to path for imports within core
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Analyzer - Generates rankings, reports, and summaries from KOL data
"""

import json
import pandas as pd
from typing import Dict, List
from datetime import datetime

from core.config import REPORTS_DIR
from ml_models import MLPipeline
from feature_engineering import calculate_features_for_all_kols


class ReportGenerator:
    """
    Generates various reports from KOL analysis results
    """

    def __init__(self):
        self.reports_dir = REPORTS_DIR

    def generate_leaderboard(self, df: pd.DataFrame, top_n: int = 50) -> str:
        """
        Generate Diamond Hands leaderboard report

        Args:
            df: DataFrame with KOL features and ML scores
            top_n: Number of top KOLs to include

        Returns:
            Formatted report string
        """
        # Filter KOLs with trades
        df_active = df[df['total_trades'] > 0].copy()

        if df_active.empty:
            return "No KOLs with trading data found."

        # Sort by diamond hand score
        top_kols = df_active.nlargest(min(top_n, len(df_active)), 'diamond_hand_score')

        report = []
        report.append("=" * 70)
        report.append("💎 DIAMOND HANDS LEADERBOARD")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append("")
        report.append(f"🏆 TOP {len(top_kols)} CONSISTENT 3x+ TRADERS")
        report.append("=" * 70)
        report.append("")

        for i, (_, row) in enumerate(top_kols.iterrows(), 1):
            name = row.get('name', 'Unknown')
            twitter = row.get('twitter_username')
            score = row.get('diamond_hand_score', 0)
            three_x_count = int(row.get('three_x_plus_count', 0))
            total_trades = int(row.get('total_trades', 0))
            three_x_rate = row.get('three_x_plus_rate', 0)
            hold_time = row.get('avg_hold_time_hours', 0)
            avg_multiple = row.get('avg_multiple', 0)
            win_rate = row.get('win_rate', 0)

            # Format rank badge
            if i <= 3:
                badges = ["🥇", "🥈", "🥉"]
                rank_badge = badges[i - 1]
            else:
                rank_badge = f"{i:2d}."

            report.append(f"{rank_badge} {name}")
            if twitter:
                report.append(f"    (@{twitter})")
            report.append(f"    💎 Diamond Hand Score: {score:.1f}/100")
            report.append(f"    📊 3x+ Trades: {three_x_count} / {total_trades} ({three_x_rate:.1%})")
            report.append(f"    ⏱️  Avg Hold Time: {hold_time:.1f} hours")
            report.append(f"    💰 Avg Multiple: {avg_multiple:.2f}x")
            report.append(f"    🎯 Win Rate: {win_rate:.1%}")
            report.append("")

        return "\n".join(report)

    def generate_scalpers_alert(self, df: pd.DataFrame, limit: int = 20) -> str:
        """
        Generate scalpers alert report

        Args:
            df: DataFrame with KOL features
            limit: Max number of scalpers to show

        Returns:
            Formatted report string
        """
        # Filter scalpers
        scalpers = df[df['is_scalper'] == True].copy()

        if scalpers.empty:
            return "No scalpers detected."

        # Sort by number of trades
        scalpers = scalpers.nlargest(min(limit, len(scalpers)), 'total_trades')

        report = []
        report.append("=" * 70)
        report.append("⚠️  SCALPERS ALERT")
        report.append("=" * 70)
        report.append("KOLs with short hold times (<5 min) - may not be suitable for long-term copy trading")
        report.append("")
        report.append("=" * 70)
        report.append("")

        for _, row in scalpers.iterrows():
            name = row.get('name', 'Unknown')
            twitter = row.get('twitter_username')
            trades = int(row.get('total_trades', 0))
            hold_time = row.get('avg_hold_time_hours', 0) * 60  # Convert to minutes

            report.append(f"⚡ {name}")
            if twitter:
                report.append(f"    (@{twitter})")
            report.append(f"    Trades: {trades}")
            report.append(f"    Avg Hold: {hold_time:.1f} minutes")
            report.append("")

        return "\n".join(report)

    def generate_cluster_summary(self, df: pd.DataFrame) -> str:
        """
        Generate cluster summary report

        Args:
            df: DataFrame with cluster assignments

        Returns:
            Formatted report string
        """
        if 'cluster_name' not in df.columns:
            return "No cluster data available."

        report = []
        report.append("=" * 70)
        report.append("📊 TRADING STYLE CLUSTERS")
        report.append("=" * 70)
        report.append("")

        cluster_stats = df.groupby('cluster_name').agg({
            'diamond_hand_score': 'mean',
            'total_trades': 'sum',
            'three_x_plus_rate': 'mean',
            'avg_hold_time_hours': 'mean',
            'wallet_address': 'count'
        }).rename(columns={'wallet_address': 'count'})

        cluster_stats = cluster_stats.sort_values('diamond_hand_score', ascending=False)

        for cluster_name, row in cluster_stats.iterrows():
            report.append(f"📁 {cluster_name}")
            report.append(f"    KOLs: {int(row['count'])}")
            report.append(f"    Avg Diamond Score: {row['diamond_hand_score']:.1f}")
            report.append(f"    Avg 3x+ Rate: {row['three_x_plus_rate']:.1%}")
            report.append(f"    Avg Hold Time: {row['avg_hold_time_hours']:.1f}h")
            report.append("")

        return "\n".join(report)

    def generate_full_report(self, df: pd.DataFrame) -> str:
        """
        Generate comprehensive report with all sections

        Args:
            df: DataFrame with complete ML analysis

        Returns:
            Complete formatted report
        """
        sections = []

        # Title
        sections.append("=" * 70)
        sections.append("        KOL TRACKER ML - WEEKLY REPORT")
        sections.append("=" * 70)
        sections.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        sections.append("")

        # Summary statistics
        summary = {
            'total': len(df),
            'with_trades': len(df[df['total_trades'] > 0]),
            'diamond_hands': len(df[df['is_diamond_hand'] == True]),
            'scalpers': len(df[df['is_scalper'] == True]),
        }

        sections.append("📊 SUMMARY")
        sections.append("-" * 70)
        sections.append(f"Total KOLs Tracked: {summary['total']}")
        sections.append(f"KOLs with Trading Data: {summary['with_trades']}")
        sections.append(f"Diamond Hands Identified: {summary['diamond_hands']}")
        sections.append(f"Scalpers Identified: {summary['scalpers']}")
        sections.append("")

        # Leaderboard
        sections.append(self.generate_leaderboard(df, top_n=10))

        # Scalpers
        sections.append("")
        sections.append(self.generate_scalpers_alert(df, limit=10))

        # Clusters
        if 'cluster_name' in df.columns:
            sections.append("")
            sections.append(self.generate_cluster_summary(df))

        return "\n".join(sections)

    def save_report(self, report: str, filename: str = None) -> str:
        """
        Save report to file

        Args:
            report: Report string
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"kol_report_{timestamp}.txt"

        filepath = self.reports_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        return str(filepath)

    def export_json(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Export KOL data to JSON

        Args:
            df: DataFrame with KOL features
            filename: Output filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"kol_data_{timestamp}.json"

        filepath = self.reports_dir / filename

        # Convert to JSON-serializable format
        data = df.to_dict(orient='records')

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        return str(filepath)


def analyze_and_report() -> Dict[str, str]:
    """
    Complete analysis pipeline: calculate features, run ML, generate reports

    Returns:
        Dictionary with file paths to generated reports
    """
    print("=" * 70)
    print("KOL TRACKER ML - ANALYSIS & REPORTING")
    print("=" * 70)

    # Calculate features
    print("\n[*] Step 1: Calculating features for all KOLs...")
    df = calculate_features_for_all_kols()

    if df.empty or df['total_trades'].sum() == 0:
        print("\n[!] No trading data found.")
        print("[!] Make sure to run wallet_tracker first to collect trade data.")
        return {}

    # Run ML pipeline
    print("\n[*] Step 2: Running ML analysis...")
    pipeline = MLPipeline()
    df = pipeline.analyze(df)

    # Generate reports
    print("\n[*] Step 3: Generating reports...")
    generator = ReportGenerator()

    # Full report
    full_report = generator.generate_full_report(df)
    full_report_path = generator.save_report(full_report)
    print(f"[+] Full report saved: {full_report_path}")

    # Leaderboard only
    leaderboard = generator.generate_leaderboard(df, top_n=50)
    leaderboard_path = generator.save_report(leaderboard, "leaderboard.txt")
    print(f"[+] Leaderboard saved: {leaderboard_path}")

    # JSON export
    json_path = generator.export_json(df)
    print(f"[+] JSON data exported: {json_path}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    summary = pipeline.get_summary(df)
    print(f"  Total KOLs: {summary['total_kols']}")
    print(f"  KOLs with trades: {summary['kols_with_trades']}")
    print(f"  Diamond Hands: {summary['diamond_hands']}")
    print(f"  Scalpers: {summary['scalpers']}")

    # Top 10
    print("\n🏆 Top 10 Diamond Hands:")
    print("-" * 70)
    top_10 = df.nlargest(10, 'diamond_hand_score')
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['name']}: {row['diamond_hand_score']:.1f}/100 "
              f"({row['three_x_plus_rate']:.1%} 3x+, {row['avg_hold_time_hours']:.1f}h avg)")

    return {
        'full_report': full_report_path,
        'leaderboard': leaderboard_path,
        'json_export': json_path
    }


if __name__ == "__main__":
    # Run complete analysis
    result = analyze_and_report()

    if result:
        print("\n" + "=" * 70)
        print("📁 Generated Files:")
        print("=" * 70)
        for name, path in result.items():
            print(f"  {name}: {path}")
