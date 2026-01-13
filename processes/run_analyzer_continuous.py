import sys
#!/usr/bin/env python3
"""
Continuous Analyzer - Analyzes KOL data continuously
Runs in background, analyzing new KOLs as they get enough data
"""

import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd

from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import db, KOL, Trade, ClosedPosition
from core.feature_engineering import KOLFeatures, PositionMatcher
from core.ml_models import DiamondHandScorer
from core.config import DIAMOND_HAND_WEIGHTS


class ContinuousAnalyzer:
    """
    Continuously analyzes KOL trading data
    """

    def __init__(self, check_interval: int = 300):
        """
        Initialize continuous analyzer

        Args:
            check_interval: Seconds between analysis runs (default: 5 min)
        """
        self.check_interval = check_interval
        self.analyzed_kols = set()
        self.scorer = DiamondHandScorer()

    def get_kols_with_enough_data(self, session, min_trades: int = 5) -> List[KOL]:
        """
        Get KOLs that have enough trades to analyze

        Args:
            session: Database session
            min_trades: Minimum number of trades required

        Returns:
            List of KOLs with enough data
        """
        # Get KOLs with at least min_trades trades
        from sqlalchemy import func

        subquery = session.query(
            Trade.kol_id,
            func.count(Trade.id).label('trade_count')
        ).group_by(
            Trade.kol_id
        ).having(
            func.count(Trade.id) >= min_trades
        ).subquery()

        kols = session.query(KOL).join(
            subquery, KOL.id == subquery.c.kol_id
        ).all()

        return kols

    def analyze_single_kol(self, kol: KOL, session) -> Dict[str, Any]:
        """
        Analyze a single KOL

        Args:
            kol: KOL object
            session: Database session

        Returns:
            Dictionary with analysis results
        """
        try:
            # Match positions first
            positions = PositionMatcher.match_positions(kol, session)

            if not positions:
                return None

            # Calculate features
            feature_calc = KOLFeatures(kol, session)
            features = feature_calc.calculate_all_features()

            # Calculate Diamond Hand Score
            score = self.scorer.calculate_score(features)
            features['diamond_hand_score'] = score

            return features

        except Exception as e:
            print(f"[!] Error analyzing {kol.name}: {e}")
            return None

    def generate_leaderboard(self, all_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate leaderboard from analysis results

        Args:
            all_results: List of analysis result dictionaries

        Returns:
            DataFrame with ranked KOLs
        """
        # Filter out None results
        valid_results = [r for r in all_results if r is not None]

        if not valid_results:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(valid_results)

        # Sort by Diamond Hand Score
        df = df.sort_values('diamond_hand_score', ascending=False)

        # Add rank
        df['rank'] = range(1, len(df) + 1)

        return df

    def print_leaderboard(self, df: pd.DataFrame, top_n: int = 20):
        """
        Print formatted leaderboard

        Args:
            df: Leaderboard DataFrame
            top_n: Number of KOLs to show
        """
        if df.empty:
            print("\n[!] No KOLs with enough data yet")
            return

        print("\n" + "="*80)
        print(f"DIAMOND HANDS LEADERBOARD - Top {min(top_n, len(df))} KOLs")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

        for _, row in df.head(top_n).iterrows():
            # Clean name for display
            name = row.get('name', 'Unknown')
            try:
                display_name = name.encode('ascii', 'ignore').decode('ascii')
            except:
                display_name = 'Unknown'

            print(f"\n#{int(row['rank'])} {display_name}")
            print(f"  Wallet: {row['wallet_address'][:8]}...{row['wallet_address'][-8:]}")
            print(f"  Diamond Hand Score: {row['diamond_hand_score']:.1f}/100")
            print(f"  Total Trades: {int(row['total_trades'])}")
            print(f"  3x+ Rate: {row['three_x_plus_rate']*100:.1f}% ({int(row['three_x_plus_count'])} trades)")
            print(f"  Win Rate: {row['win_rate']*100:.1f}%")
            print(f"  Avg Hold Time: {row['avg_hold_time_hours']:.1f} hours")
            print(f"  Total PnL: {row['total_pnl_sol']:.2f} SOL")

            if row.get('is_diamond_hand'):
                print(f"  [DIAMOND HAND]")

        print("\n" + "="*80)

        # Print summary stats
        diamond_hands = df[df['is_diamond_hand'] == True]
        scalpers = df[df['is_scalper'] == True]

        print(f"\nSUMMARY:")
        print(f"  Total KOLs analyzed: {len(df)}")
        print(f"  Diamond Hands: {len(diamond_hands)}")
        print(f"  Scalpers: {len(scalpers)}")
        print(f"  Avg 3x+ Rate: {df['three_x_plus_rate'].mean()*100:.1f}%")
        print(f"  Avg Win Rate: {df['win_rate'].mean()*100:.1f}%")
        print("="*80 + "\n")

    def save_leaderboard_to_json(self, df: pd.DataFrame, filepath: str):
        """
        Save leaderboard to JSON file for API/dashboard

        Args:
            df: Leaderboard DataFrame
            filepath: Path to save JSON file
        """
        if df.empty:
            return

        # Convert to dict-serializable format
        data = {
            'generated_at': datetime.now().isoformat(),
            'total_kols': len(df),
            'leaderboard': df.to_dict('records')
        }

        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"[+] Leaderboard saved to: {filepath}")

    def run_analysis_cycle(self) -> int:
        """
        Run one complete analysis cycle

        Returns:
            Number of KOLs analyzed
        """
        session = db.get_session()

        try:
            # Get KOLs with enough data
            kols = self.get_kols_with_enough_data(session, min_trades=5)

            # Filter out already analyzed KOLs
            new_kols = [k for k in kols if k.id not in self.analyzed_kols]

            if not new_kols:
                print(f"[*] {datetime.now().strftime('%H:%M:%S')} - No new KOLs to analyze")
                return 0

            print(f"\n[*] {datetime.now().strftime('%H:%M:%S')} - Analyzing {len(new_kols)} new KOL(s)...")

            # Analyze each KOL
            all_results = []
            for kol in new_kols:
                result = self.analyze_single_kol(kol, session)
                if result:
                    all_results.append(result)
                    self.analyzed_kols.add(kol.id)

                    # Show progress
                    try:
                        name = kol.name.encode('ascii', 'ignore').decode('ascii')
                    except:
                        name = 'Unknown'
                    print(f"  [+] Analyzed {name}: Diamond Hand Score {result['diamond_hand_score']:.1f}")

            if not all_results:
                return 0

            # Get all analyzed KOLs for full leaderboard
            all_analyzed_kols = session.query(KOL).filter(
                KOL.id.in_(self.analyzed_kols)
            ).all()

            complete_results = []
            for kol in all_analyzed_kols:
                result = self.analyze_single_kol(kol, session)
                if result:
                    complete_results.append(result)

            # Generate and print leaderboard
            df = self.generate_leaderboard(complete_results)
            self.print_leaderboard(df, top_n=20)

            # Save to JSON
            output_path = "C:\\Users\\migue\\Desktop\\claude creaciones\\kol_tracker_ml\\data\\leaderboard.json"
            self.save_leaderboard_to_json(df, output_path)

            return len(all_results)

        finally:
            session.close()

    async def run_continuous(self):
        """
        Run continuous analysis loop
        """
        print("="*80)
        print("CONTINUOUS KOL ANALYZER")
        print("="*80)
        print(f"Check interval: {self.check_interval} seconds")
        print("Press Ctrl+C to stop")
        print("="*80)

        # Initial analysis
        print("\n[*] Running initial analysis...")
        analyzed = self.run_analysis_cycle()

        if analyzed > 0:
            print(f"\n[OK] Initial analysis complete: {analyzed} KOL(s) analyzed")
        else:
            print(f"\n[*] No KOLs with enough data yet (need >=5 trades)")

        # Continuous loop
        try:
            while True:
                await asyncio.sleep(self.check_interval)
                analyzed = self.run_analysis_cycle()

                if analyzed > 0:
                    print(f"\n[OK] New KOLs analyzed: {analyzed}")

        except KeyboardInterrupt:
            print("\n\n[*] Analyzer stopped by user")


async def main():
    """Main entry point"""
    analyzer = ContinuousAnalyzer(check_interval=300)  # Check every 5 minutes
    await analyzer.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
