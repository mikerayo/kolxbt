import sys
from pathlib import Path
# Add parent directory to path for imports within core
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
ML Models for KOL Trader Classification and Scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from core.config import (
    DIAMOND_HAND_WEIGHTS, N_CLUSTERS, CLUSTER_NAMES,
    DIAMOND_HAND_MIN_HOLD_TIME, DIAMOND_HAND_MIN_3X_RATE,
    SCALPER_HOLD_TIME_THRESHOLD
)


class DiamondHandScorer:
    """
    Calculates "Diamond Hand" score for KOLs
    Higher score = better long-term consistent trader (3x+)
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scorer

        Args:
            weights: Dictionary of feature weights (sum should be 1.0)
        """
        self.weights = weights or DIAMOND_HAND_WEIGHTS

    def calculate_score(self, features: Dict[str, float]) -> float:
        """
        Calculate diamond hand score (0-100)

        Args:
            features: Dictionary of KOL features

        Returns:
            Score from 0 to 100
        """
        score = 0.0

        # 3x+ rate (35% weight) - most important
        # Scale: 0% = 0 points, 50%+ = 100 points
        three_x_rate = features.get('three_x_plus_rate', 0)
        three_x_score = min(three_x_rate / 0.5, 1.0) * 100
        score += three_x_score * self.weights.get('three_x_plus_rate', 0.35)

        # Win rate (25% weight)
        # Scale: 0% = 0 points, 80%+ = 100 points
        win_rate = features.get('win_rate', 0)
        win_rate_score = min(win_rate / 0.8, 1.0) * 100
        score += win_rate_score * self.weights.get('win_rate', 0.25)

        # Average hold time (20% weight)
        # Scale: <5min = 0 points, 24h+ = 100 points
        avg_hold_hours = features.get('avg_hold_time_hours', 0)
        hold_time_score = min(avg_hold_hours / 24, 1.0) * 100
        score += hold_time_score * self.weights.get('avg_hold_time', 0.20)

        # Consistency (15% weight)
        consistency = features.get('consistency_score', 0)
        consistency_score = consistency * 100
        score += consistency_score * self.weights.get('consistency_score', 0.15)

        # Sample size (5% weight) - reward more data
        # Scale: <10 trades = 0 points, 100+ trades = 100 points
        total_trades = features.get('total_trades', 0)
        sample_score = min(total_trades / 100, 1.0) * 100
        score += sample_score * self.weights.get('total_trades', 0.05)

        return round(score, 2)

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate scores for all KOLs in a DataFrame

        Args:
            df: DataFrame with KOL features

        Returns:
            DataFrame with added 'diamond_hand_score' column
        """
        scores = []

        for _, row in df.iterrows():
            features = row.to_dict()
            score = self.calculate_score(features)
            scores.append(score)

        df = df.copy()
        df['diamond_hand_score'] = scores
        return df


class KOLClusterer:
    """
    Clusters KOLs into trading styles using K-Means
    """

    def __init__(self, n_clusters: int = N_CLUSTERS):
        """
        Initialize clusterer

        Args:
            n_clusters: Number of clusters to create
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_names = None

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit K-Means and predict clusters

        Args:
            df: DataFrame with KOL features

        Returns:
            DataFrame with added 'cluster' and 'cluster_name' columns
        """
        # Filter KOLs with no trades
        df_active = df[df['total_trades'] > 0].copy()

        if len(df_active) < self.n_clusters:
            print(f"[!] Not enough active KOLs ({len(df_active)}) for {self.n_clusters} clusters")
            df = df.copy()
            df['cluster'] = -1
            df['cluster_name'] = 'Unknown'
            return df

        # Select features for clustering
        feature_cols = [
            'avg_hold_time_hours',
            'win_rate',
            'three_x_plus_rate',
            'consistency_score',
            'avg_multiple'
        ]

        # Filter to only available columns
        feature_cols = [col for col in feature_cols if col in df_active.columns]

        # Handle missing values
        X = df_active[feature_cols].fillna(0).values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = feature_cols

        # Fit K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)

        # Add clusters to dataframe
        df_result = df.copy()
        df_result.loc[df_active.index, 'cluster'] = clusters
        df_result['cluster'] = df_result['cluster'].fillna(-1).astype(int)

        # Name clusters based on cluster centers
        cluster_names = self._name_clusters(df_result)
        df_result['cluster_name'] = df_result['cluster'].map(cluster_names).fillna('Unknown')

        return df_result

    def _name_clusters(self, df: pd.DataFrame) -> Dict[int, str]:
        """
        Assign meaningful names to clusters based on their characteristics

        Args:
            df: DataFrame with cluster assignments

        Returns:
            Dictionary mapping cluster ID to name
        """
        cluster_names = {}

        for cluster_id in sorted(df['cluster'].unique()):
            if cluster_id == -1:
                cluster_names[cluster_id] = 'No Data'
                continue

            cluster_df = df[df['cluster'] == cluster_id]

            avg_hold = cluster_df['avg_hold_time_hours'].mean()
            three_x_rate = cluster_df['three_x_plus_rate'].mean()
            win_rate = cluster_df['win_rate'].mean()

            # Determine cluster type
            if avg_hold >= (DIAMOND_HAND_MIN_HOLD_TIME / 3600) and three_x_rate >= DIAMOND_HAND_MIN_3X_RATE:
                cluster_names[cluster_id] = 'Diamond Hands'
            elif avg_hold < (SCALPER_HOLD_TIME_THRESHOLD / 3600):
                cluster_names[cluster_id] = 'Scalpers'
            elif win_rate < 0.4:
                cluster_names[cluster_id] = 'Losers'
            elif three_x_rate < 0.1:
                cluster_names[cluster_id] = 'Inconsistent'
            else:
                cluster_names[cluster_id] = f'Cluster {cluster_id}'

        return cluster_names

    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for each cluster

        Args:
            df: DataFrame with cluster assignments

        Returns:
            DataFrame with cluster summaries
        """
        summaries = []

        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_name = cluster_df['cluster_name'].iloc[0] if 'cluster_name' in df.columns else str(cluster_id)

            summary = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'count': len(cluster_df),
                'avg_hold_time_hours': cluster_df['avg_hold_time_hours'].mean(),
                'avg_win_rate': cluster_df['win_rate'].mean(),
                'avg_three_x_rate': cluster_df['three_x_plus_rate'].mean(),
                'avg_diamond_score': cluster_df.get('diamond_hand_score', pd.Series([0])).mean()
            }

            summaries.append(summary)

        return pd.DataFrame(summaries)


class AnomalyDetector:
    """
    Detects anomalies in KOL trading patterns using Isolation Forest
    Useful for detecting when a KOL changes their trading style
    """

    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector

        Args:
            contamination: Expected proportion of outliers
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, df: pd.DataFrame):
        """
        Fit the anomaly detector

        Args:
            df: DataFrame with KOL features
        """
        # Filter KOLs with no trades
        df_active = df[df['total_trades'] > 0]

        if len(df_active) < 10:
            print("[!] Not enough data to train anomaly detector")
            return

        # Select features
        feature_cols = [
            'avg_hold_time_hours',
            'win_rate',
            'three_x_plus_rate',
            'avg_multiple'
        ]

        feature_cols = [col for col in feature_cols if col in df_active.columns]
        X = df_active[feature_cols].fillna(0).values

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        self.model.fit(X_scaled)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict anomalies

        Args:
            df: DataFrame with KOL features

        Returns:
            DataFrame with 'is_anomaly' column
        """
        if self.model is None:
            print("[!] Model not fitted. Call fit() first.")
            df = df.copy()
            df['is_anomaly'] = False
            return df

        # Filter KOLs with trades
        df_active = df[df['total_trades'] > 0].copy()

        # Select features
        feature_cols = [
            'avg_hold_time_hours',
            'win_rate',
            'three_x_plus_rate',
            'avg_multiple'
        ]

        feature_cols = [col for col in feature_cols if col in df_active.columns]
        X = df_active[feature_cols].fillna(0).values

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)
        is_anomaly = predictions == -1

        # Add to dataframe
        df_result = df.copy()
        df_result['is_anomaly'] = False
        df_result.loc[df_active.index, 'is_anomaly'] = is_anomaly

        return df_result

    def get_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get only anomalous KOLs

        Args:
            df: DataFrame with KOL features and anomaly predictions

        Returns:
            DataFrame with only anomalous KOLs
        """
        if 'is_anomaly' not in df.columns:
            df = self.predict(df)

        return df[df['is_anomaly'] == True]


class MLPipeline:
    """
    Complete ML pipeline for KOL analysis
    """

    def __init__(self):
        self.scorer = DiamondHandScorer()
        self.clusterer = KOLClusterer()
        self.anomaly_detector = AnomalyDetector()

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete ML analysis on KOL features

        Args:
            df: DataFrame with KOL features

        Returns:
            DataFrame with scores, clusters, and anomaly flags
        """
        print("[*] Running ML analysis...")

        # Calculate diamond hand scores
        print("  - Calculating Diamond Hand scores...")
        df = self.scorer.score_dataframe(df)

        # Cluster KOLs
        print("  - Clustering KOLs by trading style...")
        df = self.clusterer.fit_predict(df)

        # Detect anomalies
        print("  - Detecting trading pattern anomalies...")
        self.anomaly_detector.fit(df)
        df = self.anomaly_detector.predict(df)

        print(f"[+] Analysis complete for {len(df)} KOLs")
        return df

    def get_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics

        Args:
            df: DataFrame with ML analysis results

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_kols': len(df),
            'kols_with_trades': len(df[df['total_trades'] > 0]),
            'diamond_hands': len(df[df['is_diamond_hand'] == True]),
            'scalpers': len(df[df['is_scalper'] == True]),
            'anomalies': len(df[df['is_anomaly'] == True]) if 'is_anomaly' in df.columns else 0,
            'avg_diamond_score': df.get('diamond_hand_score', pd.Series([0])).mean(),
        }

        # Cluster breakdown
        if 'cluster_name' in df.columns:
            summary['clusters'] = df['cluster_name'].value_counts().to_dict()

        return summary


if __name__ == "__main__":
    # Test ML models
    from feature_engineering import calculate_features_for_all_kols

    print("=" * 70)
    print("ML Models Test")
    print("=" * 70)

    # Calculate features
    print("\n[*] Calculating KOL features...")
    df = calculate_features_for_all_kols()

    if df.empty or df['total_trades'].sum() == 0:
        print("\n[!] No KOLs with trades found. Run wallet_tracker first.")
    else:
        # Run ML pipeline
        pipeline = MLPipeline()
        df = pipeline.analyze(df)

        # Show results
        print("\n" + "=" * 70)
        print("ML ANALYSIS RESULTS")
        print("=" * 70)

        # Summary
        summary = pipeline.get_summary(df)
        print(f"\n📊 Summary:")
        print(f"  Total KOLs: {summary['total_kols']}")
        print(f"  KOLs with trades: {summary['kols_with_trades']}")
        print(f"  Diamond Hands: {summary['diamond_hands']}")
        print(f"  Scalpers: {summary['scalpers']}")
        print(f"  Anomalies: {summary['anomalies']}")
        print(f"  Avg Diamond Score: {summary['avg_diamond_score']:.2f}")

        # Top 10 Diamond Hands
        print("\n🏆 Top 10 Diamond Hands:")
        print("-" * 70)
        top_10 = df.nlargest(10, 'diamond_hand_score')
        for _, row in top_10.iterrows():
            print(f"  {row['name']}: {row['diamond_hand_score']:.1f}/100")

        # Cluster summary
        print("\n📊 Cluster Summary:")
        print("-" * 70)
        cluster_summary = pipeline.clusterer.get_cluster_summary(df)
        print(cluster_summary.to_string(index=False))
