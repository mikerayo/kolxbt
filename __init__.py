"""
KOL Tracker ML System

A machine learning system for tracking and analyzing Solana KOL (Key Opinion Leader) traders.
Identifies "Diamond Hands" - consistent traders who hold positions and achieve 3x+ returns.
"""

__version__ = "1.0.0"
__author__ = "KOL Tracker Team"

from .config import *
from .database import db, KOL, Trade, ClosedPosition
from .feature_engineering import KOLFeatures, PositionMatcher
from .ml_models import MLPipeline, DiamondHandScorer
from .analyzer import ReportGenerator, analyze_and_report

__all__ = [
    'db',
    'KOL',
    'Trade',
    'ClosedPosition',
    'KOLFeatures',
    'PositionMatcher',
    'MLPipeline',
    'DiamondHandScorer',
    'ReportGenerator',
    'analyze_and_report',
]
