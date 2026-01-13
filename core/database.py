"""
Database models and management for KOL Tracker ML System
"""

import json
import sys
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime,
    ForeignKey, Text, Boolean, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func

# Add parent directory to path for imports within core
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import DATABASE_URL, KOLS_DATA_FILE

# Create base class for models
Base = declarative_base()


class KOL(Base):
    """
    Key Opinion Leader (trader) model
    """
    __tablename__ = 'kols'

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(44), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    twitter_username = Column(String(100), nullable=True)
    rank = Column(Integer, nullable=True)
    avatar_url = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trades = relationship("Trade", back_populates="kol", cascade="all, delete-orphan")
    closed_positions = relationship("ClosedPosition", back_populates="kol", cascade="all, delete-orphan")
    open_positions = relationship("OpenPosition", back_populates="kol", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<KOL(id={self.id}, name={self.name}, wallet={self.wallet_address[:8]}...)>"

    @property
    def short_address(self) -> str:
        """Get shortened wallet address"""
        if not self.wallet_address:
            return "Unknown"
        return f"{self.wallet_address[:8]}...{self.wallet_address[-8:]}"


class Trade(Base):
    """
    Individual trade (buy or sell) model
    """
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True, autoincrement=True)
    kol_id = Column(Integer, ForeignKey('kols.id'), nullable=False, index=True)

    # Trade details
    token_address = Column(String(44), nullable=False, index=True)
    operation = Column(String(10), nullable=False)  # 'buy' or 'sell'

    # Amounts and price
    amount_sol = Column(Float, nullable=False)
    amount_token = Column(Float, nullable=False)
    price = Column(Float, nullable=True)  # SOL per token

    # Metadata
    dex = Column(String(50), nullable=True)  # raydium, jupiter, orca
    timestamp = Column(DateTime, nullable=False, index=True)
    tx_signature = Column(String(88), unique=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    kol = relationship("KOL", back_populates="trades")

    # Indexes for common queries
    __table_args__ = (
        Index('idx_kol_token', 'kol_id', 'token_address'),
        Index('idx_timestamp_operation', 'timestamp', 'operation'),
    )

    def __repr__(self):
        return f"<Trade(id={self.id}, kol_id={self.kol_id}, {self.operation} {self.token_address[:8]}...)>"


class ClosedPosition(Base):
    """
    Closed position (complete buy+sell trade) with calculated metrics
    """
    __tablename__ = 'closed_positions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    kol_id = Column(Integer, ForeignKey('kols.id'), nullable=False, index=True)

    # Position details
    token_address = Column(String(44), nullable=False, index=True)

    # Entry
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)  # SOL per token at entry

    # Exit
    exit_time = Column(DateTime, nullable=False)
    exit_price = Column(Float, nullable=False)  # SOL per token at exit

    # Metrics
    hold_time_seconds = Column(Float, nullable=False)
    pnl_sol = Column(Float, nullable=False)  # Profit/Loss in SOL
    pnl_multiple = Column(Float, nullable=False)  # 3x = 3.0, -0.5 = -50%

    # Metadata
    dex = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    kol = relationship("KOL", back_populates="closed_positions")

    # Indexes
    __table_args__ = (
        Index('idx_kol_position', 'kol_id', 'token_address'),
        Index('idx_hold_time', 'hold_time_seconds'),
        Index('idx_pnl_multiple', 'pnl_multiple'),
    )

    @property
    def is_profitable(self) -> bool:
        """Check if position was profitable"""
        return self.pnl_multiple > 1.0

    @property
    def is_diamond_hand(self) -> bool:
        """Check if this qualifies as a diamond hand trade (>=3x, >5min hold)"""
        return self.pnl_multiple >= 3.0 and self.hold_time_seconds > 300

    @property
    def hold_time_hours(self) -> float:
        """Hold time in hours"""
        return self.hold_time_seconds / 3600

    def __repr__(self):
        return f"<ClosedPosition(id={self.id}, kol_id={self.kol_id}, {self.pnl_multiple}x, {self.hold_time_hours:.1f}h)>"


class OpenPosition(Base):
    """
    Open position tracker - tracks currently active positions

    Monitors:
    - When KOL bought a token
    - Current price of the token
    - Unrealized PnL
    - How long they've been holding
    - Quality analysis when they sell
    """
    __tablename__ = 'open_positions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    kol_id = Column(Integer, ForeignKey('kols.id'), nullable=False, index=True)

    # Position details
    token_address = Column(String(44), nullable=False, index=True)
    token_symbol = Column(String(50), nullable=True)  # Optional: token symbol

    # Entry information
    entry_time = Column(DateTime, nullable=False, index=True)
    entry_price = Column(Float, nullable=False)  # SOL per token at entry
    entry_amount_sol = Column(Float, nullable=False)  # Total SOL invested
    entry_amount_token = Column(Float, nullable=False)  # Tokens received

    # Current tracking
    last_price_update = Column(DateTime, nullable=True)
    current_price = Column(Float, nullable=True)  # Latest known price
    peak_price = Column(Float, nullable=True)  # Highest price since entry
    peak_price_time = Column(DateTime, nullable=True)  # When peak occurred

    # Metadata
    dex = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    kol = relationship("KOL", back_populates="open_positions")

    # Indexes
    __table_args__ = (
        Index('idx_open_kol_token', 'kol_id', 'token_address'),
        Index('idx_open_entry_time', 'entry_time'),
    )

    @property
    def hold_time_seconds(self) -> float:
        """How long position has been open"""
        now = datetime.utcnow()
        delta = now - self.entry_time
        return delta.total_seconds()

    @property
    def hold_time_hours(self) -> float:
        """Hold time in hours"""
        return self.hold_time_seconds / 3600

    @property
    def current_value_sol(self) -> float:
        """Current value in SOL"""
        if self.current_price is None:
            return self.entry_amount_sol
        return self.entry_amount_token * self.current_price

    @property
    def unrealized_pnl_sol(self) -> float:
        """Unrealized profit/loss in SOL"""
        return self.current_value_sol - self.entry_amount_sol

    @property
    def unrealized_pnl_multiple(self) -> float:
        """Unrealized profit/loss as multiple"""
        if self.entry_amount_sol == 0:
            return 0
        return self.current_value_sol / self.entry_amount_sol

    @property
    def is_profitable(self) -> bool:
        """Check if position is currently profitable"""
        return self.unrealized_pnl_multiple > 1.0

    @property
    def peak_multiple_reached(self) -> float:
        """Best multiple reached so far"""
        if self.peak_price is None or self.entry_price == 0:
            return 0
        return self.peak_price / self.entry_price

    @property
    def is_above_peak(self) -> bool:
        """Check if current price is at or above peak"""
        if self.current_price is None or self.peak_price is None:
            return False
        return self.current_price >= self.peak_price

    def __repr__(self):
        return f"<OpenPosition(id={self.id}, kol_id={self.kol_id}, {self.token_address[:8]}..., {self.unrealized_pnl_multiple:.2f}x)>"


class TradeQuality(Base):
    """
    Trade quality analysis - evaluates how good a trade was

    Analyzes:
    - Did they sell too early? (missed profit)
    - Did they sell at the peak? (perfect exit)
    - Did they hold too long? (gave back profits)
    - What was the optimal exit?
    """
    __tablename__ = 'trade_quality'

    id = Column(Integer, primary_key=True, autoincrement=True)
    kol_id = Column(Integer, ForeignKey('kols.id'), nullable=False, index=True)

    # Reference to closed position
    closed_position_id = Column(Integer, ForeignKey('closed_positions.id'), nullable=False, index=True)
    token_address = Column(String(44), nullable=False, index=True)

    # Entry info
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)

    # Exit info
    exit_time = Column(DateTime, nullable=False)
    exit_price = Column(Float, nullable=False)

    # Peak analysis (what happened AFTER they sold?)
    peak_price_after_exit = Column(Float, nullable=True)  # Highest price AFTER they sold
    peak_time_after_exit = Column(DateTime, nullable=True)  # When that peak occurred
    minutes_to_peak_after_exit = Column(Float, nullable=True)  # How long after selling

    # Quality metrics
    sold_at_peak = Column(Integer, nullable=False, default=0)  # 1 if sold within 5% of peak
    sold_early = Column(Integer, nullable=False, default=0)  # 1 if price went up >20% after selling
    held_too_long = Column(Integer, nullable=False, default=0)  # 1 if price dropped >50% from peak before exit
    missed_profit_sol = Column(Float, nullable=True)  # How much SOL they left on the table
    missed_profit_percentage = Column(Float, nullable=True)  # Percentage of profit missed

    # Scoring
    quality_score = Column(Float, nullable=False)  # 0-100, higher is better
    timing_score = Column(Float, nullable=True)  # 0-100, how good was the timing

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    kol = relationship("KOL")
    closed_position = relationship("ClosedPosition")

    # Indexes
    __table_args__ = (
        Index('idx_quality_kol', 'kol_id'),
        Index('idx_quality_token', 'token_address'),
        Index('idx_quality_score', 'quality_score'),
    )

    @property
    def quality_label(self) -> str:
        """Human-readable quality label"""
        if self.quality_score >= 90:
            return "PERFECT"
        elif self.quality_score >= 75:
            return "EXCELLENT"
        elif self.quality_score >= 60:
            return "GOOD"
        elif self.quality_score >= 40:
            return "OK"
        elif self.quality_score >= 25:
            return "SUBOPTIMAL"
        else:
            return "POOR"

    def __repr__(self):
        return f"<TradeQuality(id={self.id}, score={self.quality_score:.0f}, {self.quality_label})>"


class DiscoveredTrader(Base):
    """
    Trader discovered through token analysis

    When a KOL buys a token, we analyze who else bought that token.
    If another wallet has better metrics than our KOLs, we add them here
    for tracking and potential promotion to KOL status.
    """
    __tablename__ = 'discovered_traders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    wallet_address = Column(String(44), unique=True, nullable=False, index=True)

    # Discovery info
    discovered_from_token = Column(String(44), nullable=False, index=True)  # Token where we found them
    discovered_from_kol_id = Column(Integer, ForeignKey('kols.id'), nullable=True)
    discovered_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Performance metrics (calculated from their trades)
    total_trades = Column(Integer, default=0)
    total_volume_sol = Column(Float, default=0)
    win_rate = Column(Float, default=0)  # 0-1
    three_x_rate = Column(Float, default=0)  # 0-1
    avg_hold_time_hours = Column(Float, default=0)
    total_pnl_sol = Column(Float, default=0)

    # Discovery score (how good this trader is compared to our KOLs)
    discovery_score = Column(Float, default=0)  # 0-100, higher is better

    # Status
    is_tracking = Column(Boolean, default=False)  # Are we tracking this wallet?
    promoted_to_kol = Column(Boolean, default=False)  # Has been promoted to KOL status?
    promoted_at = Column(DateTime, nullable=True)

    # Metadata
    last_activity_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    discovered_from_kol = relationship("KOL")

    # Indexes
    __table_args__ = (
        Index('idx_discovery_score', 'discovery_score'),
        Index('idx_is_tracking', 'is_tracking'),
        Index('idx_promoted', 'promoted_to_kol'),
    )

    @property
    def short_address(self) -> str:
        """Get shortened wallet address"""
        if not self.wallet_address:
            return "Unknown"
        return f"{self.wallet_address[:8]}...{self.wallet_address[-8:]}"

    @property
    def quality_label(self) -> str:
        """Human-readable quality label based on discovery_score"""
        if self.discovery_score >= 80:
            return "ELITE"
        elif self.discovery_score >= 70:
            return "EXCELLENT"
        elif self.discovery_score >= 60:
            return "GOOD"
        elif self.discovery_score >= 50:
            return "DECENT"
        else:
            return "AVERAGE"

    def __repr__(self):
        return f"<DiscoveredTrader(id={self.id}, score={self.discovery_score:.0f}, {self.quality_label})>"


class TokenInfo(Base):
    """
    Metadata de tokens obtenida de DexScreener API

    Almacena información como nombre, ticker, logo, precio, liquidez, etc.
    """
    __tablename__ = 'token_info'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Token identification
    token_address = Column(String(44), unique=True, nullable=False, index=True)

    # Token metadata
    name = Column(String(255), nullable=True)
    symbol = Column(String(50), nullable=True)  # Ticker
    logo_url = Column(String(512), nullable=True)

    # Price information
    price_usd = Column(Float, nullable=True)
    liquidity_usd = Column(Float, nullable=True)
    fdv_usd = Column(Float, nullable=True)  # Fully Diluted Valuation
    market_cap_usd = Column(Float, nullable=True)
    volume_24h_usd = Column(Float, nullable=True)

    # Price change
    change_24h_percent = Column(Float, nullable=True)

    # DEX information
    chain_id = Column(String(50), nullable=True)  # e.g., "solana"
    dex_id = Column(String(50), nullable=True)  # e.g., "pump_fun", "raydium"
    pair_address = Column(String(44), nullable=True)

    # ==================== BUBBLEMAPS DATA ====================
    # Holder distribution metrics
    top1_percentage = Column(Float, nullable=True)  # % held by #1 holder
    top10_percentage = Column(Float, nullable=True)  # % held by top 10
    top20_percentage = Column(Float, nullable=True)  # % held by top 20
    top10_retail_percentage = Column(Float, nullable=True)  # % held by top 10 retail (excl. CEX/DEX)
    gini_coefficient = Column(Float, nullable=True)  # 0-1, inequality measure
    concentration_risk = Column(Float, nullable=True)  # 0-100, risk score

    # Holder counts
    holder_count = Column(Integer, nullable=True)  # Total unique holders
    cluster_count = Column(Integer, nullable=True)  # Number of clusters
    supernode_count = Column(Integer, nullable=True)  # Number of supernodes (whales)
    dev_wallet_count = Column(Integer, nullable=True)  # Number of dev/team wallets

    # Specific holder percentages
    dev_percentage = Column(Float, nullable=True)  # % held by dev/team
    cex_percentage = Column(Float, nullable=True)  # % held by CEXs
    dex_percentage = Column(Float, nullable=True)  # % held by DEXs
    contract_percentage = Column(Float, nullable=True)  # % held by contracts
    largest_cluster_percentage = Column(Float, nullable=True)  # % in largest cluster

    # Decentralization score (from Bubblemaps)
    decentralization_score = Column(Integer, nullable=True)  # 0-100 from Bubblemaps

    # Tracking
    bubblemaps_updated = Column(DateTime, nullable=True)  # Last Bubblemaps update

    # ==================== END BUBBLEMAPS DATA ====================

    # Tracking
    last_updated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes
    __table_args__ = (
        Index('idx_token_symbol', 'symbol'),
        Index('idx_token_chain', 'chain_id'),
        Index('idx_token_dex', 'dex_id'),
    )

    @property
    def formatted_price(self) -> str:
        """Price formatted for display"""
        if self.price_usd and self.price_usd > 0:
            if self.price_usd >= 1:
                return f"${self.price_usd:.4f}"
            elif self.price_usd >= 0.000001:
                return f"${self.price_usd:.8f}"
            else:
                return f"${self.price_usd:.12f}"
        return "N/A"

    @property
    def formatted_volume(self) -> str:
        """Volume formatted for display"""
        if self.volume_24h_usd and self.volume_24h_usd > 0:
            return f"${self.volume_24h_usd:,.0f}"
        return "N/A"

    @property
    def formatted_liquidity(self) -> str:
        """Liquidity formatted for display"""
        if self.liquidity_usd and self.liquidity_usd > 0:
            return f"${self.liquidity_usd:,.0f}"
        return "N/A"

    def __repr__(self):
        return f"<TokenInfo({self.symbol or self.token_address[:8]}..., ${self.formatted_price})>"


# Database management class
class Database:
    """
    Database session and initialization manager
    """

    def __init__(self, database_url: str = DATABASE_URL):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def create_tables(self):
        """Create all tables in the database"""
        Base.metadata.create_all(bind=self.engine)
        print(f"[*] Database tables created at: {self.engine.url}")

    def drop_tables(self):
        """Drop all tables (WARNING: This deletes all data!)"""
        Base.metadata.drop_all(bind=self.engine)
        print("[!] Database tables dropped")

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    def load_kols_from_json(self, json_path: str = None) -> int:
        """
        Load KOLs from JSON file into database

        Args:
            json_path: Path to JSON file with KOL data

        Returns:
            Number of KOLs loaded
        """
        if json_path is None:
            json_path = KOLS_DATA_FILE

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            kols_data = data.get('kols', [])
            session = self.get_session()

            loaded_count = 0
            for kol_data in kols_data:
                # Check if KOL already exists
                existing = session.query(KOL).filter(
                    KOL.wallet_address == kol_data['wallet_address']
                ).first()

                if existing:
                    continue

                # Create new KOL
                kol = KOL(
                    wallet_address=kol_data['wallet_address'],
                    name=kol_data.get('name'),
                    twitter_username=kol_data.get('twitter_username'),
                    avatar_url=kol_data.get('avatar'),
                )
                session.add(kol)
                loaded_count += 1

            session.commit()
            session.close()

            print(f"[*] Loaded {loaded_count} KOLs from {json_path}")
            return loaded_count

        except FileNotFoundError:
            print(f"[!] Error: File not found: {json_path}")
            return 0
        except json.JSONDecodeError as e:
            print(f"[!] Error parsing JSON: {e}")
            return 0

    def get_all_kols(self, session: Session = None) -> List[KOL]:
        """Get all KOLs from database"""
        if session is None:
            session = self.get_session()
            should_close = True
        else:
            should_close = False

        try:
            kols = session.query(KOL).all()
            return kols
        finally:
            if should_close:
                session.close()

    def get_kol_by_wallet(self, wallet_address: str, session: Session = None) -> Optional[KOL]:
        """Get KOL by wallet address"""
        if session is None:
            session = self.get_session()
            should_close = True
        else:
            should_close = False

        try:
            kol = session.query(KOL).filter(
                KOL.wallet_address == wallet_address
            ).first()
            return kol
        finally:
            if should_close:
                session.close()


# Global database instance
db = Database()


if __name__ == "__main__":
    # Test database creation
    print("Testing database setup...")

    # Create tables
    db.create_tables()

    # Load KOLs from JSON
    count = db.load_kols_from_json()

    # Show some KOLs
    session = db.get_session()
    kols = db.get_all_kols(session)
    print(f"\n[*] Total KOLs in database: {len(kols)}")

    for kol in kols[:5]:
        print(f"  - {kol.name} ({kol.short_address})")

    session.close()
