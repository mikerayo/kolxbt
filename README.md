# 💎 KOL Tracker ML System

A Machine Learning system for tracking and analyzing Solana KOL (Key Opinion Leader) traders. Identifies **"Diamond Hands"** - consistent traders who hold positions >5 minutes and achieve 3x+ returns.

## 🎯 Features

- **Real-time Wallet Tracking**: Monitor 1800+ KOL wallets via Solana RPC
- **DEX Swap Detection**: Detects trades from Raydium, Jupiter, Orca
- **Position Matching**: Matches buy/sell pairs to calculate hold times and PnL
- **ML Scoring**: Ranks traders by "Diamond Hand" score (0-100)
- **Clustering**: Groups KOLs by trading style (Diamond Hands, Scalpers, etc.)
- **Anomaly Detection**: Identifies when traders change patterns
- **Automated Reports**: Generates weekly leaderboards and analysis

## 📊 What it Tracks

- **Hold Time**: How long a trader holds a position
- **3x+ Rate**: Percentage of trades that return 3x or more
- **Win Rate**: Percentage of profitable trades
- **Consistency**: How reliable a trader's returns are
- **Diamond Hand Score**: Composite score (0-100) ranking traders

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd kol_tracker_ml
pip install -r requirements.txt
```

### 2. Initialize Database

```bash
python run_tracker.py --mode init
```

This loads KOLs from `kolscan_complete_kols_all_periods.json` into the database.

### 3. Track Wallets (Collect Data)

Track all KOLs for the last 30 days:
```bash
python run_tracker.py --mode track --days 30
```

Or test with just 10 KOLs:
```bash
python run_tracker.py --mode track --num-kols 10 --days 7
```

⚠️ **Warning**: Tracking all KOLs can take several hours due to Solana RPC rate limits.

### 4. Match Positions

```bash
python run_tracker.py --mode match
```

This matches buy/sell trades to create closed positions.

### 5. Analyze & Generate Reports

```bash
python run_tracker.py --mode analyze
```

Generates:
- `reports/kol_report_TIMESTAMP.txt` - Full analysis report
- `reports/leaderboard.txt` - Top 50 Diamond Hands
- `reports/kol_data_TIMESTAMP.json` - Raw data export

### 6. Full Pipeline (All-in-One)

Run everything at once:
```bash
python run_tracker.py --mode all --days 30
```

Or test with a small sample:
```bash
python run_tracker.py --mode all --num-kols 10 --days 7
```

## 📁 Project Structure

```
kol_tracker_ml/
├── config.py                 # Configuration settings
├── database.py               # SQLAlchemy ORM and database management
├── wallet_tracker.py         # Solana RPC wallet tracker
├── transaction_parser.py     # DEX swap parser
├── feature_engineering.py    # Feature calculation (hold time, PnL, etc.)
├── ml_models.py              # ML models (scoring, clustering, anomaly detection)
├── analyzer.py               # Report generation
├── utils.py                  # Utility functions
├── run_tracker.py            # Main entry point
├── requirements.txt          # Python dependencies
└── data/
    ├── kol_tracker.db        # SQLite database
    └── reports/              # Generated reports
```

## 📊 Output Example

### Leaderboard Output
```
═════════════════════════════════════════════════════════
💎 DIAMOND HANDS LEADERBOARD
═════════════════════════════════════════════════════════

🥇 Cented (@Cented7)
    💎 Diamond Hand Score: 94.2/100
    📊 3x+ Trades: 47 / 89 (52.8%)
    ⏱️  Avg Hold Time: 18.3 hours
    💰 Avg Multiple: 4.7x
    🎯 Win Rate: 68.5%

🥈 Gake (@Ga__ke)
    💎 Diamond Hand Score: 91.8/100
    📊 3x+ Trades: 23 / 41 (56.1%)
    ⏱️  Avg Hold Time: 12.7 hours
    💰 Avg Multiple: 5.2x
    🎯 Win Rate: 65.9%
```

## 🧬 ML Model Details

### Diamond Hand Score Calculation

The score (0-100) is calculated using weighted features:

- **3x+ Rate (35%)**: Percentage of trades returning ≥3x
- **Win Rate (25%)**: Percentage of profitable trades
- **Hold Time (20%)**: Average time holding positions
- **Consistency (15%)**: Reliability of returns
- **Sample Size (5%)**: Number of trades (rewards more data)

### Clustering

KOLs are clustered into 5 groups:
1. **Diamond Hands**: Long holds, high 3x+ rate
2. **Scalpers**: Short holds (<5 min), high frequency
3. **Losers**: Low win rate
4. **Inconsistent**: High variance
5. **New Traders**: Few trades

## ⚙️ Configuration

Edit `config.py` to customize:

- `SOLANA_RPC_URL`: Solana RPC endpoint
- `MIN_HOLD_TIME_SECONDS`: Minimum hold time for Diamond Hands (default: 5 min)
- `TARGET_MULTIPLE`: Target return multiple (default: 3x)
- `DIAMOND_HAND_WEIGHTS`: Feature weights for scoring
- `HISTORY_DAYS`: Default days of history to fetch

## 🔧 Troubleshooting

### RPC Rate Limits

If you hit rate limits:
- Use a paid RPC provider (QuickNode, Helius, Triton)
- Increase `RATE_LIMIT_DELAY` in `config.py`
- Track fewer KOLs at once with `--num-kols`

### No Trades Found

- Check that `kolscan_complete_kols_all_periods.json` exists
- Verify wallets are active traders (some KOLs may not have recent activity)
- Try increasing `--days` to fetch more history

### Database Errors

```bash
# Reset database (deletes all data)
rm kol_tracker_ml/database/kol_tracker.db
python run_tracker.py --mode init
```

## 📝 Requirements

- Python 3.8+
- AsyncIO compatible environment
- Solana RPC access (public or paid)

## 🤝 Contributing

This is a research/educational project. Feel free to extend with:

- Additional DEX support (Meteora, Aldrin, etc.)
- Real-time streaming via WebSocket subscriptions
- Web dashboard (Streamlit/FastAPI)
- Telegram/Discord alerts for top KOL trades
- Backtesting simulation for copy trading strategies

## ⚠️ Disclaimer

This tool is for educational purposes only. Not financial advice. Always do your own research before copying any trades.

## 📄 License

MIT License - Feel free to use and modify

---

**Made with ❤️ for the Solana community**
