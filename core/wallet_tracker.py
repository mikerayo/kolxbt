import sys
from pathlib import Path
# Add parent directory to path for imports within core
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Wallet Tracker - Connects to Solana RPC to track KOL wallet transactions
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from core.config import (
    SOLANA_RPC_URL, MAX_RETRIES, REQUEST_TIMEOUT,
    RATE_LIMIT_DELAY, MAX_SIGNATURES_PER_REQUEST, HISTORY_DAYS
)
from core.transaction_parser import TransactionParser
from core.database import db, Trade, KOL


class WalletTracker:
    """
    Tracks wallet transactions from Solana RPC
    """

    def __init__(self, rpc_url: str = SOLANA_RPC_URL):
        self.rpc_url = rpc_url
        self.parser = TransactionParser()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _rpc_call(
        self,
        method: str,
        params: List[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Make an RPC call to Solana node

        Args:
            method: RPC method name
            params: Parameters for the RPC call

        Returns:
            Response data or None if failed
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or []
        }

        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.post(
                    self.rpc_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                ) as response:
                    data = await response.json()

                    if 'error' in data:
                        print(f"[!] RPC Error: {data['error']}")
                        return None

                    return data

            except asyncio.TimeoutError:
                print(f"[!] Timeout on attempt {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"[!] RPC call failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)

        return None

    async def get_signatures_for_address(
        self,
        wallet_address: str,
        limit: int = MAX_SIGNATURES_PER_REQUEST
    ) -> List[Dict[str, Any]]:
        """
        Get transaction signatures for a wallet address

        Args:
            wallet_address: Solana wallet address
            limit: Maximum number of signatures to fetch

        Returns:
            List of signature objects
        """
        result = await self._rpc_call(
            "getSignaturesForAddress",
            [wallet_address, {"limit": limit}]
        )

        if result and 'result' in result:
            return result['result']

        return []

    async def get_transaction(self, signature: str) -> Optional[Dict[str, Any]]:
        """
        Get full transaction details by signature

        Args:
            signature: Transaction signature

        Returns:
            Transaction data or None if failed
        """
        result = await self._rpc_call(
            "getTransaction",
            [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}]
        )

        if result and 'result' in result:
            return result

        return None

    async def get_recent_transactions(
        self,
        wallet_address: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent transactions for a wallet

        Args:
            wallet_address: Solana wallet address
            limit: Maximum number of transactions to fetch

        Returns:
            List of transaction data
        """
        signatures = await self.get_signatures_for_address(wallet_address, limit)

        if not signatures:
            return []

        transactions = []
        for sig in signatures:
            signature = sig.get('signature')
            if signature:
                tx_data = await self.get_transaction(signature)
                if tx_data:
                    transactions.append(tx_data)

            # Rate limiting
            await asyncio.sleep(RATE_LIMIT_DELAY)

        return transactions

    async def get_historical_transactions(
        self,
        wallet_address: str,
        days: int = HISTORY_DAYS
    ) -> List[Dict[str, Any]]:
        """
        Get historical transactions for a wallet

        Args:
            wallet_address: Solana wallet address
            days: Number of days of history to fetch

        Returns:
            List of transaction data
        """
        # Calculate cutoff time
        cutoff_time = int((datetime.utcnow() - timedelta(days=days)).timestamp())

        all_transactions = []
        last_signature = None

        while True:
            params = {"limit": MAX_SIGNATURES_PER_REQUEST}

            if last_signature:
                params["until"] = last_signature

            result = await self._rpc_call(
                "getSignaturesForAddress",
                [wallet_address, params]
            )

            if not result or 'result' not in result:
                break

            signatures = result['result']

            if not signatures:
                break

            # Check if we've gone past the cutoff time
            oldest_block_time = signatures[-1].get('blockTime', 0)

            if oldest_block_time and oldest_block_time < cutoff_time:
                # Filter signatures after cutoff
                signatures = [
                    s for s in signatures
                    if s.get('blockTime', 0) >= cutoff_time
                ]

            # Fetch transaction details
            for sig in signatures:
                signature = sig.get('signature')
                if signature:
                    tx_data = await self.get_transaction(signature)
                    if tx_data:
                        all_transactions.append(tx_data)

                await asyncio.sleep(RATE_LIMIT_DELAY)

            # Update last signature for pagination
            last_signature = signatures[-1].get('signature')

            # Check if we should stop
            if oldest_block_time and oldest_block_time < cutoff_time:
                break

            print(f"[*] Fetched {len(all_transactions)} transactions so far...")

        return all_transactions

    async def parse_trades_from_transactions(
        self,
        transactions: List[Dict[str, Any]],
        wallet_address: str
    ) -> List[Dict[str, Any]]:
        """
        Parse trades from a list of transactions

        Args:
            transactions: List of transaction data
            wallet_address: The wallet address

        Returns:
            List of parsed trades
        """
        trades = []

        for tx_data in transactions:
            trade = self.parser.parse_transaction(tx_data, wallet_address)
            if trade:
                trades.append(trade)

        return trades

    async def track_wallet(
        self,
        kol: KOL,
        days: int = HISTORY_DAYS,
        save_to_db: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Track a single wallet and extract trades

        Args:
            kol: KOL object from database
            days: Number of days of history to fetch
            save_to_db: Whether to save trades to database

        Returns:
            List of parsed trades
        """
        print(f"[*] Tracking {kol.name} ({kol.short_address})...")

        # Get historical transactions
        transactions = await self.get_historical_transactions(
            kol.wallet_address,
            days
        )

        if not transactions:
            print(f"[-] No transactions found for {kol.name}")
            return []

        # Parse trades from transactions
        trades = await self.parse_trades_from_transactions(
            transactions,
            kol.wallet_address
        )

        print(f"[+] Found {len(trades)} trades for {kol.name}")

        # Save to database if requested
        if save_to_db:
            session = db.get_session()
            try:
                for trade_data in trades:
                    # Check if trade already exists
                    existing = session.query(Trade).filter(
                        Trade.tx_signature == trade_data['tx_signature']
                    ).first()

                    if existing:
                        continue

                    # Create new trade
                    trade = Trade(
                        kol_id=kol.id,
                        token_address=trade_data['token_address'],
                        operation=trade_data['operation'],
                        amount_sol=trade_data['amount_sol'],
                        amount_token=trade_data['amount_token'],
                        price=trade_data.get('price'),
                        dex=trade_data.get('dex'),
                        timestamp=trade_data['timestamp'],
                        tx_signature=trade_data['tx_signature']
                    )
                    session.add(trade)

                session.commit()
                print(f"[+] Saved {len(trades)} trades to database")
            except Exception as e:
                session.rollback()
                print(f"[!] Error saving trades: {e}")
            finally:
                session.close()

        return trades

    async def track_multiple_wallets(
        self,
        kols: List[KOL],
        days: int = HISTORY_DAYS,
        delay_between: float = 1.0
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Track multiple wallets

        Args:
            kols: List of KOL objects
            days: Number of days of history
            delay_between: Delay between wallets (seconds)

        Returns:
            Dictionary mapping kol_id to list of trades
        """
        results = {}

        for i, kol in enumerate(kols):
            print(f"\n[{i+1}/{len(kols)}] Tracking {kol.name}...")

            try:
                trades = await self.track_wallet(kol, days)
                results[kol.id] = trades
            except Exception as e:
                print(f"[!] Error tracking {kol.name}: {e}")
                results[kol.id] = []

            # Delay between requests to avoid rate limiting
            if i < len(kols) - 1:
                await asyncio.sleep(delay_between)

        return results


async def main():
    """Test the wallet tracker"""
    print("=" * 70)
    print("Solana Wallet Tracker Test")
    print("=" * 70)

    # Load KOLs from database
    kols = db.get_all_kols()
    print(f"\n[*] Loaded {len(kols)} KOLs from database")

    # Track first KOL as test
    if kols:
        test_kol = kols[0]
        print(f"\n[*] Testing with: {test_kol.name} ({test_kol.short_address})")

        async with WalletTracker() as tracker:
            trades = await tracker.track_wallet(test_kol, days=7)

            print(f"\n[*] Results:")
            print(f"  - Total transactions fetched: {len(trades)}")

            for trade in trades[:5]:  # Show first 5 trades
                print(f"\n  Trade:")
                print(f"    Operation: {trade['operation']}")
                print(f"    Token: {trade['token_address'][:8]}...")
                print(f"    Amount SOL: {trade['amount_sol']:.4f}")
                print(f"    Amount Token: {trade['amount_token']:.2f}")
                print(f"    Price: {trade.get('price', 'N/A')}")
                print(f"    Timestamp: {trade['timestamp']}")


if __name__ == "__main__":
    asyncio.run(main())
