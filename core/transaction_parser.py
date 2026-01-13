import sys
from pathlib import Path
# Add parent directory to path for imports within core
sys.path.insert(0, str(Path(__file__).parent.parent))
"""
Transaction Parser for Solana DEX swaps
Extracts trade information from Solana transactions
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import base58
import struct

from core.config import DEX_PROGRAMS, WSOL_MINT


class TransactionParser:
    """
    Parser for Solana transactions to detect and extract DEX swaps
    """

    def __init__(self):
        self.dex_programs = set(DEX_PROGRAMS.values())
        # Importar y agregar parser de pump.fun
        try:
            from pumpfun_parser import PumpFunParser
            self.pumpfun_parser = PumpFunParser()
        except ImportError:
            self.pumpfun_parser = None

    def parse_transaction(self, tx_data: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
        """
        Parse a Solana transaction and extract swap information

        Args:
            tx_data: Raw transaction data from RPC
            wallet_address: The wallet address we're tracking

        Returns:
            Dictionary with trade data or None if not a relevant swap
        """
        if not tx_data or 'result' not in tx_data:
            return None

        transaction = tx_data['result']
        if not transaction or 'meta' not in transaction:
            return None

        meta = transaction['meta']
        message = transaction['transaction']['message']

        # Check if transaction was successful
        if meta.get('err') is not None:
            return None

        # Get timestamp
        block_time = transaction.get('blockTime')
        if block_time:
            timestamp = datetime.fromtimestamp(block_time)
        else:
            timestamp = datetime.utcnow()

        # Get signature
        signature = transaction.get('transaction', {}).get('signatures', [None])[0]

        # First, try to detect pump.fun swaps (MOST IMPORTANT for 100x gems)
        if self.pumpfun_parser:
            pump_trade = self.pumpfun_parser.parse_pumpfun_transaction(tx_data, wallet_address)
            if pump_trade:
                return pump_trade

        # Then try regular DEX swaps
        for instruction in message.get('instructions', []):
            program_id = message['accountKeys'][instruction['programIdIndex']]

            # Check if this is a DEX program
            if program_id in self.dex_programs:
                dex_name = self._get_dex_name(program_id)
                swap_data = self._parse_swap_instruction(
                    instruction, message, meta, wallet_address, dex_name
                )

                if swap_data:
                    swap_data['timestamp'] = timestamp
                    swap_data['tx_signature'] = signature
                    return swap_data

        return None

    def _get_dex_name(self, program_id: str) -> Optional[str]:
        """Get DEX name from program ID"""
        for name, pid in DEX_PROGRAMS.items():
            if pid == program_id:
                return name
        return None

    def _parse_swap_instruction(
        self,
        instruction: Dict[str, Any],
        message: Dict[str, Any],
        meta: Dict[str, Any],
        wallet_address: str,
        dex_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a swap instruction from a DEX

        This is a simplified version. For production, you'd need to implement
        specific parsers for each DEX (Jupiter, Raydium, Orca) as they have
        different instruction formats.
        """

        # Get account keys
        account_keys = message['accountKeys']

        # Try to determine input/output tokens from balance changes
        pre_balances = meta.get('preBalances', [])
        post_balances = meta.get('postBalances', [])

        # Get token balances changes
        pre_token_balances = meta.get('preTokenBalances', {})
        post_token_balances = meta.get('postTokenBalances', {})

        # Parse token balance changes to find the swap
        trade_data = self._parse_token_balance_changes(
            pre_token_balances, post_token_balances, wallet_address
        )

        if trade_data:
            trade_data['dex'] = dex_name
            return trade_data

        return None

    def _parse_token_balance_changes(
        self,
        pre_balances: List[Dict],
        post_balances: List[Dict],
        wallet_address: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse token balance changes to detect swaps

        Returns:
            Dict with token_address, operation, amount_sol, amount_token, price
        """
        # Build a map of account -> mint
        pre_mint_map = {}
        for acc in pre_balances:
            account = acc.get('accountIndex')
            mint = acc.get('mint')
            if account is not None and mint:
                pre_mint_map[account] = mint

        post_mint_map = {}
        for acc in post_balances:
            account = acc.get('accountIndex')
            mint = acc.get('mint')
            if account is not None and mint:
                post_mint_map[account] = mint

        # Get balance changes
        balance_changes = []

        for acc in pre_balances:
            account_index = acc.get('accountIndex')
            if account_index is None:
                continue

            # Get pre balance
            ui_amount = acc.get('uiTokenAmount', {})
            pre_amount = ui_amount.get('uiAmount', 0)
            decimals = ui_amount.get('decimals', 0)
            mint = acc.get('mint')

            # Find matching post balance
            post_acc = next(
                (a for a in post_balances if a.get('accountIndex') == account_index),
                None
            )

            if post_acc:
                post_amount = post_acc.get('uiTokenAmount', {}).get('uiAmount', 0)

                # Calculate change
                change = float(post_amount) - float(pre_amount)

                if abs(change) > 0.000001:  # Minimum threshold
                    balance_changes.append({
                        'mint': mint,
                        'change': change,
                        'decimals': decimals
                    })

        # If we have exactly 2 tokens changing, it's likely a swap
        if len(balance_changes) == 2:
            # Determine which is SOL and which is the token
            sol_change = None
            token_change = None
            token_address = None

            for change in balance_changes:
                if change['mint'] == WSOL_MINT:
                    sol_change = change['change']
                else:
                    token_change = change['change']
                    token_address = change['mint']

            if sol_change is not None and token_change is not None and token_address:
                # Determine operation (buy or sell)
                # If SOL decreased and token increased: BUY
                # If SOL increased and token decreased: SELL
                if sol_change < 0 and token_change > 0:
                    operation = 'buy'
                    amount_sol = abs(sol_change)
                    amount_token = token_change
                elif sol_change > 0 and token_change < 0:
                    operation = 'sell'
                    amount_sol = sol_change
                    amount_token = abs(token_change)
                else:
                    return None

                # Calculate price (SOL per token)
                if amount_token > 0:
                    price = amount_sol / amount_token
                else:
                    price = None

                return {
                    'token_address': token_address,
                    'operation': operation,
                    'amount_sol': amount_sol,
                    'amount_token': amount_token,
                    'price': price
                }

        return None

    def parse_multiple_transactions(
        self,
        transactions: List[Dict[str, Any]],
        wallet_address: str
    ) -> List[Dict[str, Any]]:
        """
        Parse multiple transactions and return all trades found

        Args:
            transactions: List of transaction data from RPC
            wallet_address: The wallet address we're tracking

        Returns:
            List of parsed trade data
        """
        trades = []

        for tx_data in transactions:
            trade = self.parse_transaction(tx_data, wallet_address)
            if trade:
                trades.append(trade)

        return trades


class SimplifiedJupiterParser:
    """
    Simplified parser for Jupiter swaps
    Jupiter is the most popular aggregator on Solana
    """

    JUPITER_PROGRAM_ID = DEX_PROGRAMS['jupiter']

    @staticmethod
    def parse_jupiter_swap(tx_data: Dict[str, Any], wallet_address: str) -> Optional[Dict[str, Any]]:
        """
        Parse a Jupiter swap transaction

        Note: This is a simplified implementation. For production use,
        you should use Jupiter's official API or implement full instruction parsing.
        """
        # Jupiter swaps go through their program
        # The instruction data contains the swap details
        # This would require decoding the instruction data

        # For now, we'll use the generic parser above
        # which works based on balance changes

        return None


# Helper function to get parser for a DEX
def get_parser(dex_name: str) -> TransactionParser:
    """Get appropriate parser for a DEX"""
    return TransactionParser()


if __name__ == "__main__":
    # Test the parser with sample data
    print("Transaction Parser Module")
    print("DEX Programs:", DEX_PROGRAMS)
    print("WSOL Mint:", WSOL_MINT)
