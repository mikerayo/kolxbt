"""
Pump.fun Parser - Specific parser for pump.fun/pumpswap DEX
Pump.fun uses a bonding curve mechanism different from traditional AMMs
"""

import base58
from typing import Dict, Any, Optional, List
from datetime import datetime


class PumpFunParser:
    """
    Parser específico para transacciones de pump.fun

    Pump.fun bonding curve info:
    - Uses a gradual price increase based on amount raised
    - Migrates to Raydium at ~$69k market cap
    - Different instruction structure than Raydium/Jupiter
    """

    def __init__(self):
        self.pump_fun_program = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

    def parse_pumpfun_transaction(
        self,
        tx_data: Dict[str, Any],
        wallet_address: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parsea una transacción de pump.fun

        Args:
            tx_data: Transacción cruda del RPC
            wallet_address: Wallet que estamos tracking

        Returns:
            Dict con datos del trade o None si no es un swap de pump.fun
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

        # Parse instructions looking for pump.fun program
        account_keys = message.get('accountKeys', [])

        # Check inner instructions (pump.fun often appears here)
        inner_instructions = meta.get('innerInstructions', [])
        loaded_addresses = meta.get('loadedAddresses', {})
        readonly_addresses = loaded_addresses.get('readonly', [])

        # First check if pump.fun program is involved at all
        pump_fun_involved = self.pump_fun_program in account_keys or self.pump_fun_program in readonly_addresses

        if not pump_fun_involved:
            return None

        # If pump.fun is involved, parse from token balance changes
        # (pump.fun transactions can be complex with multiple wrappers)
        return self._parse_from_balances(
            account_keys,
            meta,
            wallet_address,
            timestamp,
            signature
        )

    def _parse_from_balances(
        self,
        account_keys: List[str],
        meta: Dict[str, Any],
        wallet_address: str,
        timestamp: datetime,
        signature: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parsea trades desde cambios de balance cuando pump.fun esta involucrado
        """
        pre_balances = meta.get('preTokenBalances', [])
        post_balances = meta.get('postTokenBalances', [])

        # Build map of accountIndex -> owner
        pre_owners = {}
        for acc in pre_balances:
            acc_index = acc.get('accountIndex')
            owner = acc.get('owner')
            if acc_index is not None and owner:
                pre_owners[acc_index] = owner

        # Find all token mints involved
        token_mints = {}
        for acc in pre_balances:
            acc_index = acc.get('accountIndex')
            mint = acc.get('mint')
            owner = acc.get('owner')
            if acc_index is not None and mint and owner == wallet_address:
                if mint not in token_mints:
                    token_mints[mint] = acc_index

        # Also check post balances in case of new tokens
        for acc in post_balances:
            acc_index = acc.get('accountIndex')
            mint = acc.get('mint')
            owner = acc.get('owner')
            if acc_index is not None and mint and owner == wallet_address:
                if mint not in token_mints:
                    token_mints[mint] = acc_index

        # Look for pump.fun tokens (ending in "pump") or multiple tokens changing
        trades_found = []

        for token_mint, acc_index in token_mints.items():
            # Check if this is a pump.fun token (ends with "pump")
            is_pump_token = token_mint.endswith('pump')

            # Get pre and post balance
            pre_acc = next((a for a in pre_balances if a.get('accountIndex') == acc_index), None)
            post_acc = next((a for a in post_balances if a.get('accountIndex') == acc_index), None)

            if not pre_acc or not post_acc:
                continue

            # Safely extract token amounts (handle None case)
            pre_token_amount = pre_acc.get('uiTokenAmount')
            if pre_token_amount:
                ui_amount = pre_token_amount.get('uiAmount')
                pre_amount = float(ui_amount) if ui_amount is not None else 0
            else:
                pre_amount = 0

            post_token_amount = post_acc.get('uiTokenAmount')
            if post_token_amount:
                ui_amount = post_token_amount.get('uiAmount')
                post_amount = float(ui_amount) if ui_amount is not None else 0
            else:
                post_amount = 0

            token_change = post_amount - pre_amount

            # Ignore if no change or very small change
            if abs(token_change) < 0.000001:
                continue

            # For pump.fun tokens, any significant change is likely a trade
            if is_pump_token or abs(token_change) > 1000:
                # Try to find SOL change
                sol_change = None

                # Look at native SOL balance changes
                pre_sol = meta.get('preBalances', [])
                post_sol = meta.get('postBalances', [])

                # Find wallet index in accountKeys
                try:
                    wallet_index = account_keys.index(wallet_address)
                except ValueError:
                    wallet_index = None

                if wallet_index is not None and wallet_index < len(pre_sol):
                    pre_sol_amount = pre_sol[wallet_index] / 1_000_000_000  # Convert lamports to SOL
                    post_sol_amount = post_sol[wallet_index] / 1_000_000_000
                    sol_change = post_sol_amount - pre_sol_amount

                # Determine operation
                if sol_change is not None and abs(sol_change) > 0.000001:
                    if sol_change < 0 and token_change > 0:
                        operation = 'buy'
                        amount_sol = abs(sol_change)
                        amount_token = token_change

                        # Calculate price
                        if amount_token > 0:
                            price = amount_sol / amount_token
                        else:
                            price = None

                        return {
                            'token_address': token_mint,
                            'operation': operation,
                            'amount_sol': amount_sol,
                            'amount_token': amount_token,
                            'price': price,
                            'timestamp': timestamp,
                            'tx_signature': signature,
                            'dex': 'pump_fun',
                        }

                    elif sol_change > 0 and token_change < 0:
                        operation = 'sell'
                        amount_sol = sol_change
                        amount_token = abs(token_change)

                        # Calculate price
                        if amount_token > 0:
                            price = amount_sol / amount_token
                        else:
                            price = None

                        return {
                            'token_address': token_mint,
                            'operation': operation,
                            'amount_sol': amount_sol,
                            'amount_token': amount_token,
                            'price': price,
                            'timestamp': timestamp,
                            'tx_signature': signature,
                            'dex': 'pump_fun',
                        }

        return None

    def _parse_pumpfun_instruction(
        self,
        instruction: Dict[str, Any],
        account_keys: List[str],
        meta: Dict[str, Any],
        wallet_address: str,
        timestamp: datetime,
        signature: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parsea una instrucción específica de pump.fun

        Estructura típica de una transacción de pump.fun:
        - Cuenta 0: Token mint address
        - Cuenta 1: Bonding curve account
        - Cuenta 2: User wallet
        - Datos de instrucción: Amount, operation type
        """

        try:
            # Obtener data de la instrucción
            data = instruction.get('data', [])

            # Convertir de base58 si es necesario
            if isinstance(data, str):
                try:
                    data = base58.b58decode(data)
                except:
                    data = []

            # Obtener cuentas involucradas
            accounts = [
                account_keys[instruction.get(f'account{i}Index', instruction.get('accountIndexes', [])[i])]
                for i in range(10)  # Max 10 accounts
            ]

            # Para pump.fun, las cuentas son típicamente:
            # 0: Token mint
            # 1: Bonding curve
            # 2: User wallet
            # 3-9: Otras cuentas (system, etc.)

            if len(accounts) < 3:
                return None

            token_mint = accounts[0]
            user_wallet = accounts[2]

            # Verificar que es la wallet que estamos tracking
            if user_wallet != wallet_address:
                return None

            # Obtener cambios de balance para determinar buy/sell
            pre_balances = meta.get('preTokenBalances', [])
            post_balances = meta.get('postTokenBalances', [])

            trade_info = self._extract_trade_from_balances(
                pre_balances,
                post_balances,
                user_wallet,
                token_mint
            )

            if trade_info:
                trade_info.update({
                    'timestamp': timestamp,
                    'tx_signature': signature,
                    'dex': 'pump_fun',
                })

                return trade_info

        except Exception as e:
            print(f"[!] Error parsing pump.fun instruction: {e}")

        return None

    def _extract_trade_from_balances(
        self,
        pre_balances: List[Dict],
        post_balances: List[Dict],
        wallet: str,
        token_mint: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extrae información del trade desde cambios de balance

        Para pump.fun, buscamos:
        - Disminución de SOL (compra)
        - Aumento del token pump.fun
        """
        try:
            # Buscar balance del wallet para cada token
            sol_change = None
            token_change = None

            # Balances de SOL (WSOL)
            for acc in pre_balances:
                acc_index = acc.get('accountIndex')
                mint = acc.get('mint')

                if mint == self.WSOL_MINT and acc_index is not None:
                    pre_amount = float(acc.get('uiTokenAmount', {}).get('uiAmount', 0))

                    # Encontrar balance post
                    post_acc = next(
                        (a for a in post_balances
                         if a.get('accountIndex') == acc_index),
                        None
                    )

                    if post_acc:
                        post_amount = float(post_acc.get('uiTokenAmount', {}).get('uiAmount', 0))
                        sol_change = post_amount - pre_amount

            # Balances del token pump.fun
            for acc in pre_balances:
                acc_index = acc.get('accountIndex')
                mint = acc.get('mint')

                if mint == token_mint and acc_index is not None:
                    pre_amount = float(acc.get('uiTokenAmount', {}).get('uiAmount', 0))

                    # Encontrar balance post
                    post_acc = next(
                        (a for a in post_balances
                         if a.get('accountIndex') == acc_index),
                        None
                    )

                    if post_acc:
                        post_amount = float(post_acc.get('uiTokenAmount', {}).get('uiAmount', 0))
                        token_change = post_amount - pre_amount

            # Determinar operación
            if sol_change is not None and token_change is not None:
                if sol_change < 0 and token_change > 0:
                    operation = 'buy'
                    amount_sol = abs(sol_change)
                    amount_token = token_change

                    # Calcular precio aproximado
                    if amount_token > 0:
                        price = amount_sol / amount_token
                    else:
                        price = None

                    return {
                        'token_address': token_mint,
                        'operation': operation,
                        'amount_sol': amount_sol,
                        'amount_token': amount_token,
                        'price': price,
                    }

                elif sol_change > 0 and token_change < 0:
                    operation = 'sell'
                    amount_sol = sol_change
                    amount_token = abs(token_change)

                    # Calcular precio aproximado
                    if amount_token > 0:
                        price = amount_sol / amount_token
                    else:
                        price = None

                    return {
                        'token_address': token_mint,
                        'operation': operation,
                        'amount_sol': amount_sol,
                        'amount_token': amount_token,
                        'price': price,
                    }

        except Exception as e:
            print(f"[!] Error extracting trade from balances: {e}")

        return None

    @property
    def WSOL_MINT(self):
        """WSOL Mint Address"""
        return "So11111111111111111111111111111111111111112"


def test_pumpfun_parser():
    """
    Test del parser de pump.fun
    """
    print("="*60)
    print("PUMP.FUN PARSER TEST")
    print("="*60)

    parser = PumpFunParser()

    print(f"\n[*] Pump.fun Program ID: {parser.pump_fun_program}")
    print("[*] Parser ready to detect pump.fun swaps")

    # Ejemplo de cómo se usaría:
    print("\n[*] Usage example:")
    print("    trade = parser.parse_pumpfun_transaction(tx_data, wallet_address)")
    print("    if trade:")
    print("        print(f\"Pump.fun trade: {trade['operation']} {trade['token_address']}\")")

if __name__ == "__main__":
    test_pumpfun_parser()
