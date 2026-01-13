"""
Token-Centric Discovery - Descubre wallets analizando holders de tokens exitosos

Estrategia brillante:
1. KOLs compran tokens
2. Algunos tokens hacen 3x+, 10x+
3. Los holders de esos tokens exitosos son probablemente traders buenos
4. Obtenemos top holders de tokens exitosos
5. Analizamos su performance y los añadimos al tracking

Ventajas:
- No necesita parsear transacciones complejas
- Holders son estáticos y fáciles de obtener con RPC
- Token performance es un proxy de "qué tan bueno es este trade"
"""

import asyncio
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy import func
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database import db, KOL, Trade, ClosedPosition, DiscoveredTrader


class TokenPerformanceTracker:
    """
    Trackea el performance de tokens comprados por KOLs
    """

    def __init__(self):
        self.session = db.get_session()

    def get_recent_kol_tokens(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """
        Obtiene tokens comprados recientemente por KOLs

        Args:
            hours: Horas hacia atrás
            limit: Máximo de tokens a retornar

        Returns:
            List of dicts with token info
        """
        since = datetime.now() - timedelta(hours=hours)

        # Get recent buy trades grouped by token
        recent_buys = self.session.query(
            Trade.token_address,
            Trade.amount_sol,
            Trade.timestamp,
            KOL.name,
            KOL.id
        ).join(
            KOL, Trade.kol_id == KOL.id
        ).filter(
            Trade.operation == 'buy',
            Trade.timestamp >= since
        ).order_by(Trade.timestamp.desc()).all()

        # Group by token
        token_info = {}

        for token_addr, amount_sol, timestamp, kol_name, kol_id in recent_buys:
            if token_addr not in token_info:
                token_info[token_addr] = {
                    'token_address': token_addr,
                    'total_sol_invested': 0,
                    'num_kols_bought': 0,
                    'kol_names': set(),
                    'first_buy_time': timestamp,
                    'kol_ids': set()
                }

            token_info[token_addr]['total_sol_invested'] += amount_sol
            token_info[token_addr]['num_kols_bought'] += 1
            token_info[token_addr]['kol_names'].add(kol_name)
            token_info[token_addr]['kol_ids'].add(kol_id)

        # Convert to list and sort by SOL invested
        tokens_list = []
        for token_addr, info in token_info.items():
            info['kol_names'] = list(info['kol_names'])
            info['kol_ids'] = list(info['kol_ids'])
            tokens_list.append(info)

        tokens_list.sort(key=lambda x: x['total_sol_invested'], reverse=True)

        return tokens_list[:limit]

    def estimate_token_performance(self, token_address: str) -> Optional[Dict]:
        """
        Estima el performance actual de un token

        Busca closed positions para este token y calcula múltiplos promedio

        Args:
            token_address: Token mint address

        Returns:
            Dict with performance metrics
        """
        # Get closed positions for this token
        closed_positions = self.session.query(ClosedPosition).filter(
            ClosedPosition.token_address == token_address
        ).all()

        if not closed_positions:
            return None

        # Calculate metrics
        profitable = sum(1 for p in closed_positions if p.pnl_multiple > 1)
        three_x_plus = sum(1 for p in closed_positions if p.pnl_multiple >= 3)
        ten_x_plus = sum(1 for p in closed_positions if p.pnl_multiple >= 10)

        avg_multiple = sum(p.pnl_multiple for p in closed_positions) / len(closed_positions)

        return {
            'num_positions': len(closed_positions),
            'num_profitable': profitable,
            'num_3x_plus': three_x_plus,
            'num_10x_plus': ten_x_plus,
            'avg_multiple': avg_multiple,
            'win_rate': profitable / len(closed_positions) if closed_positions else 0
        }


class TokenHolderDiscovery:
    """
    Descubre nuevos wallets analizando holders de tokens exitosos
    """

    def __init__(self):
        self.session = db.get_session()

    async def get_token_holders(self, token_address: str, limit: int = 20) -> List[Dict]:
        """
        Obtiene los principales holders de un token usando RPC

        Args:
            token_address: Token mint address
            limit: Máximo de holders a obtener

        Returns:
            List of dicts with holder info
        """
        from core.wallet_tracker import WalletTracker

        tracker = WalletTracker()

        try:
            async with tracker:
                # En Solana, usamos getTokenLargestAccounts para obtener holders
                # Este es el método estándar del RPC
                response = await tracker._rpc_call("getTokenLargestAccounts", [token_address])

                if not response or 'result' not in response:
                    return []

                # getTokenLargestAccounts returns token accounts, not wallets
                # We need to get the owner of each token account
                token_accounts = response['result'].get('value', [])

                if not token_accounts:
                    return []

                # Get account info for each token account to find owners
                holders = []

                for acc in token_accounts[:limit]:
                    token_account_address = acc.get('address')
                    amount = acc.get('amount', 0)
                    decimals = acc.get('decimals', 6)
                    ui_amount = acc.get('uiAmount', 0)

                    if not token_account_address:
                        continue

                    # Get account info to find the owner (wallet)
                    account_info = await tracker._rpc_call("getAccountInfo", [token_account_address, {"encoding": "jsonParsed"}])

                    if account_info and 'result' in account_info:
                        result = account_info['result']
                        if 'value' in result and result['value']:
                            # In jsonParsed format, the owner is at result.value.owner
                            # But we need the actual token owner, which is in the parsed data
                            value_data = result['value']

                            # Try to get owner from different possible locations
                            owner = None

                            # Method 1: Direct owner field (for Token Account)
                            if 'owner' in value_data:
                                owner = value_data['owner']

                            # Method 2: Parsed data for Token Account
                            if 'data' in value_data and 'parsed' in value_data['data']:
                                parsed = value_data['data']['parsed']
                                if 'info' in parsed and 'owner' in parsed['info']:
                                    owner = parsed['info']['owner']

                            # Skip if it's the Token Program itself (system accounts)
                            if owner and 'Token' not in owner and 'TokenzQd' not in owner and 'System' not in owner:
                                holders.append({
                                    'address': owner,  # This is the wallet address
                                    'balance': int(amount) if amount else 0,
                                    'balance_tokens': ui_amount if ui_amount else 0
                                })

                return holders

        except Exception as e:
            print(f"[!] Error getting holders for {token_address[:8]}: {e}")
            return []

    async def discover_from_performing_tokens(self, min_kols: int = 2, min_multiple: float = 2.0) -> Dict:
        """
        Descubre nuevos traders analizando holders de tokens que están funcionando bien

        Args:
            min_kols: Mínimo de KOLs que deben haber comprado el token
            min_multiple: Mínimo múltiplo promedio para considerar el token "exitoso"

        Returns:
            Dict con resultados
        """
        print("=" * 70)
        print("TOKEN-CENTRIC DISCOVERY")
        print("=" * 70)
        print(f"\nBuscando tokens comprados por >= {min_kols} KOLs")
        print(f"Y que estén haciendo >= {min_multiple}x...\n")

        # Get token performance tracker
        perf_tracker = TokenPerformanceTracker()

        # Get recent tokens
        recent_tokens = perf_tracker.get_recent_kol_tokens(hours=48, limit=50)

        print(f"[*] Analizando {len(recent_tokens)} tokens recientes...")

        performing_tokens = []

        for i, token_info in enumerate(recent_tokens, 1):
            token_addr = token_info['token_address']
            num_kols = token_info['num_kols_bought']

            print(f"\n[{i}/{len(recent_tokens)}] {token_addr[:8]}...{token_addr[-4:]}")
            print(f"    KOLs que compraron: {num_kols}")
            print(f"    SOL invertido: {token_info['total_sol_invested']:.2f}")

            # Skip if not enough KOLs bought it
            if num_kols < min_kols:
                print(f"    [i] Skip: menos de {min_kols} KOLs")
                continue

            # Get performance
            perf = perf_tracker.estimate_token_performance(token_addr)

            if perf:
                print(f"    Performance:")
                print(f"      Posiciones: {perf['num_positions']}")
                print(f"      Win rate: {perf['win_rate']:.1%}")
                print(f"      3x+: {perf['num_3x_plus']}")
                print(f"      Promedio: {perf['avg_multiple']:.2f}x")

                # Check if it's performing well
                if perf['avg_multiple'] >= min_multiple:
                    print(f"    [+] TOKEN EXITOSO! Obteniendo holders...")
                    performing_tokens.append({
                        **token_info,
                        'performance': perf
                    })
                else:
                    print(f"    [i] Skip: rendimiento {perf['avg_multiple']:.2f}x < {min_multiple}x")
            else:
                print(f"    [i] No hay datos de performance aún")

        # Now get holders for performing tokens
        print("\n" + "=" * 70)
        print("OBTENIENDO HOLDERS DE TOKENS EXITOSOS")
        print("=" * 70)

        discovered_wallets = []

        for i, token_info in enumerate(performing_tokens[:3], 1):  # Limit to 3 tokens
            token_addr = token_info['token_address']

            print(f"\n[{i}/{len(performing_tokens)}] Token: {token_addr[:8]}...")
            print(f"    Performance: {token_info['performance']['avg_multiple']:.2f}x promedio")

            holders = await self.get_token_holders(token_addr, limit=10)

            if holders:
                print(f"    [+] {len(holders)} holders encontrados")

                for holder in holders[:5]:  # Top 5 holders
                    holder_addr = holder['address']
                    balance = holder['balance_tokens']

                    print(f"      - {holder_addr[:8]}...{holder_addr[-8:]} ({balance:,.0f} tokens)")

                    # Check if already a KOL
                    existing = self.session.query(KOL).filter(
                        KOL.wallet_address == holder_addr
                    ).first()

                    if existing:
                        print(f"        [~] Ya es KOL: {existing.name}")
                    else:
                        # Check if already discovered
                        existing_discovered = self.session.query(DiscoveredTrader).filter(
                            DiscoveredTrader.wallet_address == holder_addr
                        ).first()

                        if existing_discovered:
                            # Update discovery score
                            existing_discovered.discovery_score += 10  # Boost score for being found again
                            existing_discovered.last_activity_at = datetime.now()
                            print(f"        [~] Ya descubierto previamente (score: {existing_discovered.discovery_score:.0f})")
                        else:
                            print(f"        [+] NUEVA WALLET interesante!")

                            # Save to database
                            discovered = DiscoveredTrader(
                                wallet_address=holder_addr,
                                discovered_from_token=token_addr,
                                discovered_from_kol_id=None,  # Multiple KOLs bought this token
                                discovered_at=datetime.now(),
                                total_trades=0,  # Will be analyzed later
                                total_volume_sol=0,
                                win_rate=0,
                                three_x_rate=0,
                                avg_hold_time_hours=0,
                                total_pnl_sol=0,
                                discovery_score=50 + (token_info['performance']['avg_multiple'] * 10),  # Base score + token performance
                                is_tracking=False,
                                promoted_to_kol=False,
                                last_activity_at=datetime.now()
                            )

                            self.session.add(discovered)
                            self.session.commit()

                            discovered_wallets.append({
                                'wallet': holder_addr,
                                'balance_tokens': balance,
                                'token_source': token_addr,
                                'token_performance': token_info['performance']['avg_multiple']
                            })
            else:
                print(f"    [!] No se pudieron obtener holders")

        print("\n" + "=" * 70)
        print("RESUMEN")
        print("=" * 70)
        print(f"Tokens analizados: {len(recent_tokens)}")
        print(f"Tokens exitosos: {len(performing_tokens)}")
        print(f"Nuevas wallets descubiertas: {len(discovered_wallets)}")

        return {
            'tokens_analyzed': len(recent_tokens),
            'performing_tokens': len(performing_tokens),
            'wallets_discovered': len(discovered_wallets),
            'wallets': discovered_wallets
        }


def main():
    """Test token-centric discovery"""
    discovery = TokenHolderDiscovery()
    asyncio.run(discovery.discover_from_performing_tokens(min_kols=5, min_multiple=1.2))


if __name__ == "__main__":
    main()
