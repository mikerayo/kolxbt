"""
DexScreener API Integration - Obtiene información de tokens

API Docs: https://docs.dexscreener.com/api/reference
Endpoints:
- /token-v2/{token_address} - Info de un token específico
- /token-v2/{token1},{token2},... - Info de múltiples tokens (máx 30)
"""

import aiohttp
import asyncio
from typing import List, Dict, Optional
from datetime import datetime, timedelta


class DexScreenerAPI:
    """
    Cliente para la API de DexScreener
    """

    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest"
        self.session = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def get_token_info(self, token_address: str) -> Optional[Dict]:
        """
        Obtiene información completa de un token

        Args:
            token_address: Contract address del token

        Returns:
            Dict con info del token o None si error
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with DexScreenerAPI()'")

        url = f"{self.base_url}/dex/tokens/{token_address}"

        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()

                    if data and 'pairs' in data and len(data['pairs']) > 0:
                        # Get the first pair (usually the most liquid)
                        pair = data['pairs'][0]

                        # Extraer información del token
                        token_info = {
                            'token_address': token_address,
                            'chain_id': pair.get('chainId'),
                            'dex_id': pair.get('dexId'),
                            'pair_address': pair.get('pairAddress'),
                            'base_token': {
                                'address': pair['baseToken']['address'],
                                'name': pair['baseToken'].get('name'),
                                'symbol': pair['baseToken'].get('symbol'),
                                'logo': pair['baseToken'].get('logo') or pair['baseToken'].get('info', {}).get('imageUrl')
                            } if pair.get('baseToken') else None,
                            'quote_token': {
                                'address': pair['quoteToken']['address'],
                                'name': pair['quoteToken'].get('name'),
                                'symbol': pair['quoteToken'].get('symbol')
                            } if pair.get('quoteToken') else None,
                            'price_usd': pair.get('priceUsd'),
                            'price_native': pair.get('priceNative'),
                            'liquidity_usd': pair.get('liquidity', {}).get('usd'),
                            'fdv_usd': pair.get('fdv'),
                            'market_cap_usd': pair.get('marketCap'),
                            'pair_created_at': pair.get('pairCreatedAt'),
                            'txns': pair.get('txns', {}),
                            'volume_24h_usd': pair.get('volume', {}).get('h24') if pair.get('volume') else None,
                            'change_24h': pair.get('change', {}).get('h24') if pair.get('change') else None,
                            'boosts': pair.get('boosts', {}).get('total'),
                            'website': pair.get('info', {}).get('websites', [{}])[0].get('url') if pair.get('info', {}).get('websites') else None,
                            'socials': pair.get('info', {}).get('socials', []),
                            'last_updated': datetime.now()
                        }

                        return token_info

                return None

        except asyncio.TimeoutError:
            print(f"[!] Timeout fetching token {token_address[:8]}...")
            return None
        except Exception as e:
            print(f"[!] Error fetching token {token_address[:8]}: {e}")
            return None

    async def get_multiple_tokens(self, token_addresses: List[str]) -> Dict[str, Optional[Dict]]:
        """
        Obtiene información de múltiples tokens (máx 30 por llamada)

        Args:
            token_addresses: Lista de contract addresses

        Returns:
            Dict mapeando address -> token_info
        """
        # DexScreener limita a 30 tokens por llamada
        batch_size = 30
        results = {}

        for i in range(0, len(token_addresses), batch_size):
            batch = token_addresses[i:i+batch_size]
            tokens_str = ','.join(batch)

            url = f"{self.base_url}/dex/tokens/{tokens_str}"

            try:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data and 'pairs' in data:
                            # Process each pair
                            for pair in data['pairs']:
                                token_addr = pair['baseToken']['address']

                                # Skip if we already have info for this token
                                if token_addr in results:
                                    continue

                                token_info = {
                                    'token_address': token_addr,
                                    'chain_id': pair.get('chainId'),
                                    'dex_id': pair.get('dexId'),
                                    'pair_address': pair.get('pairAddress'),
                                    'name': pair['baseToken'].get('name'),
                                    'symbol': pair['baseToken'].get('symbol'),
                                    'logo': pair['baseToken'].get('logo') or pair['baseToken'].get('info', {}).get('imageUrl'),
                                    'price_usd': pair.get('priceUsd'),
                                    'liquidity_usd': pair.get('liquidity', {}).get('usd'),
                                    'fdv_usd': pair.get('fdv'),
                                    'volume_24h_usd': pair.get('volume', {}).get('h24') if pair.get('volume') else None,
                                    'change_24h': pair.get('change', {}).get('h24') if pair.get('change') else None,
                                    'last_updated': datetime.now()
                                }

                                results[token_addr] = token_info

                        # Mark tokens not found as None
                        for addr in batch:
                            if addr not in results:
                                results[addr] = None

            except Exception as e:
                print(f"[!] Error fetching batch: {e}")
                # Mark all as None
                for addr in batch:
                    if addr not in results:
                        results[addr] = None

        return results

    async def search_token(self, query: str) -> List[Dict]:
        """
        Busca tokens por nombre o símbolo

        Args:
            query: Nombre o símbolo del token

        Returns:
            Lista de tokens que coinciden
        """
        url = f"{self.base_url}/dex/search/{query}"

        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()

                    if data and 'pairs' in data:
                        results = []
                        for pair in data['pairs'][:20]:  # Limit to 20 results
                            results.append({
                                'token_address': pair['baseToken']['address'],
                                'chain_id': pair.get('chainId'),
                                'dex_id': pair.get('dexId'),
                                'name': pair['baseToken'].get('name'),
                                'symbol': pair['baseToken'].get('symbol'),
                                'logo': pair['baseToken'].get('logo'),
                                'price_usd': pair.get('priceUsd'),
                                'liquidity_usd': pair.get('liquidity', {}).get('usd')
                            })
                        return results

                return []

        except Exception as e:
            print(f"[!] Error searching for '{query}': {e}")
            return []


async def update_token_metadata():
    """
    Actualiza metadata de tokens en la base de datos usando DexScreener
    """
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    from core.database import db, Trade, TokenInfo
    from sqlalchemy import func

    print("=" * 70)
    print("ACTUALIZANDO METADATA DE TOKENS - DexScreener API")
    print("=" * 70)

    session = db.get_session()

    try:
        # Get unique tokens from trades
        result = session.query(
            Trade.token_address,
            func.count(Trade.token_address).label('trade_count')
        ).group_by(
            Trade.token_address
        ).order_by(
            func.count(Trade.token_address).desc()
        ).limit(100).all()  # Start with top 100 most traded tokens

        token_addresses = [t[0] for t in result]
        print(f"\n[*] Obteniendo info de {len(token_addresses)} tokens más tradeados...")

        # Fetch token info from DexScreener
        async with DexScreenerAPI() as api:
            results = await api.get_multiple_tokens(token_addresses)

        # Update database
        updated = 0
        created = 0
        errors = 0

        for token_addr, info in results.items():
            if not info:
                errors += 1
                continue

            # Check if already exists
            existing = session.query(TokenInfo).filter(
                TokenInfo.token_address == token_addr
            ).first()

            if existing:
                # Update
                existing.name = info.get('name')
                existing.symbol = info.get('symbol')
                existing.logo_url = info.get('logo')
                existing.price_usd = info.get('price_usd')
                existing.liquidity_usd = info.get('liquidity_usd')
                existing.fdv_usd = info.get('fdv_usd')
                existing.volume_24h_usd = info.get('volume_24h_usd')
                existing.change_24h_percent = info.get('change_24h')
                existing.last_updated = info.get('last_updated')
                updated += 1
            else:
                # Create new
                token_info = TokenInfo(
                    token_address=token_addr,
                    name=info.get('name'),
                    symbol=info.get('symbol'),
                    logo_url=info.get('logo'),
                    price_usd=info.get('price_usd'),
                    liquidity_usd=info.get('liquidity_usd'),
                    fdv_usd=info.get('fdv_usd'),
                    volume_24h_usd=info.get('volume_24h_usd'),
                    change_24h_percent=info.get('change_24h'),
                    chain_id=info.get('chain_id'),
                    dex_id=info.get('dex_id'),
                    last_updated=info.get('last_updated')
                )
                session.add(token_info)
                created += 1

        session.commit()

        print(f"\n[✓] Actualización completada:")
        print(f"    Creados: {created}")
        print(f"    Actualizados: {updated}")
        print(f"    Errores: {errors}")

    finally:
        session.close()


def main():
    """Test the DexScreener API"""
    import asyncio

    async def test():
        async with DexScreenerAPI() as api:
            # Test with a known token
            print("Testing DexScreener API...")
            print("\n[*] Fetching info for a token...")

            # Test with pump.fun tokens
            test_token = "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"  # Example

            info = await api.get_token_info(test_token)

            if info:
                print("\n[✓] Token info:")
                print(f"    Nombre: {info.get('base_token', {}).get('name')}")
                print(f"    Símbolo: {info.get('base_token', {}).get('symbol')}")
                print(f"    Logo: {info.get('base_token', {}).get('logo')}")
                print(f"    Precio USD: ${info.get('price_usd', 0):.6f}" if info.get('price_usd') else "    Precio USD: N/A")
                print(f"    Liquidez: ${info.get('liquidity_usd', 0):,.0f}" if info.get('liquidity_usd') else "    Liquidez: N/A")
                print(f"    FDV: ${info.get('fdv_usd', 0):,.0f}" if info.get('fdv_usd') else "    FDV: N/A")
            else:
                print("[!] No se pudo obtener info del token")

    asyncio.run(test())


if __name__ == "__main__":
    main()
