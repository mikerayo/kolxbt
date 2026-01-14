"""
Test full integration: DexScreener + Bubblemaps
"""
import asyncio
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from dexscreener_api import DexScreenerAPI
from bubblemaps_api import BubblemapsAPI
from core.database import db, TokenInfo
from datetime import datetime


async def test_full_integration():
    """Test combined DexScreener + Bubblemaps"""

    # Known tokens
    test_tokens = [
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # Bonk
    ]

    print("=" * 70)
    print("FULL INTEGRATION TEST - DexScreener + Bubblemaps")
    print("=" * 70)
    print(f"\nTesting {len(test_tokens)} tokens with both APIs...\n")

    async with DexScreenerAPI() as dex_api, BubblemapsAPI() as bubble_api:
        for token_addr in test_tokens:
            print(f"[*] Token: {token_addr[:8]}...")
            print()

            # DexScreener
            print("  [1/2] Fetching DexScreener data...")
            dex_data = await dex_api.get_token_info(token_addr)

            if dex_data:
                print(f"    ✓ Name: {dex_data.get('name')}")
                print(f"    ✓ Symbol: {dex_data.get('symbol')}")
                price = dex_data.get('price_usd')
                if price:
                    try:
                        price_float = float(price)
                        print(f"    ✓ Price: ${price_float:.8f}")
                    except (ValueError, TypeError):
                        print(f"    ✓ Price: ${price}")
                liquidity = dex_data.get('liquidity_usd')
                if liquidity:
                    print(f"    ✓ Liquidity: ${float(liquidity):,.0f}")
                volume = dex_data.get('volume_24h_usd')
                if volume:
                    print(f"    ✓ Volume 24h: ${float(volume):,.0f}")
            else:
                print("    ✗ No DexScreener data")

            print()

            # Bubblemaps
            print("  [2/2] Fetching Bubblemaps data...")
            bubble_data = await bubble_api.get_map_data(token_addr)

            if bubble_data:
                metrics = bubble_data.get('metrics', {})
                print(f"    ✓ Holders: {metrics.get('holder_count', 0)}")
                print(f"    ✓ Top 10: {metrics.get('top10_percentage', 0):.2f}%")
                print(f"    ✓ Risk Score: {metrics.get('concentration_risk', 0):.1f}/100")
                print(f"    ✓ Dev holdings: {metrics.get('dev_percentage', 0):.2f}%")
            else:
                print("    ✗ No Bubblemaps data")

            print()
            print("  [*] Saving to database...")

            # Save combined data
            session = db.get_session()
            try:
                existing = session.query(TokenInfo).filter(
                    TokenInfo.token_address == token_addr
                ).first()

                # Combine data
                token_data = {
                    'token_address': token_addr,
                    'last_updated': datetime.now(),

                    # DexScreener
                    'name': dex_data.get('name') if dex_data else None,
                    'symbol': dex_data.get('symbol') if dex_data else None,
                    'logo_url': dex_data.get('logo') if dex_data else None,
                    'price_usd': dex_data.get('price_usd') if dex_data else None,
                    'liquidity_usd': dex_data.get('liquidity_usd') if dex_data else None,
                    'fdv_usd': dex_data.get('fdv_usd') if dex_data else None,
                    'volume_24h_usd': dex_data.get('volume_24h_usd') if dex_data else None,
                    'change_24h_percent': dex_data.get('change_24h') if dex_data else None,
                    'chain_id': dex_data.get('chain_id') if dex_data else None,
                    'dex_id': dex_data.get('dex_id') if dex_data else None,
                    'pair_address': dex_data.get('pair_address') if dex_data else None,

                    # Bubblemaps
                    'top1_percentage': bubble_data.get('metrics', {}).get('top1_percentage') if bubble_data else None,
                    'top10_percentage': bubble_data.get('metrics', {}).get('top10_percentage') if bubble_data else None,
                    'top20_percentage': bubble_data.get('metrics', {}).get('top20_percentage') if bubble_data else None,
                    'top10_retail_percentage': bubble_data.get('metrics', {}).get('top10_retail_percentage') if bubble_data else None,
                    'gini_coefficient': bubble_data.get('metrics', {}).get('gini_coefficient') if bubble_data else None,
                    'concentration_risk': bubble_data.get('metrics', {}).get('concentration_risk') if bubble_data else None,
                    'holder_count': bubble_data.get('metrics', {}).get('holder_count') if bubble_data else None,
                    'cluster_count': bubble_data.get('metrics', {}).get('cluster_count') if bubble_data else None,
                    'supernode_count': bubble_data.get('metrics', {}).get('supernode_count') if bubble_data else None,
                    'dev_wallet_count': bubble_data.get('metrics', {}).get('dev_wallet_count') if bubble_data else None,
                    'dev_percentage': bubble_data.get('metrics', {}).get('dev_percentage') if bubble_data else None,
                    'cex_percentage': bubble_data.get('metrics', {}).get('cex_percentage') if bubble_data else None,
                    'dex_percentage': bubble_data.get('metrics', {}).get('dex_percentage') if bubble_data else None,
                    'contract_percentage': bubble_data.get('metrics', {}).get('contract_percentage') if bubble_data else None,
                    'largest_cluster_percentage': bubble_data.get('metrics', {}).get('largest_cluster_percentage') if bubble_data else None,
                    'decentralization_score': bubble_data.get('metrics', {}).get('decentralization_score') if bubble_data else None,
                    'bubblemaps_updated': datetime.now() if bubble_data else None,
                }

                # Remove None values
                token_data = {k: v for k, v in token_data.items() if v is not None}

                if existing:
                    for key, value in token_data.items():
                        setattr(existing, key, value)
                    print("    ✓ Updated existing record")
                else:
                    token_info = TokenInfo(**token_data)
                    session.add(token_info)
                    print("    ✓ Created new record")

                session.commit()
            except Exception as e:
                print(f"    ✗ Error: {e}")
                session.rollback()
            finally:
                session.close()

    print()
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    # Show final result
    session = db.get_session()
    token = session.query(TokenInfo).filter(
        TokenInfo.token_address == test_tokens[0]
    ).first()

    if token:
        print()
        print(f"Token: {token.symbol} ({token.name})")
        print()
        print("DexScreener Data:")
        if token.price_usd:
            print(f"  Price: ${float(token.price_usd):.8f}")
        if token.liquidity_usd:
            print(f"  Liquidity: ${float(token.liquidity_usd):,.0f}")
        if token.volume_24h_usd:
            print(f"  Volume 24h: ${float(token.volume_24h_usd):,.0f}")
        if token.change_24h_percent is not None:
            print(f"  Change 24h: {token.change_24h_percent:+.2f}%")
        print()
        print("Bubblemaps Data:")
        if token.holder_count is not None:
            print(f"  Holders: {token.holder_count}")
        if token.top10_percentage is not None:
            print(f"  Top 10: {token.top10_percentage:.2f}%")
        if token.top1_percentage is not None:
            print(f"  Top 1: {token.top1_percentage:.2f}%")
        if token.gini_coefficient is not None:
            print(f"  Gini Coefficient: {token.gini_coefficient:.3f}")
        if token.concentration_risk is not None:
            print(f"  Concentration Risk: {token.concentration_risk:.1f}/100")
        if token.dev_percentage is not None:
            print(f"  Dev Holdings: {token.dev_percentage:.2f}%")
        if token.cex_percentage is not None:
            print(f"  CEX Holdings: {token.cex_percentage:.2f}%")
        if token.contract_percentage is not None:
            print(f"  Contract Holdings: {token.contract_percentage:.2f}%")
        if token.cluster_count is not None:
            print(f"  Clusters: {token.cluster_count}")
        if token.dev_wallet_count is not None:
            print(f"  Dev Wallets: {token.dev_wallet_count}")

    session.close()
    print()


if __name__ == "__main__":
    asyncio.run(test_full_integration())
