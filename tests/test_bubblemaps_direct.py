"""
Test Bubblemaps API directly with known tokens
"""
import asyncio
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from bubblemaps_api import BubblemapsAPI
from database import db, TokenInfo
from datetime import datetime


async def test_known_tokens():
    """Test Bubblemaps with known Solana tokens"""

    # Known tokens with good volume
    test_tokens = [
        "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",  # Raydium
        "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",  # Bonk
    ]

    print("=" * 70)
    print("BUBBLEMAPS DIRECT TEST")
    print("=" * 70)
    print(f"\nTesting {len(test_tokens)} known tokens...\n")

    async with BubblemapsAPI() as api:
        for i, token_addr in enumerate(test_tokens, 1):
            print(f"[{i}/{len(test_tokens)}] Testing {token_addr[:8]}...")

            # Fetch Bubblemaps data
            data = await api.get_map_data(token_addr)

            if data:
                print(f"  ✓ Success!")
                print(f"    Holders: {data.get('metrics', {}).get('holder_count', 0)}")
                print(f"    Top 10: {data.get('metrics', {}).get('top10_percentage', 0):.2f}%")
                print(f"    Risk Score: {data.get('metrics', {}).get('concentration_risk', 0):.1f}/100")
                print(f"    Gini: {data.get('metrics', {}).get('gini_coefficient', 0):.3f}")
                print(f"    Dev holdings: {data.get('metrics', {}).get('dev_percentage', 0):.2f}%")

                # Save to database
                session = db.get_session()
                try:
                    # Check if exists
                    existing = session.query(TokenInfo).filter(
                        TokenInfo.token_address == token_addr
                    ).first()

                    metrics = data.get('metrics', {})
                    token_data = {
                        'token_address': token_addr,
                        'top1_percentage': metrics.get('top1_percentage'),
                        'top10_percentage': metrics.get('top10_percentage'),
                        'top20_percentage': metrics.get('top20_percentage'),
                        'top10_retail_percentage': metrics.get('top10_retail_percentage'),
                        'gini_coefficient': metrics.get('gini_coefficient'),
                        'concentration_risk': metrics.get('concentration_risk'),
                        'holder_count': metrics.get('holder_count'),
                        'cluster_count': metrics.get('cluster_count'),
                        'supernode_count': metrics.get('supernode_count'),
                        'dev_wallet_count': metrics.get('dev_wallet_count'),
                        'dev_percentage': metrics.get('dev_percentage'),
                        'cex_percentage': metrics.get('cex_percentage'),
                        'dex_percentage': metrics.get('dex_percentage'),
                        'contract_percentage': metrics.get('contract_percentage'),
                        'largest_cluster_percentage': metrics.get('largest_cluster_percentage'),
                        'decentralization_score': metrics.get('decentralization_score'),
                        'bubblemaps_updated': datetime.now(),
                    }

                    if existing:
                        # Update
                        for key, value in token_data.items():
                            if value is not None:
                                setattr(existing, key, value)
                        print(f"    ✓ Updated in database")
                    else:
                        # Create new
                        token_info = TokenInfo(**token_data)
                        session.add(token_info)
                        print(f"    ✓ Created in database")

                    session.commit()
                except Exception as e:
                    print(f"    ✗ Error saving to DB: {e}")
                    session.rollback()
                finally:
                    session.close()
            else:
                print(f"  ✗ No data found")

            # Rate limiting
            await asyncio.sleep(1)

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

    # Show saved tokens
    session = db.get_session()
    tokens = session.query(TokenInfo).all()
    print(f"\nTokens in database: {len(tokens)}")
    for token in tokens:
        symbol = token.symbol or token.token_address[:8]
        if token.concentration_risk is not None:
            print(f"  {symbol}: Risk {token.concentration_risk:.1f}/100, "
                  f"Top10 {token.top10_percentage:.2f}%, "
                  f"Gini {token.gini_coefficient:.3f}")
    session.close()


if __name__ == "__main__":
    asyncio.run(test_known_tokens())
