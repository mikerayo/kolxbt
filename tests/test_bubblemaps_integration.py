"""
Test script para Bubblemaps integration

Actualiza solo 5 tokens para probar
"""

import asyncio
from datetime import datetime

from update_tokens_both import update_tokens_with_bubblemaps
from database import db


async def main():
    """Test Bubblemaps integration"""
    print("=" * 70)
    print("BUBBLEMAPS INTEGRATION TEST")
    print("=" * 70)
    print(f"\nStart time: {datetime.now()}")
    print("Actualizando 5 tokens con DexScreener + Bubblemaps\n")

    # Recrear tablas con nuevos campos
    print("[*] Recreando tablas con campos de Bubblemaps...")
    db.drop_tables()
    db.create_tables()
    print("[✓] Tablas recreadas\n")

    # Update tokens
    result = await update_tokens_with_bubblemaps(limit=5)

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)
    print(f"\nResultados:")
    print(f"  Tokens procesados: {result['processed']}")
    print(f"  DexScreener: {result['dex_found']} tokens")
    print(f"  Bubblemaps: {result['bubble_found']} tokens")
    print(f"  Creados: {result['created']}")
    print(f"  Actualizados: {result['updated']}")
    print(f"  Errores: {result['errors']}")

    # Mostrar tokens en BD
    from database import TokenInfo
    session = db.get_session()
    tokens = session.query(TokenInfo).all()

    print(f"\n[✓] Tokens en base de datos: {len(tokens)}")

    if tokens:
        print("\nTop holders distribution:")
        for token in tokens[:5]:
            symbol = token.symbol or token.token_address[:8]
            print(f"\n  {symbol}:")
            if token.top10_percentage is not None:
                print(f"    Top 10 holders: {token.top10_percentage:.2f}%")
            if token.concentration_risk is not None:
                print(f"    Riesgo concentración: {token.concentration_risk:.1f}/100")
            if token.dev_percentage is not None:
                print(f"    Dev holdings: {token.dev_percentage:.2f}%")
            if token.gini_coefficient is not None:
                print(f"    Gini coefficient: {token.gini_coefficient:.3f}")

    session.close()


if __name__ == "__main__":
    # Fix Windows encoding
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    asyncio.run(main())
