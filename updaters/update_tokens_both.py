"""
Actualiza metadata de tokens usando DexScreener + Bubblemaps APIs

Combina datos de:
- DexScreener: Precio, volumen, liquidez
- Bubblemaps: Distribución de holders, clusters, riesgo

Ejecuta cada 35 minutos en segundo plano.
"""

import asyncio
import sys
import io
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from apis.dexscreener_api import DexScreenerAPI
from apis.bubblemaps_api import BubblemapsAPI
from core.database import db, Trade, TokenInfo
from sqlalchemy import func
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('token_updater.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def update_tokens_with_bubblemaps(limit=50):
    """
    Actualiza tokens con datos de DexScreener Y Bubblemaps

    Args:
        limit: Número de tokens a actualizar

    Returns:
        Dict con resultados
    """
    logger.info(f"Actualizando top {limit} tokens más tradeados...")
    logger.info("Obteniendo datos de DexScreener + Bubblemaps")

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
        ).limit(limit).all()

        token_addresses = [t[0] for t in result]
        logger.info(f"Obteniendo info de {len(token_addresses)} tokens...")

        # Fetch from both APIs concurrently
        async with DexScreenerAPI() as dex_api, BubblemapsAPI() as bubble_api:
            # Fetch DexScreener data
            logger.info("[1/2] Obteniendo datos de DexScreener...")
            dex_results = await dex_api.get_multiple_tokens(token_addresses)

            # Fetch Bubblemaps data (with rate limiting)
            logger.info("[2/2] Obteniendo datos de Bubblemaps...")
            bubble_results = {}
            for i, token_addr in enumerate(token_addresses):
                logger.debug(f"  [{i+1}/{len(token_addresses)}] {token_addr[:8]}...")
                data = await bubble_api.get_map_data(token_addr)
                bubble_results[token_addr] = data

                # Rate limiting: 1 second between requests
                await asyncio.sleep(1)

        # Update database
        updated = 0
        created = 0
        errors = 0
        dex_found = 0
        bubble_found = 0

        for token_addr in token_addresses:
            try:
                # Check if already exists
                existing = session.query(TokenInfo).filter(
                    TokenInfo.token_address == token_addr
                ).first()

                # Get DexScreener data
                dex_info = dex_results.get(token_addr)
                if dex_info:
                    dex_found += 1

                # Get Bubblemaps data
                bubble_data = bubble_results.get(token_addr)
                if bubble_data:
                    bubble_found += 1

                # Prepare data
                token_data = {
                    # DexScreener data
                    'name': dex_info.get('name') if dex_info else None,
                    'symbol': dex_info.get('symbol') if dex_info else None,
                    'logo_url': dex_info.get('logo') if dex_info else None,
                    'price_usd': dex_info.get('price_usd') if dex_info else None,
                    'liquidity_usd': dex_info.get('liquidity_usd') if dex_info else None,
                    'fdv_usd': dex_info.get('fdv_usd') if dex_info else None,
                    'volume_24h_usd': dex_info.get('volume_24h_usd') if dex_info else None,
                    'change_24h_percent': dex_info.get('change_24h') if dex_info else None,
                    'chain_id': dex_info.get('chain_id') if dex_info else None,
                    'dex_id': dex_info.get('dex_id') if dex_info else None,
                    'pair_address': dex_info.get('pair_address') if dex_info else None,
                    'last_updated': datetime.now(),

                    # Bubblemaps data
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
                    'bubblemaps_updated': datetime.now(),
                }

                # Remove None values
                token_data = {k: v for k, v in token_data.items() if v is not None}

                if existing:
                    # Update
                    for key, value in token_data.items():
                        setattr(existing, key, value)
                    updated += 1
                else:
                    # Create new
                    token_info = TokenInfo(
                        token_address=token_addr,
                        **token_data
                    )
                    session.add(token_info)
                    created += 1

                # Show summary for each token
                symbol = token_data.get('symbol') or token_addr[:8]
                price = token_data.get('price_usd')
                concentration = token_data.get('concentration_risk')

                logger.info(f"  [{symbol}] Precio: ${price:.8f} | Riesgo: {concentration:.1f}/100" if price else f"  [{symbol}] Riesgo: {concentration:.1f}/100")

            except Exception as e:
                logger.error(f"Error procesando {token_addr[:8]}: {e}")
                errors += 1

        session.commit()

        # Summary
        logger.info("=" * 70)
        logger.info("RESUMEN DE ACTUALIZACION")
        logger.info("=" * 70)
        logger.info(f"Tokens procesados: {len(token_addresses)}")
        logger.info(f"DexScreener encontrados: {dex_found}")
        logger.info(f"Bubblemaps encontrados: {bubble_found}")
        logger.info(f"Creados: {created}")
        logger.info(f"Actualizados: {updated}")
        logger.info(f"Errores: {errors}")

        return {
            'processed': len(token_addresses),
            'dex_found': dex_found,
            'bubble_found': bubble_found,
            'created': created,
            'updated': updated,
            'errors': errors
        }

    finally:
        session.close()


async def main():
    """Ejecuta actualizador de tokens continuamente"""
    running = True

    logger.info("=" * 70)
    logger.info("TOKEN UPDATER - DexScreener + Bubblemaps")
    logger.info("=" * 70)
    logger.info(f"Start time: {datetime.now()}")
    logger.info(f"Update interval: 35 minutes")
    logger.info(f"Tokens per update: 50")
    logger.info("=" * 70)
    logger.info("Starting continuous token update...")
    logger.info("Press Ctrl+C to stop")

    interval_minutes = 35
    interval_seconds = interval_minutes * 60

    while running:
        try:
            logger.info("=" * 70)
            logger.info(f"Starting Token Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)

            # Actualizar tokens
            result = await update_tokens_with_bubblemaps(limit=50)

            logger.info(f"Update completed: {result['dex_found']} DexScreener, {result['bubble_found']} Bubblemaps, {result['created']} created, {result['updated']} updated, {result['errors']} errors")

            if running:
                # Calcular próxima ejecución
                from datetime import timedelta
                next_run = datetime.now() + timedelta(minutes=interval_minutes)
                logger.info(f"Next update in {interval_minutes} minutes (at {next_run.strftime('%Y-%m-%d %H:%M')})")

                # Esperar intervalo
                await asyncio.sleep(interval_seconds)

        except Exception as e:
            logger.error(f"Error en actualizador: {e}")
            if running:
                logger.info("Reintentando en 5 minutos...")
                await asyncio.sleep(300)

    logger.info("Token updater stopped")


if __name__ == "__main__":
    # Fix Windows encoding
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    asyncio.run(main())
