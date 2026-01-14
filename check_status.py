#!/usr/bin/env python3
"""
Check current status of trades in database
"""
import os
import sys

# Set DATABASE_URL BEFORE importing
os.environ['DATABASE_URL'] = 'postgresql://kol_tracker_db_user:quNTItA4CvEhk9KmsK1irizJQQGWu99X@dpg-d5jcq924d50c73fqo9t0-a.virginia-postgres.render.com/kol_tracker_db'

from core.database import db, Trade
from sqlalchemy import func

def main():
    session = db.get_session()

    print("=" * 50)
    print("ESTADO ACTUAL DE TRADES")
    print("=" * 50)

    total = session.query(Trade).count()
    buys = session.query(Trade).filter(Trade.operation == 'buy').count()
    sells = session.query(Trade).filter(Trade.operation == 'sell').count()

    print(f"\nTotal trades: {total}")
    print(f"  Compras (BUY): {buys}")
    print(f"  Ventas (SELL): {sells}")

    if total > 0:
        newest = session.query(func.max(Trade.timestamp)).scalar()
        oldest = session.query(func.min(Trade.timestamp)).scalar()

        print(f"\nRango de tiempo:")
        print(f"  Mas antiguo: {oldest}")
        print(f"  Mas reciente: {newest}")

        if buys > 0:
            buy_pct = buys/total*100
            print(f"\n[OK] FIX FUNCIONANDO!")
            print(f"     {buy_pct:.1f}% son compras")
        else:
            print(f"\n[ADVERTENCIA] Todavia sin compras detectadas")
            print(f"     Posibles causas:")
            print(f"     1. El tracker en Render no ha terminado el deploy")
            print(f"     2. El tracker no ha capturado nuevos trades aun")
            print(f"     3. Los KOLs realmente solo estan vendiendo")
    else:
        print("\n[INFO] No hay trades en la base de datos")
        print("      El tracker en Render todavia no ha capturado nada")

    session.close()

if __name__ == "__main__":
    main()
