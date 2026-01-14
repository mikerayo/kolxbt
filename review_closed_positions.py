#!/usr/bin/env python3
"""
Review ClosedPositions in database
"""
import os
os.environ['DATABASE_URL'] = 'postgresql://kol_tracker_db_user:quNTItA4CvEhk9KmsK1irizJQQGWu99X@dpg-d5jcq924d50c73fqo9t0-a.virginia-postgres.render.com/kol_tracker_db'

from core.database import db, ClosedPosition, KOL

def main():
    session = db.get_session()

    cps = session.query(ClosedPosition).all()

    print(f"Total ClosedPositions: {len(cps)}")
    print()

    for cp in cps:
        kol = session.query(KOL).get(cp.kol_id)
        name = kol.name if kol else 'Unknown'
        print(f"KOL: {name} | PnL: {cp.pnl_multiple:.2f}x | Hold: {cp.hold_time_seconds/3600:.1f}h | Entry: {cp.entry_time} | Exit: {cp.exit_time}")

    session.close()

if __name__ == "__main__":
    main()
