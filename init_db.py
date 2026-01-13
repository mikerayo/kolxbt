#!/usr/bin/env python3
"""
Initialize database tables for KOL Tracker ML
Run this before starting any worker
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.database import Database
from core.config import KOLS_DATA_FILE

def main():
    """Initialize database"""
    print("=" * 70)
    print("INITIALIZING DATABASE")
    print("=" * 70)

    # Create database instance
    db = Database()

    # Create tables
    print("\n[*] Creating database tables...")
    db.create_tables()

    # Load KOLs from JSON if exists
    print(f"\n[*] Checking for KOLs data at: {KOLS_DATA_FILE}")
    if KOLS_DATA_FILE.exists():
        loaded = db.load_kols_from_json()
        if loaded > 0:
            print(f"[+] Loaded {loaded} KOLs from JSON")
        else:
            print("[i] No new KOLs loaded (may already exist)")
    else:
        print(f"[i] KOLs file not found at {KOLS_DATA_FILE}")
        print("[i] Database initialized but empty")

    # Verify
    session = db.get_session()
    from core.database import KOL
    kol_count = session.query(KOL).count()
    session.close()

    print(f"\n[+] Database ready!")
    print(f"    Total KOLs: {kol_count}")
    print("\n" + "=" * 70)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
