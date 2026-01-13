"""
Quick test to verify the system works without full tracking
"""
import asyncio
from database import db
from wallet_tracker import WalletTracker

async def quick_test():
    print("=" * 60)
    print("KOL TRACKER ML - QUICK TEST")
    print("=" * 60)

    # Get one KOL
    session = db.get_session()
    kols = db.get_all_kols(session)
    kol = kols[0]
    session.close()

    print(f"\n[*] Testing with: {kol.name}")
    print(f"    Wallet: {kol.short_address}")
    print(f"    Twitter: @{kol.twitter_username}")

    async with WalletTracker() as tracker:
        # Get just 10 recent signatures
        print(f"\n[*] Fetching last 10 signatures...")
        sigs = await tracker.get_signatures_for_address(kol.wallet_address, limit=10)
        print(f"[+] Got {len(sigs)} signatures")

        # Get first transaction
        if sigs:
            sig = sigs[0]['signature']
            print(f"\n[*] Fetching transaction: {sig[:20]}...")
            tx = await tracker.get_transaction(sig)

            if tx:
                print("[+] Transaction fetched successfully!")

                # Try to parse it
                print(f"\n[*] Parsing transaction...")
                trade = tracker.parser.parse_transaction(tx, kol.wallet_address)

                if trade:
                    print("[✓] TRADE FOUND:")
                    print(f"    Operation: {trade['operation']}")
                    print(f"    Token: {trade['token_address'][:16]}...")
                    print(f"    Amount SOL: {trade['amount_sol']:.4f}")
                    print(f"    Timestamp: {trade['timestamp']}")
                else:
                    print("[-] No trade detected (may not be a DEX swap)")
            else:
                print("[!] Failed to fetch transaction details")

    print("\n" + "=" * 60)
    print("TEST COMPLETE - System is working!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run: python run_tracker.py --mode track --num-kols 10 --days 7")
    print("  2. Run: python run_tracker.py --mode match")
    print("  3. Run: python run_tracker.py --mode analyze")

if __name__ == "__main__":
    asyncio.run(quick_test())
