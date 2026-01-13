#!/usr/bin/env python3
"""
Detailed test - Debug pump.fun detector
"""

import asyncio
from database import db
from wallet_tracker import WalletTracker

async def debug_pumpfun_detector():
    print("="*60)
    print("PUMP.FUN DETECTOR DEBUG TEST")
    print("="*60)

    session = db.get_session()
    kols = db.get_all_kols(session)
    kol = kols[0]
    session.close()

    print(f"\n[*] KOL: {kol.name}")
    print(f"    Wallet: {kol.wallet_address}")

    async with WalletTracker() as tracker:
        # Get solo 20 transacciones para debug
        sigs = await tracker.get_signatures_for_address(kol.wallet_address, limit=20)
        print(f"\n[*] Got {len(sigs)} signatures")

        print(f"\n[*] Checking first 5 transactions in detail...")
        print("="*60)

        for i, sig_data in enumerate(sigs[:5], 1):
            sig = sig_data.get('signature', 'N/A')[:16]
            block_time = sig_data.get('blockTime', 'N/A')

            print(f"\n[{i}] Signature: {sig}...")
            print(f"    Block Time: {block_time}")

            # Fetch transaction
            tx = await tracker.get_transaction(sig_data.get('signature'))
            if not tx:
                print(f"    Status: Failed to fetch")
                continue

            result = tx.get('result')
            if not result:
                print(f"    Status: No result")
                continue

            transaction = result.get('transaction')
            if not transaction:
                print(f"    Status: No transaction")
                continue

            message = transaction.get('message', {})
            account_keys = message.get('accountKeys', [])
            instructions = message.get('instructions', [])

            print(f"    Status: Fetched")
            print(f"    Instructions: {len(instructions)}")

            # Check programs involved
            programs_involved = []
            for instr in instructions:
                prog_id = instr.get('programIdIndex')
                if prog_id is not None and prog_id < len(account_keys):
                    program_id = account_keys[prog_id]
                    programs_involved.append(program_id[:16] + "...")

            print(f"    Programs: {', '.join(set(programs_involved))}")

            # Check if pump.fun is involved
            pump_fun_program = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
            if pump_fun_program in programs_involved:
                print(f"    >>> PUMP.FUN DETECTED!")

            # Try to parse
            trade = tracker.parser.parse_transaction(tx, kol.wallet_address)
            if trade:
                print(f"    >>> TRADE DETECTED: {trade['operation']} {trade.get('dex', 'unknown')}")
            else:
                print(f"    No trade detected")

            await asyncio.sleep(0.5)

        # Ahora probar parser de pump.fun directamente
        print(f"\n" + "="*60)
        print("Testing pump.fun parser directly...")
        print("="*60)

        try:
            from pumpfun_parser import PumpFunParser
            pump_parser = PumpFunParser()

            print(f"[*] Pump.fun program ID: {pump_parser.pump_fun_program}")

            # Probar con las primeras 5 transacciones
            for i, sig_data in enumerate(sigs[:5], 1):
                sig = sig_data.get('signature')
                tx = await tracker.get_transaction(sig)

                if tx:
                    pump_trade = pump_parser.parse_pumpfun_transaction(tx, kol.wallet_address)
                    if pump_trade:
                        print(f"\n[+] PUMP.FUN TRADE FOUND (direct parse):")
                        print(f"    Operation: {pump_trade['operation']}")
                        print(f"    Token: {pump_trade['token_address'][:16]}...")
                        print(f"    Amount SOL: {pump_trade['amount_sol']:.4f}")
                    else:
                        print(f"\n[-] No pump.fun trade (sig {i})")

        except ImportError as e:
            print(f"\n[!] Error importing pumpfun_parser: {e}")

if __name__ == "__main__":
    asyncio.run(debug_pumpfun_detector())
