# Enabling Real Trading — Safety Checklist

This file provides a concise checklist to safely enable real order execution
from the repository's autotrader. Follow these steps carefully and test in a
sandbox/testnet environment before using production keys.

1) Use testnet / sandbox accounts
   - Create a testnet account (Binance or exchange-provided sandbox) and generate API keys restricted to the actions you need (orders + read balances).

2) Limit exposure
   - Start with `--max-symbols 1` and `--notional-usd 10` to limit order sizes.

3) Configure credentials
   - Set API keys in environment or config (see `config/.env.example`). The exchange adapter resolves credentials from env/config.

4) Enable real orders explicitly (two steps required)
   - Run the autotrader with --no-dry-run to disable simulation.
   - Set AUTOTRADER_ALLOW_REAL_ORDERS=1 in the environment.

   Example (PowerShell):

   ```powershell
   $env:AUTOTRADER_ALLOW_REAL_ORDERS = '1'
   python scripts/autotrader.py --once --no-dry-run --max-symbols 1 --notional-usd 10
   ```

5) Pre-flight checks
   - Verify balances: `python -c "from backend.utils.exchanges import get_exchange_client; print(get_exchange_client().spot_balance())"`
   - Run the autotrader in simulated mode first and inspect `trade_logs`.

6) Have an immediate rollback plan
   - Keep credentials revocation steps handy and monitor exchange dashboards.

7) Audit logs
   - Review `trade_logs` table and exchange order history after any real order run.

Remember: real money is at risk. Use extreme caution and test thoroughly.
