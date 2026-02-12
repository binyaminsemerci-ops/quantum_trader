# Binance Testnet API Setup Instructions

**Date**: February 10, 2026  
**Current Error**: `-2015: Invalid API-key, IP, or permissions for action`

---

## 🔴 Problem Diagnosis

Testing shows that the current API keys in `/home/qt/.env` return error `-2015` for **ALL** endpoints (spot AND futures), meaning:

1. ❌ Keys are invalid/expired/revoked
2. ❌ IP `46.224.116.254` is not whitelisted (if IP restriction enabled)
3. ❌ Keys were not generated from correct testnet portal

---

## ✅ CORRECT Binance Futures Testnet Setup

### Step 1: Access Binance Futures TESTNET Portal

**URL**: https://testnet.binancefuture.com/

**IMPORTANT**: This is NOT the same as:
- ❌ https://www.binance.com/ (production)
- ❌ https://testnet.binance.vision/ (spot testnet)
- ✅ https://testnet.binancefuture.com/ (CORRECT for futures testnet)

### Step 2: Generate TESTNET API Keys

1. **Login/Register** at https://testnet.binancefuture.com/
   - Use any email (testnet doesn't require real verification)
   - Get free testnet USDT

2. **Navigate to API Management**
   - Top right → Account → API Management
   - Or direct: https://testnet.binancefuture.com/en/futures/BTCUSDT (then access API settings)

3. **Create New API Key**
   - Click "Create API"
   - Give it a label (e.g., "Quantum Trader BSC")
   - **SAVE THE KEYS IMMEDIATELY** (they're only shown once)

4. **Configure API Permissions**
   - ✅ **Enable Futures** (CRITICAL)
   - ✅ **Enable Reading** (required for position polling)
   - ❌ **Spot Trading** (not needed)
   - ❌ **Enable Withdrawals** (NEVER enable)

5. **Configure IP Whitelist**
   
   **Option A: Unrestricted (easier for testing)**
   - Leave "Restrict access to trusted IPs only" **UNCHECKED**
   - ⚠️ Less secure, but works immediately
   
   **Option B: IP Restricted (recommended for production)**
   - Check "Restrict access to trusted IPs only"
   - Add IP: `46.224.116.254` (VPS IP)
   - Save

### Step 3: Update VPS Environment Variables

**Method 1: Direct Edit (if you have SSH access)**
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Edit .env file
nano /home/qt/.env

# Replace these lines:
BINANCE_API_KEY=<YOUR_NEW_TESTNET_KEY>
BINANCE_API_SECRET=<YOUR_NEW_TESTNET_SECRET>
BINANCE_TESTNET=true

# Save (Ctrl+X, Y, Enter)
```

**Method 2: Via SCP (from Windows)**
```powershell
# Edit local copy
notepad C:\quantum_trader\testnet.env

# Copy to VPS
wsl scp -i ~/.ssh/hetzner_fresh C:\quantum_trader\testnet.env root@46.224.116.254:/home/qt/.env
```

**CRITICAL**: Ensure `.env` file format is EXACTLY:
```bash
BINANCE_API_KEY=your_key_here_no_quotes
BINANCE_API_SECRET=your_secret_here_no_quotes
BINANCE_TESTNET=true
```

**DO NOT**:
- ❌ Add quotes: `BINANCE_API_KEY="key"` (WRONG)
- ❌ Add spaces: `BINANCE_API_KEY = key` (WRONG)
- ❌ Add comments on same line: `BINANCE_API_KEY=key # my key` (WRONG)

### Step 4: Restart BSC Service

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl restart quantum-bsc.service'
```

### Step 5: Verify API Connection (within 60 seconds)

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'tail -20 /var/log/quantum/bsc.log'
```

**Expected Output (SUCCESS)**:
```
2026-02-10 XX:XX:XX [INFO] 🔍 BSC Check Cycle #1
2026-02-10 XX:XX:XX [INFO] 📊 Found 0 open position(s)
2026-02-10 XX:XX:XX [INFO] ✅ No open positions
```

**NO MORE `-2015` ERRORS**

---

## 🧪 Quick API Test (Before Restarting BSC)

Test new keys directly:

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'python3 -c "
from binance.client import Client
import os

# Load from .env (or paste directly for testing)
api_key = os.getenv(\"BINANCE_API_KEY\")
api_secret = os.getenv(\"BINANCE_API_SECRET\")

client = Client(api_key, api_secret, testnet=True)

try:
    positions = client.futures_position_information()
    print(f\"✅ SUCCESS: API works! Found {len(positions)} positions\")
except Exception as e:
    print(f\"❌ ERROR: {e}\")
"'
```

If this returns `✅ SUCCESS`, then restart BSC.

---

## 📊 Testnet vs Production Differences

| Feature | Testnet | Production |
|---------|---------|------------|
| URL | testnet.binancefuture.com | www.binance.com/en/futures |
| Real Money | ❌ Fake USDT | ✅ Real USDT |
| API Keys | Separate generation | Separate generation |
| Free Funds | ✅ Unlimited testnet USDT | ❌ Must deposit real funds |
| Order Execution | ✅ Simulated | ✅ Real exchange |
| IP Whitelist | Optional | **Strongly recommended** |

**CRITICAL**: Testnet API keys are **COMPLETELY SEPARATE** from production keys. You cannot use production keys on testnet or vice versa.

---

## 🚨 Common Errors

### Error: `-2015` (current issue)
**Means**: Invalid keys, wrong IP, or missing permissions  
**Fix**: Regenerate keys from **testnet.binancefuture.com** (NOT production Binance)

### Error: `-1022` (Signature invalid)
**Means**: API secret wrong or time sync issue  
**Fix**: Check API secret, ensure VPS time is synced (`timedatectl`)

### Error: `-2014` (API key format invalid)
**Means**: Key contains whitespace or special chars  
**Fix**: Re-copy keys, ensure no spaces/newlines in .env

### Error: `-1021` (Timestamp)
**Means**: VPS clock is wrong  
**Fix**: `timedatectl set-ntp true` on VPS

---

## ✅ Success Checklist

After updating keys:

- [ ] New keys generated from **testnet.binancefuture.com** (not production)
- [ ] Futures permission enabled in testnet API settings
- [ ] IP `46.224.116.254` whitelisted (or unrestricted for testing)
- [ ] Keys copied to `/home/qt/.env` with correct format (no quotes)
- [ ] `BINANCE_TESTNET=true` present in .env
- [ ] BSC service restarted
- [ ] Logs show NO `-2015` errors within 60s
- [ ] First poll cycle completes successfully

---

## 🔄 After Testnet Success (Future Migration to Production)

**When ready for production**:

1. Generate **production** API keys from www.binance.com
2. Change `BINANCE_TESTNET=false` in .env
3. Restart BSC
4. ⚠️ **CRITICAL**: Test with SMALL position first (BSC will close REAL positions)

**Recommended**: Keep BSC on testnet for 7-14 days to verify behavior before production.

---

**Last Updated**: 2026-02-10 22:10 UTC  
**Next Step**: Generate new testnet keys from testnet.binancefuture.com
