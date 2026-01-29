# 🔒 Sikker Testnet Credentials - Guide

## Problem
Testnet API keys ble eksponert i chat history. Selv om det er testnet, bør credentials aldri være synlige i plaintext.

## Løsning
Implementert systemd credentials encryption for alle testnet credentials.

## Sikkerhetslag

### 1. Encrypted at Rest
```bash
/etc/quantum/creds/TESTNET_API_KEY.cred      (systemd-creds encrypted)
/etc/quantum/creds/TESTNET_API_SECRET.cred   (systemd-creds encrypted)
```
- Permissions: 600, root:root
- Kan kun dekrypteres av systemd på denne maskinen
- Sikker mot disk-avlesing

### 2. Decrypted Runtime (Root-Only)
```bash
/etc/quantum/testnet-secrets/credentials.env  (600, root:root)
/etc/quantum/position-monitor.env            (600, root:root)
```
- Dekrypterte values, men kun root kan lese
- Services laster via EnvironmentFile
- Aldri eksponert i logs eller stdout

### 3. Service Isolation
```ini
# /etc/systemd/system/quantum-exit-monitor.service.d/credentials.conf
[Service]
EnvironmentFile=-/etc/quantum/testnet-secrets/credentials.env
```
- Services får credentials via environment
- Ikke synlige i `ps` output
- Isolert per-service

## Rotation Prosedyre

### Steg 1: Generer Nye Keys (Utenfor Chat)
1. Åpne: https://testnet.binancefuture.com/
2. API Management → Create New Key
3. Permissions: Enable "Futures Trading"
4. IP Whitelist: `46.224.116.254`
5. Kopier keys til sikker lokasjon (IKKE chat)

### Steg 2: Upload Script til VPS
```powershell
# Fra Windows
wsl scp -i ~/.ssh/hetzner_fresh C:\quantum_trader\secure_testnet_credentials.sh root@46.224.116.254:/root/
```

### Steg 3: Kjør Setup (SSH Direkte)
```bash
# SSH til VPS (IKKE via chat)
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Kjør setup
chmod +x /root/secure_testnet_credentials.sh
bash /root/secure_testnet_credentials.sh

# Script vil:
# 1. Åpne nano for API_KEY (paste, Ctrl+X, Y, Enter)
# 2. Åpne nano for API_SECRET (paste, Ctrl+X, Y, Enter)
# 3. Kryptere med systemd-creds
# 4. Oppdatere alle env-filer
# 5. Restarte services
# 6. Teste API connectivity
# 7. Vise VERDICT (uten å eksponere keys)
```

### Steg 4: Verifiser (Kan Deles i Chat)
```bash
# Kun summary info, ingen secrets
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 << 'SSHEOF'
echo "=== VERIFICATION (No Secrets) ==="
echo ""
echo "A) Encrypted files:"
ls -lh /etc/quantum/creds/TESTNET_*.cred | awk '{print $9, $5}'
echo ""
echo "B) Services:"
systemctl is-active quantum-exit-monitor.service quantum-position-monitor.service
echo ""
echo "C) API Key (first 10 chars):"
grep '^BINANCE_API_KEY=' /etc/quantum/testnet-secrets/credentials.env | cut -d= -f2 | cut -c1-10
echo "... (rest redacted)"
SSHEOF
```

## Hva Skjedde med Gamle Keys?

### Gamle Keys (Eksponert i Chat)
```
API Key: w2W60kzuCf... (KOMPROMITTERT)
Secret:  QI18cg4zc...  (KOMPROMITTERT)
```
**Status**: Må revokes på testnet.binancefuture.com

### Nye Keys (Kryptert)
```
Location: /etc/quantum/creds/TESTNET_*.cred
Format:   systemd-creds encrypted
Access:   Kun systemd på denne maskinen
Lesbar:   NEI (hverken i chat, logs, eller disk-avlesning)
```

## Testing (Sikker Metode)

### ✅ Test API Uten å Vise Keys
```bash
python3 /tmp/test_new_creds.py
```
Output:
```
Testing with API Key: ********** (redacted)

✅ TESTNET API WORKING
   Balance: 14194.13 USDT
   Open positions: 2

✅ VERDICT: New credentials working
```

### ❌ Aldri Gjør Dette
```bash
# IKKE gjør dette (eksponerer keys):
cat /etc/quantum/testnet-secrets/credentials.env

# IKKE gjør dette (keys i command):
export BINANCE_API_KEY="..."

# IKKE gjør dette (keys i logs):
echo "API Key: $BINANCE_API_KEY"
```

## Fordeler

### Før (Usikkert)
- ❌ Keys i plaintext i /etc/quantum/testnet.env
- ❌ Keys eksponert i chat via `cat` commands
- ❌ Keys synlige i process listings
- ❌ Keys i logs og stdout

### Etter (Sikkert)
- ✅ Keys kryptert på disk (systemd-creds)
- ✅ Dekrypterte keys kun root-readable
- ✅ Services får keys via EnvironmentFile (ikke CLI)
- ✅ Aldri eksponert i logs eller chat
- ✅ Kan verifisere uten å vise keys
- ✅ Rotation uten downtime

## Maintenance

### Sjekk Status
```bash
# Service health
systemctl status quantum-exit-monitor.service

# Logs (ingen keys her)
tail -50 /var/log/quantum/exit-monitor.log

# API connectivity (redacted)
python3 /tmp/test_new_creds.py | grep VERDICT
```

### Rotate Keys
```bash
# Kjør setup på nytt
bash /root/secure_testnet_credentials.sh

# Revoke gamle keys på testnet.binancefuture.com
```

### Backup Encrypted Credentials
```bash
# Backup encrypted files (safe to store)
tar czf /root/testnet-creds-backup-$(date +%Y%m%d).tar.gz \
    /etc/quantum/creds/TESTNET_*.cred

# These can ONLY be decrypted on THIS machine
```

## Security Properties

### Encryption
- Algorithm: AES-256-GCM
- Key derivation: TPM2 (hvis tilgjengelig) eller machine-id
- Decryption: Kun systemd på denne maskinen

### Access Control
- Encrypted files: 600, root:root
- Decrypted files: 600, root:root (ikke qt)
- Services: Laster via EnvironmentFile (systemd manages)

### Audit Trail
```bash
# Hvem har lest decrypted files?
ausearch -f /etc/quantum/testnet-secrets/credentials.env

# Service restarts med credential changes
journalctl -u quantum-exit-monitor.service --since "1 hour ago"
```

## Comparison: Mainnet vs Testnet

### Mainnet Credentials
- Location: `/etc/quantum/binance-pnl-tracker.env`
- Status: Needs same treatment
- TODO: Implement secure_mainnet_credentials.sh

### Testnet Credentials
- Location: `/etc/quantum/creds/TESTNET_*.cred` (encrypted)
- Status: ✅ Secure (after rotation)
- Method: systemd-creds + root-only decrypted files

## Next Steps

1. **Immediate**: Rotate testnet keys (revoke old, apply new via script)
2. **Soon**: Apply same method to mainnet credentials
3. **Future**: Consider HashiCorp Vault for centralized secret management

## Verification Checklist

After running secure_testnet_credentials.sh:

- [ ] Encrypted files exist: `ls /etc/quantum/creds/TESTNET_*.cred`
- [ ] Permissions correct: `ls -l` shows 600 root:root
- [ ] Services active: `systemctl is-active quantum-{exit-monitor,position-monitor}.service`
- [ ] API working: `python3 /tmp/test_new_creds.py` shows ✅
- [ ] No keys in logs: `grep -r "w2W60kzuCf" /var/log/quantum/` returns nothing
- [ ] Old keys revoked on testnet.binancefuture.com

## Contact / Issues

If API test fails after rotation:
1. Check IP whitelist includes: `46.224.116.254`
2. Verify permissions: "Futures Trading" enabled
3. Confirm testnet endpoint: `https://testnet.binancefuture.com`
4. Check service logs: `/var/log/quantum/position-monitor.log`

---

**Last Updated**: 2026-01-29  
**Security Level**: ✅ Encrypted at rest, root-only runtime access  
**Status**: Ready for production use
