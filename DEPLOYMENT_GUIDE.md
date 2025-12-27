# 🚀 QUANTUM TRADER - VPS DEPLOYMENT GUIDE

**Sist oppdatert:** 11. November 2025  
**Status:** Klar for produksjon ✅

---

## ✅ PRE-DEPLOYMENT SJEKKLISTE

### 1. Sikkerhet
- [ ] **VIKTIG:** Fjern alle API keys fra kode
- [ ] Bruk environment variables (.env fil)
- [ ] Ikke commit .env til Git
- [ ] Endre database passord
- [ ] Sett opp firewall på VPS
- [ ] Bruk SSL/HTTPS for frontend

### 2. Konfigurasjon
- [ ] Sett riktig `max_notional` og `max_daily_loss`
- [ ] Konfigurer `QT_ALLOWED_SYMBOLS` for 100 coins
- [ ] Juster position sizes etter kapital
- [ ] Test i paper trading mode først
- [ ] Verifiser alle 6 optimization komponenter

### 3. Dependencies
- [ ] Python 3.11+ installert
- [ ] Node.js 18+ for frontend
- [ ] Docker (valgfritt, men anbefalt)
- [ ] Nok diskplass (minimum 5GB)
- [ ] Minne: 4GB+ RAM anbefalt

---

## 🔐 SIKKERHET - KRITISK!

### API Key Configuration - Two Methods

**Method 1: Dashboard Settings (RECOMMENDED for Production)**

You can configure API keys dynamically via the dashboard settings page:

1. Navigate to the Settings page in the web dashboard
2. Enter your API key and secret
3. Save the settings
4. Restart the backend (or wait for next execution cycle)

**Advantages:**
- ✅ No need to edit files or restart services
- ✅ Keys stored securely in application memory
- ✅ Can be changed without redeployment
- ✅ Works with hot-reload

**Method 2: Environment Variables (Fallback)**

If dashboard settings are not configured, the system falls back to environment variables from `.env` file.

**Priority Order (First Match Wins):**
1. 🥇 Dashboard settings (set via `/settings` API endpoint)
2. 🥈 Environment variables (`.env` file or system env)

This dual-method approach ensures:
- Dashboard settings take priority when available
- Environment variables provide a reliable fallback
- No service restarts needed for dashboard changes

### Environment Variables (.env fil)

Opprett `.env` fil i root directory:

```bash
# Trading Config
QT_ALLOWED_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT,...
QT_MAX_NOTIONAL_PER_TRADE=1000.0
QT_MAX_DAILY_LOSS=500.0
QT_KILL_SWITCH=false
STAGING_MODE=false
QT_MARKET_TYPE=spot            # spot | usdm_perp | coinm_perp
QT_MARGIN_MODE=cross           # cross | isolated (kun futures)
QT_DEFAULT_LEVERAGE=5          # 1-125 (bruk konservativ først)
QT_LIQUIDITY_STABLE_QUOTES=USDT,USDC  # Kun USDT/USDC pairs

# Exchange API (hvis du bruker live trading OG ikke bruker dashboard)
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here

# Database
DATABASE_URL=sqlite:///./backend/data/trades.db

# Security
QT_ADMIN_TOKEN=generate_strong_random_token_here
SECRET_KEY=generate_another_strong_token_here
```

### Generer Sterke Tokens:
```bash
# På Linux/Mac:
openssl rand -hex 32

# På Windows (PowerShell):
-join ((48..57) + (65..90) + (97..122) | Get-Random -Count 32 | % {[char]$_})
```

---

## 📦 DEPLOYMENT METODER

### Metode 1: Docker (ANBEFALT)

**Fordeler:**
- ✅ Isolert miljø
- ✅ Enkel deployment
- ✅ Samme oppsett på alle servere
- ✅ Automatisk restart ved crash
 - ✅ Enklere å bytte mellom spot og futures

### Futures / Margin Modus

For å aktivere Binance USDT perpetual futures (paper / begrenset test):

1. Sett miljøvariabler:
```bash
QT_MARKET_TYPE=usdm_perp
QT_MARGIN_MODE=cross
QT_DEFAULT_LEVERAGE=5
STAGING_MODE=true  # ANBEFALT VED TEST
```
2. Restart backend.
3. Verifiser i loggene at endpoint bruker `fapi.binance.com` og kun PERPETUAL kontrakter hentes.
4. Sjekk at symbols nå inkluderer futures-par (f.eks. BTCUSDT, ETHUSDT) – samme tickere men derivatdata.

For coin-margined futures:
```bash
QT_MARKET_TYPE=coinm_perp
```

### Aktivere Futures Ordreutførelse

Standard kjører systemet i spot/paper modus. For å aktivere futures execution adapter (tørre ordre i staging):

```bash
QT_EXECUTION_EXCHANGE=binance-futures   # eller 'paper' for simulering
STAGING_MODE=true                       # beholder ordre som DRY-RUN
```

Når du er klar for LIVE futures (høy risiko):

```bash
STAGING_MODE=false                      # fjerner dry-run beskyttelse
QT_EXECUTION_EXCHANGE=binance-futures
QT_MARKET_TYPE=usdm_perp                # eller coinm_perp
QT_MARGIN_MODE=cross                    # cross eller isolated
QT_DEFAULT_LEVERAGE=5                   # anbefalt lavt å starte
```

### Live Sikkerhetsjekk Før Du Slår Av Staging
1. Verifiser at API keys har riktige permissions (trade futures)
2. Sjekk at `QT_MAX_NOTIONAL_PER_TRADE` er konservativ
3. Sett `QT_KILL_SWITCH=true` eller implementer PnL guard
4. Logg overvåkes (fyll vs dry-run linjer)
5. Test én liten ordre manuelt før full rebalance
6. Ikke øk leverage før stabil drift >= 48 timer

### Dry-Run Logging
I staging vil alle futures ordre logges som:

```
[DRY-RUN] Futures order BUY BTCUSDT qty=0.001 price=... 
```

Når staging er av, vil du se faktiske orderId i loggene.

### Begrensninger (Foreløpig)
* Funding rate håndtering ikke implementert
* Unrealized PnL / liquidation margin ikke i risk_guard ennå
* Leverage/marginType settes lazily ved første ordre per symbol
* Ingen automatisk reduksjon av posisjoner ved økt risiko

Planlagt forbedring: egen `FuturesRiskGuard` med funding accrual, maintenance margin buffer og intraday liquidation stress test.

### Aktivere Futures Ordreutførelse

Standard kjører systemet i spot/paper modus. For å aktivere futures execution adapter (tørre ordre i staging):

```bash
QT_EXECUTION_EXCHANGE=binance-futures   # eller 'paper' for simulering
STAGING_MODE=true                       # beholder ordre som DRY-RUN
```

Når du er klar for LIVE futures (høy risiko):

```bash
STAGING_MODE=false                      # fjerner dry-run beskyttelse
QT_EXECUTION_EXCHANGE=binance-futures
QT_MARKET_TYPE=usdm_perp                # eller coinm_perp
QT_MARGIN_MODE=cross                    # cross eller isolated
QT_DEFAULT_LEVERAGE=5                   # anbefalt lavt å starte
```

### Live Sikkerhetsjekk Før Du Slår Av Staging
1. Verifiser at API keys har riktige permissions (trade futures)
2. Sjekk at `QT_MAX_NOTIONAL_PER_TRADE` er konservativ
3. Sett `QT_KILL_SWITCH=true` eller implementer PnL guard
4. Logg overvåkes (fyll vs dry-run linjer)
5. Test én liten ordre manuelt før full rebalance
6. Ikke øk leverage før stabil drift >= 48 timer

### Dry-Run Logging
I staging vil alle futures ordre logges som:

```
[DRY-RUN] Futures order BUY BTCUSDT qty=0.001 price=... 
```

Når staging er av, vil du se faktiske orderId i loggene.

### Begrensninger (Foreløpig)
* Funding rate håndtering ikke implementert
* Unrealized PnL / liquidation margin ikke i risk_guard ennå
* Leverage/marginType settes lazily ved første ordre per symbol
* Ingen automatisk reduksjon av posisjoner ved økt risiko

Planlagt forbedring: egen `FuturesRiskGuard` med funding accrual, maintenance margin buffer og intraday liquidation stress test.

### Viktig For Futures:
- Ikke gå live før du har verifisert margin / likviditet i staging.
- Bruk lav leverage (≤ 5) til å begynne med.
- Oppdater risk-regler: `QT_MAX_NOTIONAL_PER_TRADE` bør nedjusteres pga. høyere implisitt risiko.
- Vurder å legge til ekstra kill switch basert på unrealized PnL.
- Utvid risk manager senere for maintenance margin / liquidation buffer.

**Sjekk at `docker-compose.yml` eksisterer:**
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - QT_MAX_NOTIONAL_PER_TRADE=1000
      - QT_MAX_DAILY_LOSS=500
    volumes:
      - ./backend/data:/app/data
    restart: unless-stopped
  
  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    restart: unless-stopped
```

**Deploy med Docker:**
```bash
# 1. Last opp til VPS
scp -r quantum_trader user@your-vps-ip:/home/user/

# 2. SSH til VPS
ssh user@your-vps-ip

# 3. Gå til directory
cd /home/user/quantum_trader

# 4. Opprett .env fil med dine settings

# 5. Bygg og start
docker-compose up -d

# 6. Sjekk logs
docker-compose logs -f backend
```

---

### Metode 2: Direkte Installasjon

**1. Forbered lokalt:**
```bash
# Fjern unødvendige filer
cd quantum_trader
Remove-Item -Recurse -Force __pycache__, .pytest_cache, node_modules, .venv

# Test at alt fungerer
python backend/main.py
```

**2. Last opp til VPS:**
```bash
# Via SCP
scp -r quantum_trader user@your-vps-ip:/home/user/

# Eller via Git (anbefalt)
git add .
git commit -m "Ready for deployment"
git push origin main

# På VPS:
ssh user@your-vps-ip
cd /home/user
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git
```

**3. Installer på VPS:**
```bash
cd quantum_trader

# Installer Python dependencies
python3 -m venv .venv
source .venv/bin/activate  # På Linux/Mac
pip install -r requirements.txt

# Installer Frontend dependencies (hvis du bruker frontend)
cd frontend
npm install
npm run build
cd ..

# Opprett .env fil
nano .env
# (Legg inn dine environment variables)

# Test backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

**4. Sett opp som systemd service (Linux):**

Opprett `/etc/systemd/system/quantum-trader.service`:
```ini
[Unit]
Description=Quantum Trader Backend
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/home/your-user/quantum_trader/backend
Environment="PATH=/home/your-user/quantum_trader/.venv/bin"
ExecStart=/home/your-user/quantum_trader/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Aktiver service:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable quantum-trader
sudo systemctl start quantum-trader
sudo systemctl status quantum-trader
```

---

## 🔥 FIREWALL KONFIGURASJON

```bash
# Ubuntu/Debian (ufw)
sudo ufw allow 22/tcp      # SSH
sudo ufw allow 8000/tcp    # Backend API
sudo ufw allow 5173/tcp    # Frontend (eller 80/443 for nginx)
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=5173/tcp
sudo firewall-cmd --reload
```

---

## 🌐 NGINX REVERSE PROXY (Anbefalt for produksjon)

**Installer nginx:**
```bash
sudo apt update
sudo apt install nginx
```

**Opprett `/etc/nginx/sites-available/quantum-trader`:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Aktiver:**
```bash
sudo ln -s /etc/nginx/sites-available/quantum-trader /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**SSL med Let's Encrypt:**
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 📊 MONITORING & LOGGING

### 1. System Logs
```bash
# Systemd service logs
sudo journalctl -u quantum-trader -f

# Docker logs
docker-compose logs -f backend

# Direkte backend logs
tail -f backend/logs/*.log
```

### 2. Performance Monitoring
```bash
# CPU/Memory
htop

# Disk space
df -h

# Network
netstat -tuln
```

### 3. Application Monitoring
```bash
# API health check
curl http://localhost:8000/health

# Trading stats
curl http://localhost:8000/ai/stats

# Recent trades
curl http://localhost:8000/ai/trades | jq
```

---

## 🛡️ SIKKERHETSHERDENING

### 1. SSH Sikkerhet
```bash
# Deaktiver root login
sudo nano /etc/ssh/sshd_config
# Sett: PermitRootLogin no
# Sett: PasswordAuthentication no (bruk keys)

sudo systemctl restart sshd
```

### 2. Automatiske Oppdateringer
```bash
# Ubuntu/Debian
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### 3. Fail2ban (mot brute-force)
```bash
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

---

## ⚠️ PAPER TRADING MODE (Anbefalt først!)

Før du går live, test i paper trading mode:

```bash
# I .env fil:
STAGING_MODE=true
QT_MAX_NOTIONAL_PER_TRADE=100  # Lav verdi for testing
```

**Kjør i paper mode i 1-2 uker:**
- Monitor win rate
- Sjekk at alle 100 coins evalueres
- Verifiser position sizing
- Observer drawdowns
- Test risk management

**Når du er klar for live:**
```bash
# I .env fil:
STAGING_MODE=false
QT_MAX_NOTIONAL_PER_TRADE=1000  # Din faktiske verdi
```

---

## 🚨 BACKUP STRATEGI

### Automatisk Database Backup
```bash
# Opprett backup script: /home/user/backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
cp /home/user/quantum_trader/backend/data/trades.db \
   /home/user/backups/trades_$DATE.db

# Hold kun siste 30 dager
find /home/user/backups -name "trades_*.db" -mtime +30 -delete
```

**Legg til i crontab:**
```bash
crontab -e
# Backup hver 6. time:
0 */6 * * * /home/user/backup.sh
```

---

## 📈 POST-DEPLOYMENT SJEKKLISTE

- [ ] Backend responderer på port 8000
- [ ] Frontend er tilgjengelig
- [ ] Scheduler kjører (sjekk logs)
- [ ] Database opprettes og fungerer
- [ ] AI modeller lastes korrekt
- [ ] Liquidity refresh skjer hver 15. min
- [ ] Trading cycle skjer hver 30. min
- [ ] Logs skrives uten feil
- [ ] Health check returnerer "healthy"
- [ ] API docs tilgjengelig på /docs

---

## 🆘 FEILSØKING

### Backend starter ikke:
```bash
# Sjekk port er ledig
sudo netstat -tuln | grep 8000

# Sjekk Python environment
source .venv/bin/activate
python --version

# Sjekk dependencies
pip install -r requirements.txt --upgrade

# Sjekk logs
tail -f backend/logs/*.log
```

### Database feil:
```bash
# Slett og gjenskape database
rm backend/data/trades.db
cd backend
python -c "from database import init_db; init_db()"
```

### Høy CPU/Memory:
```bash
# Reduser antall coins
# I .env: QT_ALLOWED_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT  # Kun 3 coins

# Øk scheduler intervaller
# I backend/utils/scheduler.py: interval_seconds=600  # 10 min i stedet for 3
```

---

## 🎯 PRODUCTION READY CHECKLIST

- [x] ✅ Alle 6 optimizations implementert
- [x] ✅ 100 coins konfigurert
- [x] ✅ Ensemble model trent
- [x] ✅ Risk management aktiv
- [x] ✅ Position sizing (Kelly Criterion)
- [x] ✅ Smart execution (TWAP/Iceberg)
- [x] ✅ Regime detection
- [x] ✅ Automated scheduler
- [ ] ⚠️ .env fil opprettet med dine settings
- [ ] ⚠️ API keys sikret
- [ ] ⚠️ Firewall konfigurert
- [ ] ⚠️ SSL aktivert (for produksjon)
- [ ] ⚠️ Monitoring satt opp
- [ ] ⚠️ Backups konfigurert
- [ ] ⚠️ Paper trading testet (1-2 uker)

---

## 📞 SUPPORT & DOKUMENTASJON

- **Fullstendig coin liste:** Se `COINS_CONFIGURATION.md`
- **Optimization guide:** Se `PROFIT_OPTIMIZATION_GUIDE.md`
- **API docs:** `http://your-vps:8000/docs`
- **Test results:** Se `END_TO_END_TEST_RESULTS.md`

---

## 🎉 DU ER KLAR!

Systemet er testet og klar for deployment. Følg stegene over, start i paper trading mode, og gå gradvis over til live trading når du er komfortabel.

**Lykke til!** 🚀📈💰
