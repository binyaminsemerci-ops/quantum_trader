# 🚀 QUANTUM TRADER - VPS DEPLOYMENT (AUTOMATISK)

## ✅ ALT ER KLART!

Jeg har gjort alt klart for deg. Følg disse 3 enkle stegene:

---

## 📋 STEG 1: ÅPNE DEPLOYMENT-SKRIPTET

Åpne filen `deploy-to-vps.sh` og sett VPS IP-adressen din på linje 19:

```bash
VPS_IP="din.vps.ip.adresse"  # <-- ENDRE DENNE!
```

For eksempel:
```bash
VPS_IP="185.123.45.67"
```

---

## 🚀 STEG 2: KJØR DEPLOYMENT

Kjør dette i WSL:

```bash
cd ~/quantum_trader
chmod +x deploy-to-vps.sh
./deploy-to-vps.sh
```

**Det er alt! Skriptet gjør automatisk:**
1. ✅ Tester SSH-tilkobling
2. ✅ Installerer Podman + Python + Git
3. ✅ Cloner repository fra GitHub
4. ✅ Kopierer .env fil
5. ✅ Kopierer AI-modeller (110MB)
6. ✅ Starter Redis + AI Engine
7. ✅ Verifiserer at alt fungerer

---

## 🔍 STEG 3: SJEKK AT DET FUNGERER

Når skriptet er ferdig, SSH til VPS:

```bash
ssh root@YOUR_VPS_IP
```

Sjekk at containere kjører:

```bash
podman ps
```

Test health endpoint:

```bash
curl http://localhost:8001/health
```

Se logs:

```bash
podman logs -f quantum_ai_engine
```

---

## 📊 MANUAL COMMANDS (HVIS NØDVENDIG)

Hvis du vil kjøre steg-for-steg manuelt:

### På VPS (etter SSH):

```bash
# Se status
podman ps

# Restart en service
podman-compose -f docker-compose.wsl.yml restart ai-engine

# Se logs
podman logs quantum_ai_engine

# Stopp alt
podman-compose -f docker-compose.wsl.yml down

# Start på nytt
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine
```

---

## 🛠️ TROUBLESHOOTING

### Problem: SSH fungerer ikke
```bash
# Test SSH først
ssh root@YOUR_VPS_IP

# Hvis det ikke fungerer, sett opp SSH-nøkkel:
ssh-keygen -t ed25519 -f ~/.ssh/vps_key
ssh-copy-id -i ~/.ssh/vps_key.pub root@YOUR_VPS_IP
```

### Problem: Container starter ikke
```bash
# SSH til VPS
ssh root@YOUR_VPS_IP

# Se logs
podman logs quantum_ai_engine

# Rebuild og start på nytt
cd ~/quantum_trader
podman-compose -f docker-compose.wsl.yml build ai-engine
podman-compose -f docker-compose.wsl.yml up -d ai-engine
```

### Problem: Health check feiler
```bash
# SSH til VPS
ssh root@YOUR_VPS_IP

# Sjekk at Redis kjører
podman exec quantum_redis redis-cli ping

# Test health endpoint
curl http://localhost:8001/health

# Se detaljerte logs
podman logs --tail 100 quantum_ai_engine
```

---

## 📁 FILER SOM ER LAGET FOR DEG

1. **`deploy-to-vps.sh`** - Komplett automatisk deployment script
2. **`docker-compose.wsl.yml`** - Podman-compose konfigurasjon (fungerer på både WSL og VPS)
3. **`scripts/start-wsl-podman.sh`** - Start services
4. **`scripts/verify-wsl-podman.sh`** - Verifiser at alt fungerer
5. **`scripts/setup-vps.sh`** - VPS initial setup (kjøres automatisk av deploy-script)

---

## ✅ SUCCESS CRITERIA

Du vet at deployment er vellykket når:

1. ✅ `podman ps` viser `quantum_redis` og `quantum_ai_engine` som "Up"
2. ✅ `curl http://localhost:8001/health` returnerer HTTP 200 med JSON
3. ✅ Logs viser "Application startup complete"
4. ✅ Ingen ImportError i logs

---

## 🎯 NESTE STEG ETTER DEPLOYMENT

1. **Test Backend API:**
   ```bash
   curl http://YOUR_VPS_IP:8001/health
   ```

2. **Overvåk systemet:**
   ```bash
   ssh root@YOUR_VPS_IP
   podman logs -f quantum_ai_engine
   ```

3. **Sett opp automatisk restart ved reboot:**
   ```bash
   ssh root@YOUR_VPS_IP
   crontab -e
   # Legg til:
   @reboot cd /root/quantum_trader && podman-compose -f docker-compose.wsl.yml up -d
   ```

---

## 💬 SPØRSMÅL?

Hvis noe går galt:
1. Se logs: `podman logs quantum_ai_engine`
2. Sjekk at .env er kopiert riktig
3. Verifiser at VPS har nok diskplass: `df -h`
4. Sjekk at portene er åpne: `ufw status`

---

**Lykke til! 🚀**
