# 🚀 QUANTUM TRADER - COPY-PASTE KOMMANDOER

## DEPLOYMENT TIL VPS (3 STEG)

### STEG 1: Åpne og endre VPS IP

```bash
cd ~/quantum_trader
nano deploy-to-vps.sh
```

Endre linje 19 til din VPS IP:
```bash
VPS_IP="185.123.45.67"  # <-- Din IP her!
```

Lagre med: `Ctrl+O`, `Enter`, `Ctrl+X`

---

### STEG 2: Gjør skriptet kjørbart

```bash
chmod +x deploy-to-vps.sh
```

---

### STEG 3: Kjør deployment

```bash
./deploy-to-vps.sh
```

**FERDIG!** Alt skjer automatisk nå! ☕

---

## VERIFISERING (ETTER DEPLOYMENT)

### SSH til VPS:
```bash
ssh root@YOUR_VPS_IP
```

### Sjekk containere:
```bash
podman ps
```

### Test health:
```bash
curl http://localhost:8001/health
```

### Se logs:
```bash
podman logs -f quantum_ai_engine
```

---

## NYTTIGE KOMMANDOER PÅ VPS

### Restart AI Engine:
```bash
cd ~/quantum_trader
podman-compose -f docker-compose.wsl.yml restart ai-engine
```

### Restart Redis:
```bash
podman-compose -f docker-compose.wsl.yml restart redis
```

### Stopp alt:
```bash
podman-compose -f docker-compose.wsl.yml down
```

### Start alt på nytt:
```bash
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine
```

### Rebuild og start:
```bash
podman-compose -f docker-compose.wsl.yml up -d --build ai-engine
```

---

## TROUBLESHOOTING

### Se detaljerte logs:
```bash
podman logs --tail 100 quantum_ai_engine
```

### Test Redis:
```bash
podman exec quantum_redis redis-cli ping
```

### Sjekk Python paths (skal ikke ha /mnt/c):
```bash
podman exec quantum_ai_engine python3 -c "import sys; print('\n'.join(sys.path))"
```

### Test import:
```bash
podman exec quantum_ai_engine python3 -c "from microservices.ai_engine.service_health import ServiceHealth; print('OK')"
```

---

## OVERVÅKNING

### Live logs (følg i sanntid):
```bash
podman logs -f quantum_ai_engine
```

### Se siste 50 linjer:
```bash
podman logs --tail 50 quantum_ai_engine
```

### Sjekk container status:
```bash
podman ps -a
```

### Sjekk ressursbruk:
```bash
podman stats
```

---

## AUTOMATISK RESTART VED REBOOT

```bash
crontab -e
```

Legg til:
```bash
@reboot cd /root/quantum_trader && podman-compose -f docker-compose.wsl.yml up -d
```

---

## BACKUP KOMMANDOER

### Backup Redis data:
```bash
podman exec quantum_redis redis-cli SAVE
cp ~/quantum_trader/redis_data/dump.rdb ~/backup/redis-$(date +%Y%m%d).rdb
```

### Backup logs:
```bash
tar -czf ~/backup/logs-$(date +%Y%m%d).tar.gz ~/quantum_trader/logs/
```

---

## NETWORKING

### Åpne ports i firewall:
```bash
ufw allow 8000/tcp  # Backend API
ufw allow 8001/tcp  # AI Engine
ufw status
```

### Test fra ekstern maskin:
```bash
curl http://YOUR_VPS_IP:8001/health
```

---

## HVIS DU MÅ STARTE FRA SCRATCH

### Full cleanup:
```bash
cd ~/quantum_trader
podman-compose -f docker-compose.wsl.yml down
podman system prune -a --volumes
rm -rf ~/quantum_trader
```

### Deretter kjør deployment-skriptet på nytt!

---

**Tips:** Kopier denne filen til VPS for rask referanse:

```bash
scp COPY_PASTE_COMMANDS.md root@YOUR_VPS_IP:~/
```

Så kan du se den på VPS med:
```bash
cat ~/COPY_PASTE_COMMANDS.md
```

---

**Lykke til! 🚀**
