# Crypto VPS kravliste + akseptansetest (Quantum Trader / Podman-Compose)

Dette dokumentet er laget for Quantum Trader som kjører på Linux VPS med Podman + podman-compose.

## 1) Minimumskrav (anbefalt baseline)

**Region**
- Primær: Singapore (ofte best “all-round” for Binance/crypto)
- Alternativ: Tokyo. Hongkong kun hvis målinger gir lavere jitter/p95.

**Maskinvare (CPU-only)**
- For opptil ~100 symbols/strategier: **8 vCPU / 16 GB RAM / NVMe 160 GB**
- Hvis AI/ML-prosesser er tunge eller du ser høy RAM-bruk: **8 vCPU / 32 GB RAM**
- Minimum (billigst, men mer risiko for spikes): **4 vCPU / 8 GB RAM**

**Nettverk**
- 1 Gbps er fint; viktigere er stabil routing (lav jitter) og god peering.
- Unngå “budget/shared” dersom leverandør tilbyr “compute/dedicated”.

**OS**
- Ubuntu 22.04/24.04 LTS.

## 2) Sikkerhetskrav (må være på plass)

**Tilgang**
- Kun SSH med nøkler. Slå av passord-login (`PasswordAuthentication no`).
- 2FA på leverandørkonto.

**Brannmur**
- Eksponer kun det du trenger offentlig.
- Anbefalt: `80/443` (frontend/reverse proxy) + `22` (SSH, gjerne låst til din IP).

**Ikke eksponer internt**
- Ikke eksponer `redis` (`6379`) offentlig.
- Ikke eksponer backend (`8000`) og ai-engine (`8001`) offentlig på VPS.
  - Hold dem på internt docker/podman-nett, eller kun tilgjengelig via `localhost`/VPN.

**Secrets**
- API-nøkler skal ligge i `.env`/secret store, ikke hardkodet i compose.
- Rotér nøkler før “live” hvis de noen gang har vært sjekket inn i repo.

## 3) Drift (24/7 stabilitet)

**Prosess-oppstart**
- Containerne må starte automatisk etter reboot.
- Kjør rootless Podman (egen bruker), og bruk systemd user services.

**Overvåkning (minimum)**
- CPU/RAM/disk, container health, latency mot exchange, feilrate på ordre.
- Alarmer (Telegram/Discord/e-post) ved:
  - Backend/AI health fail
  - gjentatte ordre-feil
  - høy p95/p99 latency

**Backups**
- Daglige snapshots hos leverandør.
- Backup av:
  - `.env` (kryptert)
  - database/state
  - modeller/checkpoints
  - `logs` ved behov

## 4) Podman + podman-compose (VPS praktiske krav)

**Pakker**
- `podman`, `python3-pip`
- `podman-compose` via `pip3 install --user podman-compose`

**Rootless**
- Kjør som vanlig bruker (ikke root) for bedre sikkerhet.

**Compose-filer**
- Bruk egne overrides for VPS slik at porter ikke eksponeres unødvendig.
  - Typisk: frontend på `80/443`, alt annet internt.

## 5) Akseptansetest (kjør før du går live)

### A. Baseline system
1) Sjekk tidssynk:
- `timedatectl status`

2) Sjekk ressurser:
- `nproc; free -h; df -h`

3) Sjekk nettverksstabilitet (jitter/packet loss):
- `ping -c 50 1.1.1.1`

### B. Start stacken
1) Start:
- `podman-compose -f docker-compose.yml -f docker-compose.vps.yml up -d`

2) Status:
- `podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"`
- `podman-compose ps`

3) Health lokalt:
- `curl -sS http://localhost:8000/health | head`
- `curl -sS http://localhost:8001/health | head`

> Merk: På VPS bør `8000/8001` typisk ikke være eksponert offentlig. Test derfor via `localhost`.

### C. Verifiser at ingenting farlig er åpent
1) Lyttende porter:
- `ss -lntp`

Forventet:
- `80` (og evt. `443`) er åpne.
- `6379`, `8000`, `8001` skal IKKE være åpne mot verden.

### D. Binance latency-test (mål p95/p99)

**Spot**
- `for i in {1..50}; do curl -s -o /dev/null -w "%{time_total}\n" https://api.binance.com/api/v3/time; done | sort -n | tail -n 5`

**Futures**
- `for i in {1..50}; do curl -s -o /dev/null -w "%{time_total}\n" https://fapi.binance.com/fapi/v1/time; done | sort -n | tail -n 5`

Tolkning:
- Se på de høyeste verdiene (tail) for en enkel “worst-ish” indikasjon.
- For bedre p95/p99: kjør 200–1000 samples og beregn percentiler.

### E. Reboot-test
1) Reboot:
- `sudo reboot`

2) Etter reboot:
- `podman ps`
- `curl -sS http://localhost:8000/health`
- `curl -sS http://localhost:8001/health`

Krav:
- Tjenestene kommer opp uten manuell inngripen.

---

## Anbefalt startkonfig (for dine mål)
- Region: Singapore
- 8 vCPU / 16 GB RAM / NVMe 160 GB
- Eksponer kun 80/443 offentlig, hold 8000/8001/6379 interne
