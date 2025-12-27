# Docker Disk Cleanup Guide - Frigjør 88 GB!

## Status
- **Før cleanup**: 170.53 GB Docker disk
- **Etter cleanup**: Skal bli ~82 GB
- **Data slettet fra Docker**: ✅ 88.5 GB frigjort
- **VHDX disk krympet**: ⏳ Venter

## Problem
Docker's virtuelle disk (VHDX) krymper ikke automatisk selv om vi sletter data.

---

## ✅ ANBEFALT LØSNING: Docker Desktop GUI

### Steg 1: Åpne Docker Desktop
1. Start Docker Desktop
2. Vent til den er helt klar (grønn ikon i system tray)

### Steg 2: Gå til Innstillinger
1. Klikk på **⚙️ Settings** (tannhjul øverst til høyre)
2. Velg **Resources** → **Advanced**

### Steg 3: Reduser Disk Image
1. Finn **"Disk image size"** eller **"Virtual disk limit"**
2. Klikk på **"Compact"** knappen (hvis tilgjengelig)
3. ELLER: Reduser tallet fra 170 GB til f.eks. 100 GB
4. Klikk **"Apply & Restart"**

Docker vil nå komprimere VHDX-filen og frigjøre ~88 GB!

---

## 🔧 ALTERNATIV 1: PowerShell (Krever Admin)

```powershell
# 1. Stopp Docker helt
Stop-Service docker
wsl --shutdown

# 2. Åpne PowerShell som Administrator og kjør:
wsl --manage docker-desktop-data --set-sparse true

# 3. Start Docker igjen
Start-Service docker
```

---

## 🔧 ALTERNATIV 2: Manuell Diskpart (Krever Admin)

```powershell
# 1. Stopp Docker og WSL helt
Stop-Process -Name "Docker Desktop" -Force
wsl --shutdown

# 2. Åpne PowerShell som Administrator og kjør:
$vhdxPath = "C:\Users\belen\AppData\Local\Docker\wsl\disk\docker_data.vhdx"

# Lage diskpart script
@"
select vdisk file=$vhdxPath
attach vdisk readonly
compact vdisk
detach vdisk
"@ | Out-File "$env:TEMP\compact.txt" -Encoding ASCII

# Kjør komprimering (tar 10-20 minutter!)
diskpart /s "$env:TEMP\compact.txt"

# 3. Start Docker igjen
```

---

## 🔧 ALTERNATIV 3: Eksport/Import (Mest pålitelig)

Denne metoden er sikker men tar lengst tid:

```powershell
# 1. Stopp alt
wsl --shutdown

# 2. Eksporter docker-desktop-data (tar 10-30 min)
wsl --export docker-desktop-data C:\temp\docker-data-backup.tar

# 3. Avregistrer den gamle
wsl --unregister docker-desktop-data

# 4. Importer tilbake (dette komprimerer automatisk)
wsl --import docker-desktop-data C:\Users\belen\AppData\Local\Docker\wsl\data C:\temp\docker-data-backup.tar

# 5. Slett backup
Remove-Item C:\temp\docker-data-backup.tar

# 6. Start Docker Desktop
```

---

## 📊 Verifiser Resultat

```powershell
$vhdx = Get-Item "C:\Users\belen\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
$sizeGB = [math]::Round($vhdx.Length / 1GB, 2)
Write-Host "Docker disk størrelse: $sizeGB GB"
```

**Forventet resultat**: ~82 GB (ned fra 170.53 GB)

---

## 💡 Tips for Fremtiden

### Automatisk cleanup hver uke:
```powershell
# Legg til i Task Scheduler
docker system prune -af --volumes
```

### Sett disk limit i Docker Desktop:
- Settings → Resources → Advanced
- Sett "Virtual disk limit" til 100 GB
- Dette forhindrer at disken vokser ukontrollert

---

## 🆘 Hvis Ingenting Fungerer

**Siste utvei: Reset Docker Desktop**
1. Åpne Docker Desktop
2. Settings → Troubleshoot → **Reset to factory defaults**
3. Klikk **"Reset"**
4. Dette sletter ALT og gir deg en fresh start på ~2-5 GB

⚠️ **ADVARSEL**: Du må rebuilde alle images etterpå!

---

## Status Fil

Denne filen beskriver situasjonen pr. 21. desember 2025.
Vi har slettet 88.5 GB data fra Docker, men VHDX-filen må fortsatt krympes.
