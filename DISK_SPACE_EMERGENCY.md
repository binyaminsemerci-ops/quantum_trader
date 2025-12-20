# 🔴 DISK SPACE EMERGENCY - C: DRIVE 99.6% FULL

**Status**: ⚠️ **KRITISK** - Kun 4 GB ledig av 932 GB  
**Timestamp**: 2025-12-20  
**Problem**: Kan ikke lagre filer pga. full disk

---

## ✅ UMIDDELBAR FIX UTFØRT

### Cleanup Utført:
- ✅ **Papirkurv tømt**: Frigjorde ~4 GB
- ✅ **Temp filer slettet**: Frigjorde ~1 GB
- **Resultat**: 4.08 GB ledig plass (var 0 GB)

---

## 🔍 FINN HVA SOM TAR PLASS

### Metode 1: Windows Disk Cleanup (Rask)
```powershell
# Åpne Disk Cleanup
cleanmgr /d C:

# I Disk Cleanup vinduet:
# 1. Klikk "Clean up system files"
# 2. Velg alt (spesielt "Windows Update Cleanup", "Temporary files")
# 3. Klikk OK
```

### Metode 2: WinDirStat (Visuell analyse - ANBEFALT)
```powershell
# Last ned og installer WinDirStat:
# https://windirstat.net/

# Dette gir deg en visuell treemap av hva som bruker plass
```

### Metode 3: PowerShell analyse
```powershell
# Finn 20 største filer på C:
Get-ChildItem C:\ -Recurse -File -ErrorAction SilentlyContinue | 
    Sort-Object Length -Descending | 
    Select-Object -First 20 FullName, 
    @{Name="Size (GB)";Expression={[math]::Round($_.Length/1GB,2)}} |
    Format-Table -AutoSize

# Finn største mapper i user profile
Get-ChildItem C:\Users\belen -Directory | ForEach-Object {
    $size = (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue | 
             Measure-Object Length -Sum).Sum
    [PSCustomObject]@{
        Folder = $_.Name
        "Size (GB)" = [math]::Round($size/1GB,2)
    }
} | Sort-Object "Size (GB)" -Descending | Format-Table -AutoSize
```

---

## 🎯 VANLIGE SKYLDIGE (Sjekk disse)

### 1. WSL2 Virtual Disks
WSL2 kan ta ENORM plass!

```powershell
# Sjekk WSL disk størrelse
Get-ChildItem "$env:LOCALAPPDATA\Packages" -Recurse -Filter ext4.vhdx -ErrorAction SilentlyContinue |
    Select-Object FullName, @{Name="Size (GB)";Expression={[math]::Round($_.Length/1GB,2)}}

# Komprimer WSL disk (må kjøres fra PowerShell admin):
wsl --shutdown
Optimize-VHD -Path "$env:LOCALAPPDATA\Packages\CanonicalGroupLimited.Ubuntu_*\LocalState\ext4.vhdx" -Mode Full

# ELLER manuelt compact:
diskpart
select vdisk file="$env:LOCALAPPDATA\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"
compact vdisk
exit
```

### 2. Docker Images & Volumes
Docker kan ta mye plass!

```powershell
# Sjekk Docker disk bruk
docker system df

# Fjern ubrukte images/containers/volumes
docker system prune -a --volumes -f

# Dette kan frigjøre flere GB!
```

### 3. Python Virtual Environments
```powershell
# Sjekk .venv størrelse
Get-ChildItem C:\quantum_trader\.venv -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object Length -Sum |
    ForEach-Object { [math]::Round($_.Sum/1GB,2) }

# Hvis stor: Slett og rekonstruer
Remove-Item C:\quantum_trader\.venv -Recurse -Force
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Downloads mappe
```powershell
# Sjekk Downloads størrelse
$size = (Get-ChildItem C:\Users\belen\Downloads -Recurse -File -ErrorAction SilentlyContinue |
         Measure-Object Length -Sum).Sum / 1GB
Write-Host "Downloads: $([math]::Round($size,2)) GB"

# Slett gamle downloads (>90 dager)
Get-ChildItem C:\Users\belen\Downloads -Recurse -File |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-90) } |
    Remove-Item -Force
```

### 5. Browser Cache
```powershell
# Chrome cache
$chromePath = "$env:LOCALAPPDATA\Google\Chrome\User Data\Default\Cache"
if (Test-Path $chromePath) {
    $size = (Get-ChildItem $chromePath -Recurse -ErrorAction SilentlyContinue |
             Measure-Object Length -Sum).Sum / 1GB
    Write-Host "Chrome Cache: $([math]::Round($size,2)) GB"
    Remove-Item "$chromePath\*" -Recurse -Force -ErrorAction SilentlyContinue
}

# Edge cache
$edgePath = "$env:LOCALAPPDATA\Microsoft\Edge\User Data\Default\Cache"
if (Test-Path $edgePath) {
    Remove-Item "$edgePath\*" -Recurse -Force -ErrorAction SilentlyContinue
}
```

### 6. Windows Update filer
```powershell
# Sjekk SoftwareDistribution
$updatePath = "C:\Windows\SoftwareDistribution\Download"
$size = (Get-ChildItem $updatePath -Recurse -ErrorAction SilentlyContinue |
         Measure-Object Length -Sum).Sum / 1GB
Write-Host "Windows Update: $([math]::Round($size,2)) GB"

# Slett (krever admin):
Stop-Service wuauserv
Remove-Item "$updatePath\*" -Recurse -Force -ErrorAction SilentlyContinue
Start-Service wuauserv
```

---

## ⚡ RASK FIX (Hvis du har lite tid)

### 1-Click Cleanup Script
```powershell
# Kjør dette for å frigjøre ~10-20 GB raskt (krever admin):

# Stop tjenester
Stop-Service wuauserv -ErrorAction SilentlyContinue

# Cleanup
Remove-Item "$env:TEMP\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "C:\Windows\Temp\*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "C:\Windows\SoftwareDistribution\Download\*" -Recurse -Force -ErrorAction SilentlyContinue
Clear-RecycleBin -Force -ErrorAction SilentlyContinue

# Restart tjenester
Start-Service wuauserv -ErrorAction SilentlyContinue

# Docker cleanup
docker system prune -a --volumes -f

# WSL shutdown og compact
wsl --shutdown

Write-Host "`n✅ Cleanup ferdig! Sjekk disk space nå." -ForegroundColor Green
```

---

## 🎯 ANBEFALT STRATEGI

### 1. Installer WinDirStat (5 min)
- Last ned: https://windirstat.net/
- Skann C: drive
- Se visuelt hva som tar plass
- Slett store unødvendige filer

### 2. Kjør Windows Disk Cleanup (10 min)
```powershell
cleanmgr /d C:
# Velg "Clean up system files"
# Kryss av alt
# Klikk OK
```

### 3. Sjekk WSL2 disk (kan være 100+ GB!)
```powershell
# Finn ext4.vhdx filer
Get-ChildItem "$env:LOCALAPPDATA\Packages" -Recurse -Filter ext4.vhdx -ErrorAction SilentlyContinue

# Shutdown og compact
wsl --shutdown
# Høyreklikk ext4.vhdx -> Properties -> Advanced -> Compress
```

### 4. Docker cleanup
```powershell
docker system prune -a --volumes -f
```

---

## 📊 MÅL

- **Minimum**: 20 GB ledig plass (2%)
- **Anbefalt**: 50 GB ledig plass (5%)
- **Ideelt**: 100 GB ledig plass (10%)

---

## ⚠️ MIDLERTIDIG LØSNING

Siden du har 4 GB ledig nå, kan du:

1. ✅ **Lagre små filer** (<1 GB)
2. ⚠️ **Ikke kjør store operasjoner** (docker build, git clone store repos)
3. ⚠️ **Ikke installer store programmer** 
4. 🔴 **Gjør full disk cleanup ASAP** (før du får 0 GB igjen)

---

## 🆘 HVIS ALT FEILER

Hvis du ikke finner hva som tar plass:

1. **Move data til annen disk**:
   ```powershell
   # Flytt quantum_trader til D: (hvis du har)
   Move-Item C:\quantum_trader D:\quantum_trader
   ```

2. **Ekstern disk**: Kjøp USB disk og flytt store filer

3. **Cloud storage**: Flytt gamle filer til OneDrive/Google Drive

4. **Reinstall Windows**: Siste utvei hvis disken er full av system filer

---

## 📝 STATUS CHECK

```powershell
# Kjør dette for å sjekke status:
$drive = Get-Volume -DriveLetter C
Write-Host "Ledig: $([math]::Round($drive.SizeRemaining/1GB,2)) GB av $([math]::Round($drive.Size/1GB,2)) GB"

# Hvis <10 GB: 🔴 KRITISK - Må rydde NÅ
# Hvis 10-20 GB: ⚠️ ADVARSEL - Rydd snart
# Hvis >20 GB: ✅ OK - Men monitor
```

---

**NESTE STEG**: Kjør WinDirStat for å finne de største filene, eller bruk PowerShell kommandoene over.

**Mål**: Frigjør minst 20 GB ekstra plass!
