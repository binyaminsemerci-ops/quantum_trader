# 🤖 QUANTUM TRADER AUTO TRAINING SETUP

Dette systemet automatisk trener AI-modellen når PC-en starter og hver time etterpå.

## 🚀 QUICK SETUP (1 minutt)

### Steg 1: Installer (Anbefalt - ingen admin nødvendig)
```powershell
cd C:\quantum_trader
.\setup_simple.ps1 startup
```

### Steg 2: Sjekk status  
```powershell
.\setup_simple.ps1 status
```

### Steg 3: Test manuelt (valgfritt)
```powershell
.\setup_simple.ps1 test
```

## 📋 HVA SYSTEMET GJØR

### 🔄 **Automatisk Schedule:**
- **Ved oppstart**: Venter 2 minutter, så starter initial training (1200 samples)
- **Hver time**: Oppdaterer modellen med fresh data (1200 samples)
- **Kun når online**: Krever internett-tilkobling

### 📊 **Data Sources:**
- **Binance**: Live OHLCV cryptocurrency data
- **Twitter**: Sentiment analysis (hvis tilgjengelig)
- **CoinGecko**: Live market prices
- **Synthetic**: Fallback hvis live data feiler

### 📂 **Logging:**
- **Sted**: `C:\quantum_trader\logs\auto_training\`
- **Format**: `startup_YYYYMMDD_HHMM.log` og `hourly_YYYYMMDD_HHMM.log`
- **Retention**: Automatisk sletting etter 24 timer

## 🛠️ ADMINISTRASJON

### Sjekk status:
```powershell
.\setup_auto_training.ps1 -Status
```

### Stopp auto training:
```powershell
.\setup_auto_training.ps1 -Uninstall
```

### Manuell training:
```batch
**Rask training (100 samples):**
```bat
.\start_training_optimized.bat 100
```

**Normal training (1200 samples):**
```bat
.\start_training_optimized.bat
```

**Full training (2000 samples):**
```bat
.\start_training_optimized.bat full
```
```

### Se siste resultater:
```batch
python main_train_and_backtest.py report
```

## 📈 FORVENTET BEHAVIOR

### ✅ **Første oppstart:**
```
🚀 PC starter → 2 min delay → Initial AI training (1200 samples)
✅ Model lagret til ai_engine\models\
⏰ Venter til neste time → Hourly update
```

### ✅ **Kontinuerlig drift:**
```
⏰ Hver time: Training (1200 samples) → Modell oppdatert
📊 Accuracy typisk: 75-85% (høyere med mer data!)
🤖 Model alltid fresh med nyeste markedsdata
```

### ✅ **Dashboard integration:**
- Dashboard bruker alltid nyeste trente modell
- Live signals blir kontinuerlig forbedret
- Portfolio tracking blir mer nøyaktig over tid

## 🎯 RESULTAT

AI-modellen din blir **automatisk forbedret 24/7** uten manuell inngripen!

- 📈 **Bedre prediksjoner** med kontinuerlige oppdateringer
- 🤖 **Alltid fresh model** med nyeste markedsdata  
- 🔄 **Zero maintenance** - fungerer automatisk
- 📊 **Optimalisert performance** med timelige updates

---

**Tip**: Bruk `.\setup_auto_training.ps1 -Status` for å overvåke systemet! 🚀