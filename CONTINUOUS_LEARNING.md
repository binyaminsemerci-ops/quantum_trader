# Kontinuerlig AI Læring - Implementasjonsguide

## 🎯 Oversikt

Quantum Trader har nå et komplett system for **kontinuerlig læring (continuous learning)** hvor AI-modellen automatisk forbedres basert på faktiske handelsresultater.

## 🏗️ Arkitektur

### 1. Data Samling
Hver gang AI tar en handelsbeslutning:
- ✅ Features (100+ indikatorer) lagres i database
- ✅ Prediction (BUY/SELL/HOLD + confidence) lagres
- ✅ Utførelse (entry pris, mengde, tidspunkt) lagres

Når posisjon lukkes:
- ✅ Exit pris og tidspunkt oppdateres
- ✅ Faktisk P&L beregnes
- ✅ Target label (% return) beregnes for trening

### 2. Database Modeller

**AITrainingSample**: Lagrer hver AI-prediction med outcome
```python
- symbol, timestamp
- predicted_action, confidence
- features (JSON array)
- entry_price, entry_time
- exit_price, realized_pnl
- target_label (% return)
- target_class (WIN/LOSS/NEUTRAL)
```

**AIModelVersion**: Sporer ulike model-versjoner
```python
- version_id (f.eks. v20251112_150000)
- training_samples, trained_at
- train_accuracy, validation_accuracy
- is_active (hvilken modell er i bruk)
- live_accuracy, total_pnl
```

### 3. Automatisk Retraining

**Scheduler Job**: Kjører hver natt kl 03:00 UTC
- Henter fullførte samples fra siste 30 dager
- Krever minimum 100 samples for retraining
- Bygger X (features) og y (% return) dataset
- Splitter 80/20 train/validation
- Trener ny XGBoost modell
- Validerer og lagrer performance metrics
- Lagrer ny modell som `xgb_model_v{timestamp}.pkl`

**Sikkerhetsmekanisme**: Ny modell aktiveres IKKE automatisk
- Du må manuelt aktivere etter å ha vurdert performance
- Forhindrer at dårlige modeller deployes automatisk

## 📡 API Endpoints

### Trigger Manuell Retraining
```bash
POST http://localhost:8000/ai/retrain?min_samples=100
X-Admin-Token: live-admin-token

Response:
{
  "status": "success",
  "version_id": "v20251112_150000",
  "training_samples": 250,
  "validation_samples": 62,
  "train_accuracy": 0.68,
  "validation_accuracy": 0.62,
  "train_mae": 0.0245,
  "validation_mae": 0.0312,
  "model_path": "ai_engine/models/xgb_model_v20251112_150000.pkl",
  "message": "New model trained and saved. Activate via /ai/activate-model/v20251112_150000"
}
```

### List Alle Model-Versjoner
```bash
GET http://localhost:8000/ai/models
X-Admin-Token: live-admin-token

Response:
{
  "status": "ok",
  "count": 5,
  "models": [
    {
      "version_id": "v20251112_150000",
      "model_type": "xgboost_continuous",
      "trained_at": "2025-11-12T15:00:00Z",
      "training_samples": 250,
      "train_accuracy": 0.68,
      "validation_accuracy": 0.62,
      "is_active": false,
      "total_predictions": 0,
      "live_accuracy": null,
      "total_pnl": 0.0
    }
  ]
}
```

### Aktiver Ny Modell
```bash
POST http://localhost:8000/ai/activate-model/v20251112_150000
X-Admin-Token: live-admin-token

Response:
{
  "status": "success",
  "activated_version": "v20251112_150000",
  "model_type": "xgboost_continuous",
  "train_accuracy": 0.68,
  "validation_accuracy": 0.62,
  "message": "Model activated. Restart backend to load new model."
}
```

**Etter aktivering**: Restart backend
```powershell
cd backend
.\stop_backend.ps1
.\start_live.ps1
```

### Se Training Samples
```bash
GET http://localhost:8000/ai/training-samples?limit=50&outcome_known=true
X-Admin-Token: live-admin-token

Response:
{
  "status": "ok",
  "count": 50,
  "samples": [
    {
      "id": 123,
      "symbol": "BTCUSDC",
      "timestamp": "2025-11-12T10:30:00Z",
      "predicted_action": "BUY",
      "prediction_confidence": 0.75,
      "executed": true,
      "execution_side": "BUY",
      "entry_price": 45000.0,
      "exit_price": 45500.0,
      "realized_pnl": 50.0,
      "target_label": 0.0111,  // 1.11% return
      "target_class": "WIN",
      "outcome_known": true
    }
  ]
}
```

## ⚙️ Konfigurasjon

### Environment Variables

**Aktiver/deaktiver automatisk retraining:**
```bash
QT_AI_RETRAINING_ENABLED=1  # 1=enabled, 0=disabled
```

**Schedule (standard: daglig kl 03:00 UTC):**
For å endre, rediger `backend/utils/scheduler.py` linje ~435:
```python
scheduler.add_job(
    _run_ai_retraining,
    "cron",
    hour=3,      # UTC time
    minute=0,
    id="ai-retraining",
)
```

Andre schedule-eksempler:
```python
# Hver 6. time:
scheduler.add_job(_run_ai_retraining, "interval", hours=6)

# Hver søndag kl 02:00:
scheduler.add_job(_run_ai_retraining, "cron", day_of_week="sun", hour=2)

# To ganger daglig (06:00 og 18:00):
scheduler.add_job(_run_ai_retraining, "cron", hour="6,18", minute=0)
```

## 🔄 Workflow

### Dag 1-7: Data Samling
1. Backend kjører live trading
2. AI tar beslutninger (BUY/SELL/HOLD)
3. Ordre utføres basert på AI-signaler
4. Features + predictions lagres i database
5. Posisjoner lukkes etter en stund
6. P&L oppdateres i training samples

### Dag 7+: Første Retraining
1. Scheduler trigger retraining kl 03:00 UTC
2. System sjekker: 100+ samples med outcome?
3. Hvis JA: Bygg dataset fra samples
4. Train ny XGBoost modell
5. Valider på holdout set (20%)
6. Lagre ny modell med versjon ID
7. Send notification (via logs)

### Manual Review & Activation
1. Sjekk logs for retraining resultater
2. GET `/ai/models` - sammenlign accuracy
3. Hvis ny modell er bedre:
   - POST `/ai/activate-model/{version_id}`
   - Restart backend
4. Hvis ikke, la gammel modell være aktiv

### Kontinuerlig Forbedring
1. Ny aktiv modell tar bedre beslutninger
2. Samler mer data med forbedret accuracy
3. Neste retraining bruker bedre data
4. Modellen blir stadig smartere 📈

## 📊 Performance Tracking

### Metrics å overvåke:

**Training Metrics:**
- `train_accuracy`: Accuracy på training set
- `validation_accuracy`: Accuracy på validation set (viktigst!)
- `train_mae`: Mean Absolute Error (hvor mye predictions feiler med)
- `validation_mae`: MAE på validation (lavere = bedre)

**Live Metrics:**
- `total_predictions`: Antall predictions gjort med denne modellen
- `correct_predictions`: Hvor mange var korrekte
- `live_accuracy`: Real-world accuracy
- `total_pnl`: Total profit/loss med denne modellen

### Hva er "god" accuracy?

- **>60%**: Bra! Bedre enn tilfeldig (50%)
- **>65%**: Veldig bra - modellen lærer patterns
- **>70%**: Utmerket - sterk predictive power
- **>75%**: Fantastisk - profesjonelt nivå

**Viktig**: `validation_accuracy` er mer pålitelig enn `train_accuracy`!
- Høy train_accuracy men lav validation = overfitting
- Validation accuracy viser hvordan modellen håndterer ny, usett data

## 🛠️ Setup & Testing

### 1. Kjør Database Migrations
```powershell
cd backend
alembic upgrade head
```

### 2. Verifiser Tabeller Opprettet
```powershell
sqlite3 backend/quantum_trader.db
.tables
# Should see: ai_training_samples, ai_model_versions
.quit
```

### 3. Start Backend med AI Retraining
```powershell
cd backend
$env:QT_AI_RETRAINING_ENABLED = "1"
.\start_live.ps1
```

### 4. La System Samle Data
Vent 1-2 uker for å samle nok data (min 100 samples med outcomes).

### 5. Trigger Manuell Retraining (Testing)
```powershell
curl -X POST http://localhost:8000/ai/retrain?min_samples=10 `
  -H "X-Admin-Token: live-admin-token"
```

*Note: `min_samples=10` kun for testing. I produksjon bruk 100+*

### 6. Sjekk Resultater
```powershell
curl http://localhost:8000/ai/models `
  -H "X-Admin-Token: live-admin-token"
```

### 7. Aktiver Beste Modell
```powershell
curl -X POST http://localhost:8000/ai/activate-model/v20251112_150000 `
  -H "X-Admin-Token: live-admin-token"

# Restart backend
.\stop_backend.ps1
.\start_live.ps1
```

## 🚨 Troubleshooting

### "insufficient_samples"
**Problem**: Ikke nok data for retraining
**Løsning**: Vent lengre eller senk `min_samples` midlertidig

### "too_few_valid_samples"
**Problem**: Mange samples men få har outcome_known=True
**Løsning**: Posisjoner har ikke lukket ennå. Vent eller lukk manuelt.

### "Model retraining failed"
**Problem**: Error under training
**Løsning**: Sjekk logs for detaljer. Vanlige årsaker:
- Feature mismatch (antall features endret)
- Corrupt data (NaN/Inf values)
- Memory issues (for mange samples)

### Ny modell presterer dårligere
**Problem**: Validation accuracy lavere enn gammel modell
**Løsning**:
- IKKE aktiver ny modell
- La gammel modell samle mer data
- Vent på neste retraining cycle
- Vurder å justere training parameters

## 📈 Best Practices

### 1. **Vent med aktivering**
Ikke aktiver ny modell med en gang. Overvåk:
- Validation accuracy > current model
- Validation MAE < current model
- Training samples > 200 (mer data = bedre)

### 2. **A/B Testing (avansert)**
Kjør to backends samtidig:
- Backend A: Gammel modell
- Backend B: Ny modell (paper trading)
- Sammenlign P&L etter 1 uke
- Aktiver beste modell

### 3. **Backup modeller**
Alle modeller lagres permanent:
- `ai_engine/models/xgb_model_v{timestamp}.pkl`
- Du kan alltid gå tilbake til tidligere versjon
- POST `/ai/activate-model/{old_version_id}`

### 4. **Monitor live accuracy**
Selv om en modell hadde god validation accuracy, kan live accuracy avvike.
Hvis live accuracy faller under 55%, vurder å:
- Deaktivere modellen
- Gå tilbake til tidligere versjon
- Retrain med mer variert data

### 5. **Feature engineering**
Hvis modellen ikke forbedres, vurder:
- Legge til nye features (flere indikatorer)
- Fjerne irrelevante features
- Normalize features annerledes
- Endre lookback perioder

## 🎓 Summary

Du har nå et **selvstendig lærende AI-system** som:

✅ **Samler data** fra hver handel automatisk  
✅ **Trener nye modeller** hver natt på akkumulert data  
✅ **Validerer** modeller før deployment  
✅ **Sporer performance** av hver modell-versjon  
✅ **Lar deg aktivere** beste modell manuelt  
✅ **Forbedres kontinuerlig** over tid  

AI-en vil bli **smartere og smartere** jo lenger den kjører! 🚀

---

**Opprettet**: 2025-11-12  
**Status**: ✅ Implementert - Klar for testing  
**Next**: Kjør live i 1-2 uker, deretter første retraining
