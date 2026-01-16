# 🔴 CLM TRAINING ISSUE - CRITICAL BUG

**Dato:** 18. desember 2025, 11:42 UTC  
**Status:** 🔴 **KRITISK BUG FUNNET - INGEN TRENINGER KJØRER**

---

## 🔍 PROBLEM OPPSUMMERING

CLM v3 service kjører men **ingen modell-treninger har skjedd**:

```
✅ CLM container: Running (Up 30 minutes)
✅ Scheduler: Started
✅ Orchestrator: Initialized
❌ Training Jobs: 0 created
❌ Training Logs: 0 entries
❌ Runtime Directory: Doesn't exist
❌ Model Retraining: NOT HAPPENING
```

---

## 🐛 ROOT CAUSE - KRITISK ARKITEKTUR GAP

### Problem Flow:

**Hva som SKULLE skje:**
```
Scheduler (hver 30 min)
    ↓
Sjekker om trening nødvendig
    ↓
Lager TrainingJob (status="pending")
    ↓
Registrerer i ModelRegistry
    ↓
[MANGLER] Job Poller henter pending jobs
    ↓
Sender til Orchestrator
    ↓
Orchestrator kjører training pipeline
```

**Hva som FAKTISK skjer:**
```
Scheduler (hver 30 min)
    ↓
Sjekker om trening nødvendig (✅)
    ↓
Lager TrainingJob (✅ status="pending")
    ↓
Registrerer i ModelRegistry (✅)
    ↓
[GAP] 🔴 INGEN POLLER EKSISTERER!
    ↓
Job blir liggende som "pending" for alltid
    ↓
Orchestrator får aldri jobben
    ↓
INGEN TRENING SKJER ❌
```

---

## 📂 CODE ANALYSIS

### 1. Scheduler (`backend/services/clm_v3/scheduler.py`)

**✅ FUNGERER:**
```python
async def _scheduler_loop(self):
    """Main scheduler loop - checks for training needs periodically."""
    interval_seconds = self.config["check_interval_minutes"] * 60  # 30 min
    
    while self._running:
        try:
            await self._check_periodic_training()  # ✅ Denne kjører
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
        
        await asyncio.sleep(interval_seconds)

async def trigger_training(...) -> TrainingJob:
    """Manually trigger a training job."""
    job = TrainingJob(
        model_type=model_type,
        status="pending",  # ✅ Lager job med status "pending"
        ...
    )
    
    # Register in registry
    job = self.registry.register_training_job(job)  # ✅ Registrerer i registry
    
    logger.info(f"Created training job {job.id}")
    
    return job  # ✅ Returnerer job... MEN INGEN POLLER DEN FRA REGISTRY!
```

**❌ PROBLEMET:**
Ingen kode som sier "fetch all pending jobs from registry and run them".

---

### 2. Orchestrator (`backend/services/clm_v3/orchestrator.py`)

**✅ FUNGERER:**
```python
async def handle_training_job(self, job: TrainingJob) -> Optional[ModelVersion]:
    """
    Handle complete training job pipeline.
    
    Steps:
    1. Update job status to in_progress
    2. Fetch training data
    3. Train model
    4. Save model version to registry
    5. Evaluate model (backtest)
    6. Make promotion decision
    """
    logger.info(f"Starting training job {job.id}...")
    
    # This method is PERFECT, but it's NEVER CALLED!
    ...
```

**❌ PROBLEMET:**
`handle_training_job()` kalles bare fra:
1. `handle_manual_training()` event (ingen events kommer)
2. **Ikke noe background task som poller pending jobs fra registry!**

---

### 3. Main Service (`backend/services/clm_v3/main.py`)

**Hva som mangler:**
```python
# MISSING CODE:

async def _job_processor_loop(self):
    """
    Background task that polls pending jobs from registry
    and sends them to orchestrator.
    """
    while self._running:
        try:
            # Get pending jobs
            pending_jobs = self.registry.get_pending_jobs()
            
            for job in pending_jobs:
                # Start training in background
                asyncio.create_task(self.orchestrator.handle_training_job(job))
        
        except Exception as e:
            logger.error(f"Error in job processor: {e}")
        
        await asyncio.sleep(60)  # Check every minute
```

**Dette finnes IKKE i koden!**

---

## 🧪 VERIFICATION - LOGS ANALYSIS

### Hva vi så i loggene:

```bash
$ journalctl -u quantum_clm.service 2>&1 | wc -l
58 lines total  # ❌ Kun startup logs, ingen aktivitet!

$ journalctl -u quantum_clm.service | grep "training\|retrain\|job"
# Resultat: Kun startup - ingen "Created training job", "Starting training job", etc.

$ ls runtime/clm_v3/
No such directory  # ❌ Ingen runtime metadata lagret!
```

### Hva vi SKULLE sett:

```
[CLM v3 Scheduler] Checking periodic training...
[CLM v3 Scheduler] Periodic training due for xgboost_main (168h interval)
[CLM v3 Scheduler] Created training job abc123 (model=xgboost, trigger=periodic)
[CLM v3 Orchestrator] Starting training job abc123...
[CLM v3 Orchestrator] Fetching training data...
[CLM v3 Orchestrator] Training model...
[CLM v3 Orchestrator] Model trained successfully - Sharpe: 1.45
[CLM v3 Orchestrator] Promoting to CANDIDATE status
```

**Men ingen av dette skjer!**

---

## 🛠️ LØSNING

### Option A: Legg til Job Poller (ANBEFALT)

Legg til en background task i `main.py` som kontinuerlig poller pending jobs:

```python
# backend/services/clm_v3/main.py

class ClmServiceV3:
    def __init__(self, ...):
        ...
        self._job_processor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start CLM service."""
        ...
        
        # Start scheduler
        await self.scheduler.start()
        
        # 🆕 Start job processor
        self._job_processor_task = asyncio.create_task(self._job_processor_loop())
        
        logger.info("[CLM v3] ✅ Service started")
    
    async def _job_processor_loop(self):
        """
        Background task that processes pending training jobs.
        Polls registry every minute for pending jobs and sends to orchestrator.
        """
        while self._running:
            try:
                # Get all pending jobs
                pending_jobs = self.registry.get_jobs_by_status("pending")
                
                if pending_jobs:
                    logger.info(
                        f"[CLM v3 Job Processor] Found {len(pending_jobs)} pending jobs"
                    )
                
                for job in pending_jobs:
                    # Check if already being processed (status might be stale)
                    current_job = self.registry.get_training_job(job.id)
                    if current_job.status != "pending":
                        continue  # Already started
                    
                    logger.info(
                        f"[CLM v3 Job Processor] Starting job {job.id} "
                        f"(model={job.model_type.value}, trigger={job.trigger_reason.value})"
                    )
                    
                    # Start training in background
                    asyncio.create_task(self.orchestrator.handle_training_job(job))
                    
                    # Small delay to avoid race conditions
                    await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(
                    f"[CLM v3 Job Processor] Error processing jobs: {e}",
                    exc_info=True
                )
            
            # Wait before next poll
            await asyncio.sleep(60)  # Poll every minute
    
    async def stop(self):
        """Stop CLM service."""
        logger.info("[CLM v3] Stopping service...")
        self._running = False
        
        # Stop scheduler
        await self.scheduler.stop()
        
        # 🆕 Stop job processor
        if self._job_processor_task:
            self._job_processor_task.cancel()
            try:
                await self._job_processor_task
            except asyncio.CancelledError:
                pass
        
        ...
```

**Fordeler:**
- ✅ Clean separation of concerns
- ✅ Scheduler lager jobs, poller konsumerer dem
- ✅ Skalerer bra (flere jobs kan køes opp)
- ✅ Jobs persisteres i registry (overlever restarts)

---

### Option B: Kjør job direkte fra scheduler

Alternativ: Kjør `handle_training_job()` direkte fra scheduler uten registry polling:

```python
# backend/services/clm_v3/scheduler.py

async def trigger_training(...) -> TrainingJob:
    """Trigger training and start it immediately."""
    job = TrainingJob(...)
    
    # Register in registry
    job = self.registry.register_training_job(job)
    
    logger.info(f"Created training job {job.id}")
    
    # 🆕 Start training immediately (requires orchestrator reference)
    if hasattr(self, 'orchestrator') and self.orchestrator:
        asyncio.create_task(self.orchestrator.handle_training_job(job))
    
    return job
```

**Ulemper:**
- ❌ Scheduler må ha referanse til orchestrator (circular dependency)
- ❌ Jobs som feiler blir ikke retry-et
- ❌ Hvis service restarter, pending jobs går tapt

**Option A er bedre!**

---

## 📊 IMPACT ASSESSMENT

### Hva dette betyr:

**Siden systemet startet (30+ minutter geleden):**
- ❌ 0 modell-treninger har skjedd
- ❌ Ingen XGBoost retraining
- ❌ Ingen LGBM retraining
- ❌ Ingen NHITS/PatchTST retraining
- ❌ Ingen RL v3 agent retraining

**CLM status:**
```
Scheduler: ✅ Running (checking every 30 min)
Registry:  ✅ Initialized (0 jobs registered)
Orchestrator: ✅ Ready (waiting for jobs)
Job Poller: ❌ MISSING - THIS IS THE GAP!
```

**Modellene kjører fortsatt:**
- AI Engine bruker OLD modeller (fra disk)
- Modellene har IKKE fått nye data fra testnet trading
- CLM lærer IKKE fra nye trades

**Dette er OK for nå** (testnet evaluation period) men må fikses før:
1. Extended testnet evaluation (trenger continuous learning)
2. Mainnet deployment (CRITICAL)

---

## ✅ ACTION PLAN

### Immediate (Kritisk):
1. ✅ **Legg til job poller** i `main.py` (Option A)
2. ✅ **Deploy til server**
3. ✅ **Restart CLM container**
4. ✅ **Verify at jobs blir opprettet og kjørt**

### Verification Steps:
```bash
# 1. Check scheduler creates jobs
journalctl -u quantum_clm.service | grep "Created training job"

# 2. Check job processor picks them up
journalctl -u quantum_clm.service | grep "Job Processor.*Starting job"

# 3. Check orchestrator starts training
journalctl -u quantum_clm.service | grep "Orchestrator.*Starting training"

# 4. Check runtime directory is created
ls -la runtime/clm_v3/

# 5. Check for trained models
ls -la models/trained/
```

### Follow-up (Post-Deploy):
1. Monitor first training job completion
2. Verify backtest evaluation runs
3. Check model promotion logic
4. Confirm metadata is saved to registry

---

## 🏁 KONKLUSJON

**CRITICAL BUG FUNNET:**
- CLM Scheduler lager TrainingJobs men ingen poller kjører dem
- Ingen modell-treninger har skjedd i 30+ minutter
- Job processor task mangler i arkitekturen

**LØSNING:**
- Legg til `_job_processor_loop()` background task
- Poller pending jobs fra registry hver minutt
- Sender jobs til orchestrator for utførelse

**NEXT STEP:**
Implementere Option A (job poller) og deploye til server.

---

**Rapport generert:** 2025-12-18 11:42 UTC  
**Oppdaget av:** GitHub Copilot Agent  
**Severity:** 🔴 CRITICAL (ingen kontinuerlig læring skjer)

