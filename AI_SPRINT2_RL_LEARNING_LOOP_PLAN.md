# Sprint 2: RL Learning Loop - Implementation Plan
**Status:** 🚀 READY TO START  
**Duration:** 5-7 days  
**Prerequisites:** ✅ Tier 1 Core Loop validated (100% tests passed)

---

## Executive Summary

Sprint 2 builds the **Reinforcement Learning (RL) feedback loop** on top of Tier 1's execution pipeline. This enables the system to:
- Learn from trade outcomes in real-time
- Optimize position sizing using PPO (Proximal Policy Optimization)
- Detect model drift and trigger automatic retraining
- Close the AI learning loop: Signal → Execute → Learn → Improve

### Architecture Overview
```
AI Engine v5 (DONE) → EventBus (DONE) → Risk Safety (DONE) → Execution (DONE) → Position Monitor (DONE)
       ↓                                                                                    ↓
   Publishes                                                                         PnL tracking
   signals                                                                                ↓
                                                                              RL Feedback Bridge (NEW)
                                                                                         ↓
                                                                              PPO Position Sizer (NEW)
                                                                                         ↓
                                                                              CLM Drift Detector (NEW)
                                                                                         ↓
                                                                              Auto-retrain triggers (NEW)
```

---

## Sprint 2 Deliverables

### Phase 2.1: AI Engine Integration (Day 1) ✅ IN PROGRESS
**Status:** Code committed, ready for deployment testing

**Objective:** Activate signal publishing from AI Engine to EventBus

**Implementation:** (DONE)
- ✅ Modified `ensemble_manager.py` with EventBus client
- ✅ Added async `_publish_to_eventbus()` method
- ✅ TradeSignal publishing to `trade.signal.v5` topic
- ✅ Non-blocking async task creation (no performance impact)
- ✅ Ensemble metadata included (votes, meta override, governer approval)
- ✅ Test script created (`test_eventbus_integration.py`)

**Deployment Steps:**
```bash
# On VPS
cd /home/qt/quantum_trader
git pull
source /opt/quantum/venvs/ai-engine/bin/activate

# Test EventBus integration
python3 test_eventbus_integration.py

# Verify AI Engine can publish signals
# (Integration will be fully activated when AI scanner is running)
```

**Success Criteria:**
- ✅ EnsembleManager initializes with EventBus enabled
- ⏳ Test script publishes signal to Redis (pending VPS deployment)
- ⏳ No performance degradation in prediction latency
- ⏳ Signals include all metadata (votes, confidence, governer)

---

### Phase 2.2: RL Feedback Bridge (Day 2-3)
**Status:** 🔴 NOT STARTED

**Objective:** Subscribe to trade outcomes and calculate RL rewards

**Files to Create:**
1. **`ai_engine/services/rl_feedback_bridge.py`** (400 lines)
   - Subscribe to `trade.execution.res` (entry prices)
   - Subscribe to `trade.position.update` (PnL updates)
   - Track trade lifecycle (entry → updates → exit)
   - Calculate rewards (Sharpe ratio, win rate, profit factor)
   - Publish to `rl.feedback` topic

2. **`ai_engine/rl/reward_calculator.py`** (200 lines)
   - Sharpe ratio calculation
   - Risk-adjusted returns
   - Max drawdown penalties
   - Duration penalties (favor quick profits)

**Implementation Details:**
```python
# ai_engine/services/rl_feedback_bridge.py
class RLFeedbackBridge:
    """Track trade outcomes and calculate RL rewards"""
    
    def __init__(self):
        self.eventbus = EventBusClient()
        self.active_trades = {}  # order_id -> trade_data
        
    async def execution_consumer(self):
        """Track trade entries"""
        async for result_data in self.eventbus.subscribe("trade.execution.res"):
            result = ExecutionResult(**result_data)
            
            # Track entry
            self.active_trades[result.order_id] = {
                "entry_time": datetime.fromisoformat(result.timestamp),
                "entry_price": result.entry_price,
                "symbol": result.symbol,
                "action": result.action,
                "size_usd": result.position_size_usd,
                "confidence": result.confidence,  # From original signal
                "ensemble_votes": result.ensemble_votes
            }
    
    async def position_consumer(self):
        """Track position updates and calculate rewards"""
        async for update_data in self.eventbus.subscribe("trade.position.update"):
            update = PositionUpdate(**update_data)
            
            # Find matching trade
            order_id = self._find_order_id(update.symbol, update.side)
            if not order_id:
                continue
            
            trade = self.active_trades[order_id]
            
            # Calculate current PnL
            pnl_usd = update.unrealized_pnl
            pnl_pct = pnl_usd / trade["size_usd"]
            
            # Calculate duration
            duration_mins = (datetime.utcnow() - trade["entry_time"]).total_seconds() / 60
            
            # Calculate Sharpe contribution (reward signal)
            sharpe = pnl_pct / np.sqrt(duration_mins / 60)  # Annualized volatility proxy
            
            # If position closed, publish final reward
            if update.side == "CLOSED":
                reward = self._calculate_final_reward(trade, update)
                
                await self.eventbus.publish("rl.feedback", {
                    "order_id": order_id,
                    "symbol": update.symbol,
                    "pnl_usd": update.realized_pnl,
                    "pnl_pct": pnl_pct,
                    "duration_mins": duration_mins,
                    "sharpe_contribution": sharpe,
                    "reward": reward,
                    "confidence": trade["confidence"],
                    "ensemble_votes": trade["ensemble_votes"],
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                })
                
                # Remove from active trades
                del self.active_trades[order_id]
```

**Reward Function Design:**
```python
# ai_engine/rl/reward_calculator.py
def calculate_final_reward(trade_data, position_update):
    """
    Calculate RL reward for completed trade.
    
    Reward components:
    1. Profit/loss (primary signal)
    2. Risk-adjusted return (Sharpe ratio)
    3. Duration penalty (favor quick exits)
    4. Drawdown penalty (penalize large unrealized losses)
    5. Confidence alignment (did high confidence → high PnL?)
    """
    pnl_pct = position_update.realized_pnl / trade_data["size_usd"]
    duration_mins = ...
    max_drawdown = trade_data.get("max_drawdown", 0)
    confidence = trade_data["confidence"]
    
    # Base reward: PnL percentage
    reward = pnl_pct * 100  # Scale to [-100, +100] range
    
    # Duration penalty: Favor quick profits
    if pnl_pct > 0:
        duration_penalty = min(0.2, duration_mins / 1440)  # Max 20% penalty after 1 day
        reward *= (1 - duration_penalty)
    
    # Drawdown penalty: Penalize large swings
    if max_drawdown < -0.02:  # If dropped >2% from entry
        reward *= 0.8  # 20% penalty
    
    # Confidence alignment: High confidence should → high PnL
    if confidence > 0.80 and pnl_pct < 0:
        reward *= 0.5  # Heavily penalize high-confidence losses
    
    # Sharpe bonus: Extra reward for risk-adjusted returns
    sharpe = pnl_pct / np.sqrt(duration_mins / 60)
    if sharpe > 1.0:
        reward *= 1.2  # 20% bonus for Sharpe > 1
    
    return reward
```

**Deployment:**
```bash
# Create systemd service
sudo tee /etc/systemd/system/quantum-rl-feedback.service << EOF
[Unit]
Description=Quantum Trader RL Feedback Bridge
After=redis.service

[Service]
Type=simple
User=qt
WorkingDirectory=/home/qt/quantum_trader
Environment="PYTHONPATH=/home/qt/quantum_trader"
ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 -m ai_engine.services.rl_feedback_bridge
Restart=always
StandardOutput=append:/var/log/quantum/rl-feedback.log
StandardError=append:/var/log/quantum/rl-feedback.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable quantum-rl-feedback
sudo systemctl start quantum-rl-feedback
```

**Success Criteria:**
- ⏳ All trade outcomes tracked (entry → exit)
- ⏳ Rewards calculated for completed trades
- ⏳ Feedback published to `rl.feedback` topic
- ⏳ Reward function validated (positive trades → positive rewards)

---

### Phase 2.3: PPO Position Sizer (Day 4-5)
**Status:** 🔴 NOT STARTED

**Objective:** Train RL agent to optimize position sizing

**Files to Create:**
1. **`ai_engine/rl/ppo_position_sizer.py`** (500 lines)
   - PPO agent architecture (PyTorch)
   - State space: [confidence, volatility, regime, exposure, drawdown]
   - Action space: position size multiplier (0.5x - 2.0x)
   - Training loop (on-policy updates every N trades)

2. **`ai_engine/rl/rl_trainer.py`** (300 lines)
   - Collect experiences from `rl.feedback` topic
   - Batch training (update every 50 trades)
   - Model checkpointing
   - Tensorboard logging

**PPO Agent Architecture:**
```python
# ai_engine/rl/ppo_position_sizer.py
import torch
import torch.nn as nn

class PPOPositionSizer:
    """
    PPO agent for dynamic position sizing.
    
    Learns to adjust position size based on:
    - Signal confidence
    - Market volatility
    - Current portfolio exposure
    - Recent drawdown
    - Market regime (trending/ranging)
    """
    
    def __init__(self, state_dim=5, action_dim=1, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # Output: [0, 1] → scale to [0.5x, 2.0x]
        )
        
        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        
    def get_action(self, state):
        """
        Get position size multiplier.
        
        Args:
            state: [confidence, volatility, regime, exposure, drawdown]
        
        Returns:
            multiplier: 0.5x to 2.0x (scale default Kelly sizing)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            action = self.policy_net(state_tensor).item()
        
        # Scale [0, 1] → [0.5, 2.0]
        multiplier = 0.5 + (action * 1.5)
        
        return multiplier
    
    def train_step(self, states, actions, rewards, next_states, dones):
        """
        PPO training step (on-policy update).
        
        Uses clipped surrogate objective to prevent large policy updates.
        """
        states_t = torch.tensor(states, dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.float32)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.tensor(next_states, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        
        # Compute returns (discounted rewards)
        returns = self._compute_returns(rewards_t, dones_t, gamma=0.99)
        
        # Compute advantages (A = Q - V)
        values = self.value_net(states_t).squeeze()
        advantages = returns - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO policy loss (clipped surrogate objective)
        old_log_probs = self._compute_log_probs(states_t, actions_t).detach()
        
        for _ in range(10):  # PPO epochs
            new_log_probs = self._compute_log_probs(states_t, actions_t)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            clip_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)  # epsilon = 0.2
            policy_loss = -torch.min(
                ratio * advantages,
                clip_ratio * advantages
            ).mean()
            
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()
        
        # Value loss (MSE between predicted and actual returns)
        value_loss = nn.MSELoss()(values, returns)
        
        self.optimizer_value.zero_grad()
        value_loss.backward()
        self.optimizer_value.step()
        
        return policy_loss.item(), value_loss.item()
```

**Training Loop:**
```python
# ai_engine/rl/rl_trainer.py
async def rl_training_loop():
    """
    Continuous RL training loop.
    
    1. Collect experiences from rl.feedback topic
    2. Batch training every 50 trades
    3. Save checkpoints every 100 trades
    4. Log metrics to Tensorboard
    """
    ppo_agent = PPOPositionSizer()
    experience_buffer = []
    
    async with EventBusClient() as bus:
        async for feedback_data in bus.subscribe("rl.feedback"):
            # Parse feedback
            state = [
                feedback_data["confidence"],
                feedback_data.get("volatility", 0.01),
                feedback_data.get("regime", 0.5),
                feedback_data.get("exposure_pct", 0.10),
                feedback_data.get("drawdown_pct", 0.0)
            ]
            action = feedback_data.get("multiplier_used", 1.0)
            reward = feedback_data["reward"]
            next_state = [...]  # Current market state
            done = True  # Episode ends after each trade
            
            # Store experience
            experience_buffer.append((state, action, reward, next_state, done))
            
            # Train every 50 experiences
            if len(experience_buffer) >= 50:
                states, actions, rewards, next_states, dones = zip(*experience_buffer)
                
                policy_loss, value_loss = ppo_agent.train_step(
                    states, actions, rewards, next_states, dones
                )
                
                logger.info(
                    f"[RL] Training step {len(experience_buffer)} trades | "
                    f"Policy loss={policy_loss:.4f} | "
                    f"Value loss={value_loss:.4f}"
                )
                
                # Clear buffer
                experience_buffer = []
                
                # Save checkpoint
                torch.save(ppo_agent.policy_net.state_dict(), 
                          "/home/qt/quantum_trader/models/ppo_policy_latest.pt")
```

**Integration with GovernerAgent:**
```python
# Modify ai_engine/agents/governer_agent.py
class GovernerAgent:
    def __init__(self, config: RiskConfig):
        self.config = config
        
        # Load RL position sizer if available
        try:
            self.ppo_sizer = PPOPositionSizer()
            self.ppo_sizer.policy_net.load_state_dict(
                torch.load("models/ppo_policy_latest.pt")
            )
            logger.info("[GOVERNER] PPO position sizer loaded")
        except FileNotFoundError:
            self.ppo_sizer = None
            logger.info("[GOVERNER] PPO sizer not found - using static Kelly")
    
    def allocate_position(self, symbol, action, confidence, balance, meta_override):
        # Get base Kelly sizing
        kelly_size_pct = self._calculate_kelly(confidence)
        
        # Apply PPO multiplier if available
        if self.ppo_sizer:
            state = [
                confidence,
                self._get_volatility(symbol),
                self._get_regime(),
                self._get_current_exposure(),
                self._get_current_drawdown()
            ]
            
            multiplier = self.ppo_sizer.get_action(state)
            adjusted_size_pct = kelly_size_pct * multiplier
            
            logger.info(
                f"[RL] Position sizing: Kelly={kelly_size_pct:.2%} × "
                f"PPO={multiplier:.2f} = {adjusted_size_pct:.2%}"
            )
        else:
            adjusted_size_pct = kelly_size_pct
        
        # Apply caps
        final_size_pct = min(adjusted_size_pct, self.config.max_position_size_pct)
        
        return PositionAllocation(
            approved=True,
            position_size_pct=final_size_pct,
            position_size_usd=balance * final_size_pct,
            ...
        )
```

**Success Criteria:**
- ⏳ PPO agent trains on feedback (50-trade batches)
- ⏳ Position sizing improves Sharpe ratio (vs. static Kelly)
- ⏳ Model checkpoints saved after training
- ⏳ Tensorboard metrics logged

---

### Phase 2.4: CLM Drift Detection (Day 6-7)
**Status:** 🔴 NOT STARTED

**Objective:** Detect model drift and trigger auto-retraining

**Files to Create:**
1. **`ai_engine/services/clm_drift_detector.py`** (350 lines)
   - Monitor prediction accuracy vs. actual PnL
   - Track confidence calibration drift
   - K-S test for distribution changes
   - Trigger retrain when drift detected

2. **`ai_engine/clm/retrain_trigger.py`** (200 lines)
   - Queue retraining jobs
   - Notify via EventBus
   - Track retrain history

**Drift Detection Logic:**
```python
# ai_engine/services/clm_drift_detector.py
class CLMDriftDetector:
    """
    Continuous Learning Manager - Drift Detection Component.
    
    Monitors model predictions vs. actual outcomes to detect:
    - Accuracy drift (MAPE increase)
    - Confidence calibration drift (high confidence → poor results)
    - Distribution shift (K-S test on features)
    """
    
    def __init__(self, lookback_trades=100, mape_threshold=0.15, ks_threshold=0.3):
        self.lookback = lookback_trades
        self.mape_threshold = mape_threshold
        self.ks_threshold = ks_threshold
        
        self.recent_predictions = []
        self.recent_outcomes = []
        
    async def monitor_executions(self):
        """Subscribe to execution results and track accuracy"""
        async with EventBusClient() as bus:
            async for result_data in bus.subscribe("trade.execution.res"):
                result = ExecutionResult(**result_data)
                
                # Track prediction
                self.recent_predictions.append({
                    "symbol": result.symbol,
                    "predicted_action": result.action,
                    "confidence": result.confidence,
                    "timestamp": result.timestamp
                })
    
    async def monitor_feedback(self):
        """Subscribe to RL feedback and track outcomes"""
        async with EventBusClient() as bus:
            async for feedback_data in bus.subscribe("rl.feedback"):
                # Track outcome
                self.recent_outcomes.append({
                    "order_id": feedback_data["order_id"],
                    "pnl_pct": feedback_data["pnl_pct"],
                    "confidence": feedback_data["confidence"],
                    "timestamp": feedback_data["timestamp"]
                })
                
                # Check drift every 100 trades
                if len(self.recent_outcomes) >= self.lookback:
                    drift_detected = self._check_drift()
                    
                    if drift_detected:
                        await self._trigger_retrain(drift_detected)
                        
                        # Reset buffers
                        self.recent_predictions = []
                        self.recent_outcomes = []
    
    def _check_drift(self):
        """
        Check for model drift using multiple metrics.
        
        Returns:
            dict: Drift info if detected, None otherwise
        """
        # 1. Calculate MAPE (Mean Absolute Percentage Error)
        predictions = [p["confidence"] for p in self.recent_predictions[-self.lookback:]]
        outcomes = [
            1.0 if o["pnl_pct"] > 0 else 0.0 
            for o in self.recent_outcomes[-self.lookback:]
        ]
        
        mape = np.mean(np.abs(np.array(predictions) - np.array(outcomes)))
        
        # 2. Check confidence calibration (high conf → good results?)
        high_conf_predictions = [
            p for p in self.recent_predictions[-self.lookback:] 
            if p["confidence"] > 0.80
        ]
        high_conf_outcomes = [
            o for o in self.recent_outcomes[-self.lookback:] 
            if o["confidence"] > 0.80
        ]
        
        if high_conf_outcomes:
            high_conf_accuracy = np.mean([
                1.0 if o["pnl_pct"] > 0 else 0.0 
                for o in high_conf_outcomes
            ])
        else:
            high_conf_accuracy = 0.5
        
        # 3. K-S test for distribution shift
        recent_confidences = predictions[-50:]
        older_confidences = predictions[-100:-50] if len(predictions) >= 100 else predictions[:50]
        
        from scipy.stats import ks_2samp
        ks_stat, ks_pvalue = ks_2samp(recent_confidences, older_confidences)
        
        # Detect drift
        drift_info = None
        
        if mape > self.mape_threshold:
            drift_info = {
                "type": "accuracy_drift",
                "mape": mape,
                "threshold": self.mape_threshold,
                "message": f"Accuracy degraded: MAPE={mape:.3f} > {self.mape_threshold}"
            }
        elif high_conf_accuracy < 0.60:
            drift_info = {
                "type": "calibration_drift",
                "high_conf_accuracy": high_conf_accuracy,
                "message": f"High-confidence trades underperforming: {high_conf_accuracy:.1%} win rate"
            }
        elif ks_stat > self.ks_threshold and ks_pvalue < 0.05:
            drift_info = {
                "type": "distribution_shift",
                "ks_stat": ks_stat,
                "ks_pvalue": ks_pvalue,
                "message": f"Feature distribution changed: K-S={ks_stat:.3f}"
            }
        
        if drift_info:
            logger.warning(f"[CLM] DRIFT DETECTED: {drift_info['message']}")
        
        return drift_info
    
    async def _trigger_retrain(self, drift_info):
        """
        Trigger model retraining.
        
        Publishes retrain request to EventBus for training worker to pick up.
        """
        async with EventBusClient() as bus:
            await bus.publish("model.retrain.trigger", {
                "model": "meta_v5",
                "reason": drift_info["type"],
                "drift_info": drift_info,
                "triggered_at": datetime.utcnow().isoformat() + "Z",
                "priority": "high" if drift_info["type"] == "accuracy_drift" else "normal"
            })
        
        logger.info(
            f"[CLM] Retrain triggered: {drift_info['type']} | "
            f"Message published to model.retrain.trigger"
        )
```

**Deployment:**
```bash
# Create systemd service
sudo tee /etc/systemd/system/quantum-clm-drift.service << EOF
[Unit]
Description=Quantum Trader CLM Drift Detector
After=redis.service

[Service]
Type=simple
User=qt
WorkingDirectory=/home/qt/quantum_trader
Environment="PYTHONPATH=/home/qt/quantum_trader"
ExecStart=/opt/quantum/venvs/ai-engine/bin/python3 -m ai_engine.services.clm_drift_detector
Restart=always
StandardOutput=append:/var/log/quantum/clm-drift.log
StandardError=append:/var/log/quantum/clm-drift.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable quantum-clm-drift
sudo systemctl start quantum-clm-drift
```

**Success Criteria:**
- ⏳ Drift detection runs every 100 trades
- ⏳ MAPE, calibration, and K-S tests functional
- ⏳ Retrain triggers published to EventBus
- ⏳ No false positives (verify on real data)

---

## System Integration

After Sprint 2, the complete system flow:

```
1. Market Data → Feature Engineering
2. AI Engine Ensemble → Prediction (action, confidence)
3. ✅ [NEW] AI Engine → EventBus (publish signal)
4. EventBus → Risk Safety (governer validation)
5. Risk Safety → Execution (paper/live trade)
6. Execution → Position Monitor (PnL tracking)
7. ✅ [NEW] Position Monitor → RL Feedback Bridge (track outcomes)
8. ✅ [NEW] RL Feedback Bridge → PPO Agent (learn position sizing)
9. ✅ [NEW] PPO Agent → Governer (optimize Kelly multiplier)
10. ✅ [NEW] CLM Drift Detector → Retrain Trigger (adapt to market changes)
```

**Closed Loop:** AI generates signals → System executes → RL learns → AI improves

---

## Testing Strategy

### Unit Tests
Create `tests/test_sprint2_components.py`:
```python
@pytest.mark.asyncio
async def test_rl_feedback_bridge():
    """Test that RL bridge tracks trade outcomes"""
    # Publish execution result
    # Publish position updates
    # Verify reward calculation
    # Verify feedback published to rl.feedback

@pytest.mark.asyncio
async def test_ppo_position_sizer():
    """Test PPO agent action selection"""
    # Create test state
    # Get multiplier
    # Verify output in range [0.5, 2.0]
    # Test training step

@pytest.mark.asyncio
async def test_clm_drift_detection():
    """Test drift detection logic"""
    # Simulate predictions vs. outcomes
    # Inject drift (MAPE increase)
    # Verify drift detected
    # Verify retrain trigger published
```

### Integration Tests
Create `tests/test_rl_learning_loop.py`:
```python
@pytest.mark.asyncio
async def test_full_rl_loop():
    """Test complete RL learning loop"""
    # 1. AI Engine publishes signal
    # 2. Execution fills order
    # 3. Position Monitor tracks PnL
    # 4. RL Feedback Bridge calculates reward
    # 5. PPO Agent receives feedback
    # 6. Verify learning occurred (policy update)
```

### Performance Tests
- RL training latency: <500ms per batch
- Feedback calculation: <10ms per trade
- Drift detection: <100ms per check
- End-to-end loop: <5s signal → feedback

---

## Monitoring & Observability

### Prometheus Metrics (extend existing exporter)
```python
# Add to services/prometheus_exporter.py

# RL metrics
rl_rewards_total = Counter('quantum_rl_rewards_total', 'Total RL rewards collected')
rl_reward_avg = Gauge('quantum_rl_reward_avg', 'Average RL reward (recent 100)')
rl_training_steps = Counter('quantum_rl_training_steps', 'PPO training steps')
rl_policy_loss = Gauge('quantum_rl_policy_loss', 'PPO policy loss')
rl_value_loss = Gauge('quantum_rl_value_loss', 'PPO value loss')

# CLM metrics
clm_drift_checks = Counter('quantum_clm_drift_checks', 'Drift checks performed')
clm_drifts_detected = Counter('quantum_clm_drifts_detected', 'Drift detections', ['type'])
clm_retrains_triggered = Counter('quantum_clm_retrains_triggered', 'Retrain triggers')
clm_mape = Gauge('quantum_clm_mape', 'Current MAPE')
clm_confidence_calibration = Gauge('quantum_clm_confidence_calibration', 'High-confidence win rate')
```

### Grafana Dashboards
**Dashboard: RL Learning Loop**
- Panel 1: Reward distribution (histogram)
- Panel 2: Cumulative rewards over time
- Panel 3: PPO loss curves (policy + value)
- Panel 4: Position size multipliers (actual vs. PPO suggested)
- Panel 5: Sharpe ratio (before/after RL)

**Dashboard: CLM Drift Detection**
- Panel 1: MAPE over time (with threshold line)
- Panel 2: Confidence calibration (high-conf win rate)
- Panel 3: K-S statistic (feature drift)
- Panel 4: Retrain triggers timeline
- Panel 5: Model version history

---

## Rollout Plan

### Day 1: AI Engine Integration ✅ DONE (Code Ready)
- ✅ Commit: `ee05e31e` (EventBus integration)
- ⏳ Deploy to VPS
- ⏳ Run `test_eventbus_integration.py`
- ⏳ Verify signals published to Redis

### Day 2-3: RL Feedback Bridge
- Create `rl_feedback_bridge.py` + `reward_calculator.py`
- Deploy as systemd service
- Test with existing trade data
- Validate reward calculations

### Day 4-5: PPO Position Sizer
- Create `ppo_position_sizer.py` + `rl_trainer.py`
- Train on historical feedback (if available)
- Integrate with GovernerAgent
- A/B test: Static Kelly vs. PPO-adjusted

### Day 6-7: CLM Drift Detection
- Create `clm_drift_detector.py`
- Deploy as systemd service
- Simulate drift scenarios
- Validate retrain triggers

### Day 8: Integration Testing
- Run full loop: Signal → Execute → Feedback → Learn
- Performance testing (latency, throughput)
- Stress test (100+ trades)
- Documentation update

---

## Success Metrics (Sprint 2)

### Quantitative
- ✅ AI Engine publishes 20+ signals/hour
- ✅ RL Feedback tracks 100% of trade outcomes
- ✅ PPO improves Sharpe ratio by 10-20% (vs. static Kelly)
- ✅ CLM detects drift within 100 trades (95% accuracy)
- ✅ End-to-end latency <5s (signal → feedback)
- ✅ 0 errors in production logs (48-hour stability test)

### Qualitative
- ✅ System learns from mistakes (bad trades → smaller positions)
- ✅ System exploits good signals (high-confidence wins → larger positions)
- ✅ System adapts to market changes (drift detection → retrain → improved accuracy)
- ✅ Monitoring dashboards provide clear visibility
- ✅ Documentation enables team onboarding

---

## Risk Mitigation

### Risk 1: RL Agent Learns Bad Policy
**Mitigation:**
- Conservative action space (0.5x - 2.0x multiplier)
- Clip policy updates (PPO epsilon = 0.2)
- Manual override: GovernerAgent caps position size regardless of PPO
- Rollback: Keep static Kelly as fallback

### Risk 2: Drift Detection False Positives
**Mitigation:**
- Multiple drift metrics (MAPE + calibration + K-S)
- Threshold tuning on historical data
- Manual review before auto-retrain (Phase 1)
- Gradual automation (Phase 2+)

### Risk 3: EventBus Performance Impact
**Mitigation:**
- Non-blocking async publishing
- Redis Streams (high throughput)
- MAXLEN limits prevent memory issues
- Monitoring: Publish latency <10ms

### Risk 4: Training Instability
**Mitigation:**
- Batch training (50-trade batches)
- Experience replay buffer
- Gradient clipping
- Learning rate scheduling

---

## Post-Sprint 2 Roadmap

### Sprint 3: Brain Services Integration (Week 3)
- CEO Agent (strategic planning)
- Strategy Manager (regime detection)
- Risk Orchestra (multi-model coordination)

### Sprint 4: Production Hardening (Week 4)
- Real money paper trading (testnet)
- Live trading (small positions)
- Circuit breakers
- Disaster recovery

### Sprint 5: Advanced Features (Week 5+)
- Multi-asset portfolio optimization
- Options strategies
- Market making
- High-frequency signals

---

## Dependencies

### Python Packages (add to requirements_core.txt)
```
# RL training
torch==2.1.0
numpy==1.24.3
scipy==1.11.3

# Monitoring
tensorboard==2.14.1
```

### System Requirements
- Redis: Already installed ✅
- Python 3.10+: Already installed ✅
- GPU: Optional (CPU fine for PPO with small state space)

---

## Team Coordination

**Roles:**
- AI Engineer: Implement RL components (Feedback Bridge, PPO)
- DevOps: Deploy services, monitoring setup
- QA: Integration testing, performance validation

**Daily Standups:**
- Progress update
- Blockers
- Next 24-hour plan

**Communication:**
- Slack: `#quantum-trader-sprint2`
- GitHub: Use issues for tracking
- Documentation: Update this plan daily

---

## Conclusion

Sprint 2 completes the **AI learning loop**, enabling the system to:
- ✅ Learn from every trade
- ✅ Optimize position sizing dynamically
- ✅ Adapt to market changes automatically
- ✅ Close the feedback loop: Signal → Execute → Learn → Improve

**Next Steps:**
1. Deploy AI Engine integration (Day 1)
2. Implement RL Feedback Bridge (Day 2-3)
3. Train PPO Position Sizer (Day 4-5)
4. Activate CLM Drift Detection (Day 6-7)
5. Integration testing (Day 8)

**Expected Outcome:**
A self-learning trading system that improves over time, adapts to market conditions, and optimizes risk-adjusted returns.

🚀 **Let's build the future of autonomous trading!**
