DEL A — SYSTEM TRUTH MAP (verified at 2026-02-17T21:46:04Z)

- Host/VPS
  - Status: ✅
  - Running: YES
  - Connected to: systemd, Redis
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: Ubuntu 24.04.3 LTS, kernel 6.8.0-90, uptime 28d

- systemd quantum-* (overall)
  - Status: ⚠️
  - Running: PARTIAL
  - Connected to: 101 loaded units, 137 unit files
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: failed units = quantum-harvest-brain, quantum-risk-proposal, quantum-rl-agent, quantum-verify-ensemble

- AI Engine (quantum-ai-engine)
  - Status: ✅
  - Running: YES
  - Connected to: trade.intent, ai.signal_generated, ai.decision.made
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; uvicorn process on port 8001

- AI Strategy Router (quantum-ai-strategy-router)
  - Status: ✅
  - Running: YES
  - Connected to: ai.decision.made
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; process present

- Apply Layer (quantum-apply-layer)
  - Status: ✅
  - Running: YES
  - Connected to: apply.plan -> apply.result
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; apply.plan/apply.result streams updating

- Intent Bridge (quantum-intent-bridge)
  - Status: ✅
  - Running: YES
  - Connected to: trade.intent -> apply.plan
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; trade.intent and apply.plan updating

- Execution Service (quantum-execution)
  - Status: ⚠️
  - Running: YES
  - Connected to: execution.result, trade.execution.res
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; execution.result last event 2026-02-09

- Risk Brain (quantum-risk-brain)
  - Status: ✅
  - Running: YES
  - Connected to: risk.events
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; process present

- Risk Proposal (quantum-risk-proposal)
  - Status: ❌
  - Running: NO
  - Connected to: UNKNOWN
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd failed

- Harvest Brain (quantum-harvest-brain)
  - Status: ❌
  - Running: NO
  - Connected to: harvest.intent, harvest.proposal
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd failed

- Harvest proposal pipeline (quantum-harvest-proposal, quantum-harvest-optimizer)
  - Status: ✅
  - Running: YES
  - Connected to: harvest.proposal
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; harvest.proposal updating

- Portfolio governance chain (portfolio-* services)
  - Status: ✅
  - Running: YES
  - Connected to: portfolio.state, portfolio.snapshot_updated, portfolio.exposure_updated
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: systemd active/running; streams updating

- Market ingest + features
  - Status: ✅
  - Running: YES
  - Connected to: exchange.raw -> exchange.normalized -> features -> ai-engine
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: streams updating; feature publisher and ensemble predictor running

- RL stack
  - Status: ⚠️
  - Running: PARTIAL
  - Connected to: rl_rewards, rl policy publisher, rl sizer
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: rl-monitor, rl-policy-publisher, rl-sizer active; rl-agent failed; rl-trainer auto-restart

- Frontend/UI
  - Status: ⚠️
  - Running: PARTIAL
  - Connected to: dashboard API only
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: quantum-dashboard-api active; quantum-frontend and quantum-quantumfond-frontend inactive/dead

- Redis
  - Status: ✅
  - Running: YES
  - Connected to: multiple producers/consumers
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: redis 7.0.15; streams active; 40+ quantum:stream keys

- Env/Config
  - Status: ⚠️
  - Running: UNKNOWN (mapping not verified)
  - Connected to: /etc/quantum/*.env
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: /etc/quantum contains many env files and backups; testnet.env sets TRADING_MODE=TESTNET, TRADING_ENABLED=true

- Repo & code structure on VPS
  - Status: ⚠️
  - Running: UNKNOWN
  - Connected to: /home/qt/quantum_trader (path referenced in processes)
  - Last verified: 2026-02-17T21:46:04Z
  - Evidence: processes reference /home/qt/quantum_trader; code/module usage not verified beyond running processes

DEL B — WHAT ACTUALLY WORKS (running + input + output)

- Exchange ingest: exchange.raw and exchange.normalized are active with recent events.
- Feature pipeline: features stream updates and ensemble_predictor consumes it.
- Trade intent ingress: trade.intent receives events; intent bridge consumes.
- Apply pipeline: apply.plan and apply.result update with active consumer groups.
- Portfolio state: portfolio.state, portfolio.snapshot_updated, portfolio.exposure_updated update.
- Market data: market.tick and market.klines update.
- Dashboard API: quantum-dashboard-api service is active.

DEL C — WHAT IS BROKEN / DEAD (verified)

- quantum-harvest-brain.service is failed.
- quantum-risk-proposal.service is failed.
- quantum-rl-agent.service is failed.
- quantum-verify-ensemble.service is failed.
- execution.result and trade.execution.res last events are 2026-02-09 (stale relative to other streams).
- apply.plan.manual, model.retrain, policy.updated, reconcile.close streams are empty (length 0).

DEL D — SYSTEMIC ROOT CAUSES (evidence-based)

- Failed critical services exist alongside active core pipeline (systemd shows failed units while core streams keep updating).
- Control-plane streams are empty (policy.updated, model.retrain, reconcile.close) while execution/decision streams are active, indicating missing control-plane activity.
- Configuration sprawl on /etc/quantum (many env files and backups) with no verified mapping to active units in this report.

DEL E — RISK CLASSIFICATION

- P0: Harvest Brain failed
  - If ignored: profit harvesting path is not running while other trading services remain active.

- P0: Risk Proposal failed
  - If ignored: risk proposal publishing is not running; downstream risk gates rely on stale or absent proposals.

- P1: Execution feedback stream stale
  - If ignored: execution result feedback is not updating since 2026-02-09, reducing observability of execution outcomes.

- P1: RL Agent failed
  - If ignored: RL agent shadow path is not running; RL trainer in auto-restart indicates unstable RL training loop.

- P1: Control-plane streams empty
  - If ignored: policy and retrain feedback loops are not producing events.

REDIS — STREAM TRUTH TABLE (from probe output)

| STREAM | PRODUSER (source) | CONSUMER GROUPS | AKTIV? | SIST EVENT | LENGTH |
|---|---|---|---|---|---|
| quantum:stream:account.balance | UNKNOWN | NONE | YES | 1771364740 | 116 |
| quantum:stream:ai.decision.made | ai-engine | router | YES | 2026-02-17T21:42:20.636131 | 10002 |
| quantum:stream:ai.exit.decision | UNKNOWN | autonomous-trader:exit-listeners | YES | 1771364727 | 46882 |
| quantum:stream:ai.signal_generated | ai-engine | NONE | YES | 2026-02-17T21:41:21.951447 | 10003 |
| quantum:stream:allocation.decision | UNKNOWN | NONE | YES | 1771364765 | 1012 |
| quantum:stream:apply.heat.observed | UNKNOWN | NONE | YES | 2026-02-17T21:46:05+00:00 | 10020 |
| quantum:stream:apply.plan | UNKNOWN | apply_layer_entry,governor,heat_gate,intent_executor,p33 | YES | 1771364759 | 10008 |
| quantum:stream:apply.plan.manual | UNKNOWN | intent_executor_manual | NO | UNKNOWN | 0 |
| quantum:stream:apply.result | UNKNOWN | exit_intelligence,harvest_brain:execution,metricpack_builder,p35_decision_intel,quantum:group:execution:trade.intent,trade_history_logger | YES | 1771364768 | 10002 |
| quantum:stream:bsc.events | UNKNOWN | NONE | YES | 2026-02-17T21:45:27.152077+00:00 | 1251 |
| quantum:stream:clm.intent | UNKNOWN | NONE | YES | 1771299214400-0 | 702 |
| quantum:stream:exchange.normalized | UNKNOWN | feature_publisher,quantum:group:ai-engine:exchange.normalized | YES | 1771308360 | 5423029 |
| quantum:stream:exchange.raw | UNKNOWN | quantum:group:ai-engine:exchange.raw | YES | 1771364760 | 1538643 |
| quantum:stream:execution.result | execution-service | quantum:group:bridge:execution.result | YES | 2026-02-09T23:20:25.849704Z | 2154 |
| quantum:stream:exitbrain.pnl | UNKNOWN | NONE | YES | 1770263286488-0 | 6 |
| quantum:stream:features | UNKNOWN | ensemble_predictor | YES | 1771308363511-0 | 10001 |
| quantum:stream:harvest.intent | UNKNOWN | intent_executor_harvest | YES | 1771364413 | 4879 |
| quantum:stream:harvest.proposal | UNKNOWN | p26_heat_gate,p26_portfolio_gate | YES | 2026-02-17T21:45:59+00:00 | 20369 |
| quantum:stream:market.klines | market-publisher | quantum:group:ai-engine:market.klines | YES | 2026-02-17T21:46:00.290728 | 10005 |
| quantum:stream:market.tick | market-publisher | feature_publisher,quantum:group:ai-engine:market.tick | YES | 2026-02-17T21:46:07.664899 | 10005 |
| quantum:stream:marketstate | UNKNOWN | NONE | YES | 1771364731 | 10028 |
| quantum:stream:model.retrain | UNKNOWN | retraining_workers | NO | UNKNOWN | 0 |
| quantum:stream:policy.audit | UNKNOWN | NONE | YES | 1770667391783-0 | 1003 |
| quantum:stream:policy.update | UNKNOWN | NONE | YES | 1770667391.7833147 | 104 |
| quantum:stream:policy.updated | UNKNOWN | quantum:group:ai-engine:policy.updated | NO | UNKNOWN | 0 |
| quantum:stream:portfolio.cluster_state | UNKNOWN | NONE | YES | 2026-02-17T21:46:07+00:00 | 1154 |
| quantum:stream:portfolio.exposure_updated | portfolio_intelligence | NONE | YES | 2026-02-17T21:46:03.766051 | 6815 |
| quantum:stream:portfolio.gate | UNKNOWN | NONE | YES | 2026-02-17T21:45:59+00:00 | 20369 |
| quantum:stream:portfolio.snapshot_updated | portfolio_intelligence | NONE | YES | 2026-02-17T21:46:03.765302 | 6816 |
| quantum:stream:portfolio.state | portfolio-state-publisher | NONE | YES | 1771364765 | 1037 |
| quantum:stream:position.snapshot | balance-tracker | NONE | YES | 1771364740 | 1017 |
| quantum:stream:reconcile.close | UNKNOWN | apply_recon | NO | UNKNOWN | 0 |
| quantum:stream:reconcile.events | UNKNOWN | NONE | YES | 1771364768621-1 | 10004 |
| quantum:stream:risk.events | UNKNOWN | NONE | YES | 1771361218.8681164 | 30 |
| quantum:stream:rl_rewards | UNKNOWN | NONE | YES | 2026-02-17T03:19:12.376657+00:00 | 110 |
| quantum:stream:signal.score | UNKNOWN | NONE | YES | 2026-02-17T06:06:03.532822Z | 10004 |
| quantum:stream:trade.closed | autonomous_trader | quantum:group:ai-engine:trade.closed | YES | 2026-02-17T21:26:35.884213 | 1002 |
| quantum:stream:trade.execution.res | execution-service | NONE | YES | 2026-02-09T23:20:25.849704Z | 2154 |
| quantum:stream:trade.intent | ai-engine | quantum:group:execution:trade.intent,quantum:group:intent_bridge | YES | 2026-02-17T21:42:43.991667 | 10000 |
| quantum:stream:utf | UNKNOWN | clm | YES | 1771364762523-0 | 200008 |

SLUTTORD (obligatorisk)

Hvis systemet stoppes i dag, er dette de TOP 5 grunnene.
- Harvest Brain er failed i systemd.
- Risk Proposal er failed i systemd.
- Execution feedback stream (execution.result) er stale siden 2026-02-09.
- RL Agent er failed; RL trainer er i auto-restart.
- Kontrollplan-streams er tomme (policy.updated, model.retrain, reconcile.close).
