# DPO Training — GPU Cloud Launch Guide

## What this is

A **proxy fine-tuning** experiment.
The live inference model (`llama-3.3-70b-versatile` on Groq API) is not
directly trainable via LoRA.  Instead we fine-tune a local 8B proxy model
(`meta-llama/Meta-Llama-3.1-8B-Instruct`) on the same DPO dataset.
After training the adapter is served locally on the VPS in shadow mode
(read-only, no live trades) to compare action distributions.

---

## 1. Rent a GPU instance

| Provider      | Machine       | VRAM  | Cost/hr | Notes                    |
|---------------|---------------|-------|---------|--------------------------|
| RunPod        | RTX 4090      | 24 GB | ~$0.44  | Fast; use PyTorch template |
| Vast.ai       | RTX 3090      | 24 GB | ~$0.20  | Cheapest option           |
| Lambda Labs   | A10           | 24 GB | ~$0.60  | Stable; good for bfloat16 |
| Google Colab  | A100 40 GB    | 40 GB | $0/free | Free tier; may disconnect |

Minimum: **24 GB VRAM**.  Estimated runtime: **20–35 min** for 251 pairs × 2 epochs.

---

## 2. Start the instance and SSH in

```bash
# RunPod example — use the provided SSH command
ssh root@<pod-ip> -p <port>
```

---

## 3. Clone the repo and install deps

```bash
git clone https://github.com/<YOUR_ORG>/quantum_trader /workspace/qt
cd /workspace/qt/ops/offline
pip install -r requirements_train.txt
```

---

## 4. Copy the frozen dataset from VPS

```bash
# From your LOCAL machine (Windows: run in WSL)
scp -i ~/.ssh/hetzner_fresh \
    root@46.224.116.254:/home/qt/quantum_trader/logs/dpo/dpo_patch10a_v2_FROZEN.jsonl \
    /tmp/dpo_patch10a_v2_FROZEN.jsonl

# Then copy to the cloud instance
scp -P <port> /tmp/dpo_patch10a_v2_FROZEN.jsonl \
    root@<pod-ip>:/workspace/qt/ops/offline/dpo_patch10a_v2.jsonl
```

Or, if the GPU instance can reach the VPS directly:
```bash
# On the GPU instance
scp -i /path/to/hetzner_key \
    root@46.224.116.254:/home/qt/quantum_trader/logs/dpo/dpo_patch10a_v2_FROZEN.jsonl \
    /workspace/qt/ops/offline/dpo_patch10a_v2.jsonl
```

**Verify integrity before training:**
```bash
sha256sum /workspace/qt/ops/offline/dpo_patch10a_v2.jsonl
# Must print: 5fb175f398e9b87846559d0867351644c25e63ae5f46cdcd118ed311ea5ced80
```

---

## 5. Format the dataset

```bash
cd /workspace/qt/ops/offline
python format_dpo_dataset.py
# Output: dpo_formatted/dpo_train.jsonl (~200 rows)
#         dpo_formatted/dpo_val.jsonl   (~51 rows)
```

---

## 6. Authenticate with Hugging Face

```bash
huggingface-cli login
# Paste your HF token (needs read access to meta-llama/Meta-Llama-3.1-8B-Instruct)
# Accept the Llama 3.1 license at https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
```

---

## 7. Run training (two terminals)

**Terminal A — training:**
```bash
cd /workspace/qt/ops/offline
python train_dpo.py 2>&1 | tee train_run.log
```

**Terminal B — monitor (start immediately after A):**
```bash
cd /workspace/qt/ops/offline
python watch_training.py --state dpo_run_output/trainer_state.json
```

The monitor prints a line every 30 s.  It will flag `COLLAPSE` if val_loss
rises to 1.25× baseline and `DONE: GO` / `DONE: NO-GO` at the end.

---

## 8. Run shadow evaluation

```bash
cd /workspace/qt/ops/offline
python shadow_eval.py \
    --adapter dpo_run_output \
    --val     dpo_formatted/dpo_val.jsonl \
    --base    meta-llama/Meta-Llama-3.1-8B-Instruct
# Output: shadow_eval_results.json
```

---

## 9. Run go/no-go gate

```bash
python go_no_go.py \
    --trainer-state dpo_run_output/trainer_state.json \
    --shadow-eval   shadow_eval_results.json
# Exit 0 → GO    Exit 1 → NO-GO
```

---

## 10. Copy adapter back to VPS (only if GO)

```bash
# On GPU instance — pack the adapter
tar -czf dpo_run_output.tar.gz dpo_run_output/

# On LOCAL (WSL)
scp -P <port> root@<pod-ip>:/workspace/qt/ops/offline/dpo_run_output.tar.gz /tmp/

scp -i ~/.ssh/hetzner_fresh /tmp/dpo_run_output.tar.gz \
    root@46.224.116.254:/home/qt/quantum_trader/ops/offline/
```

**On VPS:**
```bash
cd /home/qt/quantum_trader/ops/offline
tar -xzf dpo_run_output.tar.gz
```

---

## 11. Enable shadow mode on VPS (testnet only)

Edit the systemd service override:
```bash
systemctl edit qt-exit-agent --force
```

Add:
```ini
[Service]
Environment="EXIT_AGENT_QWEN3_SHADOW=true"
Environment="EXIT_AGENT_DPO_ADAPTER=/home/qt/quantum_trader/ops/offline/dpo_run_output"
Environment="BINANCE_TESTNET=true"
```

Then:
```bash
systemctl daemon-reload
systemctl restart qt-exit-agent
```

The shadow mode activates the local 8B adapter alongside the Groq model.
It logs `[SHADOW_DPO]` events to Redis but does **not** override live decisions.

---

## Key file paths (summary)

| File                                    | Purpose                          |
|-----------------------------------------|----------------------------------|
| `ops/offline/format_dpo_dataset.py`     | Convert raw pairs → HF DPO JSONL |
| `ops/offline/train_dpo.py`              | TRL DPOTrainer script            |
| `ops/offline/watch_training.py`         | Live collapse monitor            |
| `ops/offline/shadow_eval.py`            | Post-training accuracy benchmark |
| `ops/offline/go_no_go.py`              | Pass/fail gate (exits 0 or 1)    |
| `ops/offline/requirements_train.txt`    | GPU cloud pip deps               |
| `ops/offline/dpo_run_output/`           | Adapter + tokenizer (after train)|
| `logs/dpo/dpo_patch10a_v2_FROZEN.jsonl` | Frozen baseline dataset          |

---

## Go / No-Go criteria (quick reference)

### Must all PASS to deploy shadow

| Criterion                          | Threshold                                  |
|------------------------------------|--------------------------------------------|
| Train loss reduction               | final < initial − 0.05                     |
| Val loss stability                 | final_val ≤ 1.25 × best_val               |
| Epoch 1 loss decreasing            | second-half mean < first-half mean         |
| Tuned accuracy on val set          | ≥ 40 %                                     |
| Accuracy lift vs base model        | ≥ +5 percentage points                     |
| JSON parse-error rate              | ≤ 5 %                                      |
| Max single-action share            | ≤ 80 % (no degenerate collapse)            |

### Immediate NO-GO triggers

| Trigger                            | Meaning                                    |
|------------------------------------|--------------------------------------------|
| val_loss at end > 1.25 × best_val  | Overfitting / catastrophic forgetting      |
| Tuned accuracy < base accuracy     | Training made the model worse              |
| TIME_STOP_EXIT share > 20 %        | Hallucinating time-exits on random inputs  |
| Parse errors > 5 %                 | Model no longer produces valid JSON        |

---

## Dataset fingerprint

```
File:    dpo_patch10a_v2_FROZEN.jsonl
SHA256:  5fb175f398e9b87846559d0867351644c25e63ae5f46cdcd118ed311ea5ced80
Pairs:   251
Split:   80/20 (seed=42)  →  train ≈200, val ≈51
```
