"""Quick post-training verification — run once after both v2 training scripts complete."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 55)
print("  POST-TRAINING VERIFICATION")
print("=" * 55)

# ── 1. Metadata checks ────────────────────────────────────
import json
with open("ai_engine/models/lgbm_metadata.json") as f:
    lm = json.load(f)
print(f"\n[LGBM]  version={lm['version']}  num_features={lm['num_features']}"
      f"  val_acc={lm['val_accuracy']}%  best_iter={lm['best_iteration']}")
assert lm["num_features"] == 49, "LGBM num_features != 49!"
assert lm["version"] == "v2",    "LGBM version != v2!"

with open("ai_engine/models/nhits_metadata.json") as f:
    nm = json.load(f)
print(f"[NHiTS] version={nm['version']}  num_features={nm['num_features']}"
      f"  seq_len={nm['sequence_length']}  val_acc={nm['val_accuracy']}%")
assert nm["num_features"] == 49,  "NHiTS num_features != 49!"
assert nm["sequence_length"] == 120, "NHiTS seq_len != 120!"
assert nm["version"] == "v2",     "NHiTS version != v2!"

# ── 2. NHiTS checkpoint ───────────────────────────────────
import torch
ckpt = torch.load("ai_engine/models/nhits_model.pth", map_location="cpu", weights_only=False)
assert ckpt["num_features"] == 49, "Checkpoint num_features != 49!"
print(f"[NHiTS] checkpoint num_features={ckpt['num_features']}  (agent line-96 will read this correctly)")

# ── 3. LGBM agent — no fallback ───────────────────────────
from ai_engine.agents.lgbm_agent import LightGBMAgent
agent = LightGBMAgent(
    model_path  ="ai_engine/models/lgbm_model.pkl",
    scaler_path ="ai_engine/models/lgbm_scaler.pkl",
)
assert agent.model  is not None, "LGBM model not loaded!"
assert agent.scaler is not None, "LGBM scaler not loaded!"
print(f"[LGBM]  model loaded OK  ({type(agent.model).__name__})")

features_49 = {f: 0.5 for f in [
    "returns","log_returns","price_range","body_size","upper_wick","lower_wick",
    "is_doji","is_hammer","is_engulfing","gap_up","gap_down",
    "rsi","macd","macd_signal","macd_hist","stoch_k","stoch_d","roc",
    "ema_9","ema_9_dist","ema_21","ema_21_dist","ema_50","ema_50_dist","ema_200","ema_200_dist",
    "sma_20","sma_50","adx","plus_di","minus_di",
    "bb_middle","bb_upper","bb_lower","bb_width","bb_position",
    "atr","atr_pct","volatility",
    "volume_sma","volume_ratio","obv","obv_ema","vpt",
    "momentum_5","momentum_10","momentum_20","acceleration","relative_spread",
]}
result = agent.predict("BTCUSDT", features_49)
is_fallback = "fallback" in result.get("model", "")
print(f"[LGBM]  predict → action={result['action']}  conf={result['confidence']:.2f}"
      f"  model={result['model']}  fallback={is_fallback}")
assert not is_fallback, "LGBM is still hitting fallback!"

# ── 4. NHiTS architecture check ──────────────────────────
from ai_engine.nhits_simple import SimpleNHiTS
nhits = SimpleNHiTS(input_size=120, num_features=49)
dummy = torch.zeros(1, 120, 49)
logits, _ = nhits(dummy)
print(f"[NHiTS] forward pass OK  input=[1,120,49]  output={tuple(logits.shape)}")
assert logits.shape == (1, 3), f"Unexpected output shape: {logits.shape}"

print("\n" + "=" * 55)
print("  ALL CHECKS PASSED")
print("  LGBM:  real ML model, 49 features, no fallback")
print("  NHiTS: real sequence model, 120x49, checkpoint OK")
print("=" * 55)
