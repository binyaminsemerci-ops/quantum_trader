import glob, os
dirs = ["/opt/quantum/ai_engine/models", "/opt/quantum/model_registry/approved"]
patterns = ["scaler_v20251115_*", "scaler_v20251116_*", "scaler_v20251117_*"]
removed = 0
for d in dirs:
    for p in patterns:
        for f in glob.glob(d + "/" + p):
            os.remove(f)
            removed += 1
print(f"Removed {removed} legacy scaler files")
remaining = len(glob.glob("/opt/quantum/ai_engine/models/*.pkl"))
print(f"Remaining pkl in models/: {remaining}")