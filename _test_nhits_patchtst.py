import sys, os
sys.path.insert(0, '/home/qt/quantum_trader')
os.chdir('/home/qt/quantum_trader')

print("=== TESTING AGENT LOADING ===\n")

for agent_class, label in [
    ("NHiTSAgent", "N-HiTS"),
    ("PatchTSTAgent", "PatchTST"),
]:
    print(f"--- {label} ---")
    try:
        mod = __import__('ai_engine.agents.unified_agents', fromlist=[agent_class])
        cls = getattr(mod, agent_class)
        agent = cls()
        print(f"model type: {type(agent.model).__name__}")
        print(f"scaler type: {type(agent.scaler).__name__}")
        print(f"pytorch_model: {type(agent.pytorch_model).__name__ if agent.pytorch_model else 'None'}")
        print(f"features count: {len(agent.features)}")
        print(f"SUCCESS\n")
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()
        print()

print("=== DONE ===")
