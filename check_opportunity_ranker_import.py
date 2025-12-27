"""Check if OpportunityRanker can be imported."""
import logging
logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("CHECKING OPPORTUNITY RANKER IMPORTS")
print("=" * 80)

# Step 1: Check factory import
try:
    from backend.integrations.opportunity_ranker_factory import (
        create_opportunity_ranker,
        get_default_symbols
    )
    print("✅ Factory import successful")
    print(f"   create_opportunity_ranker: {create_opportunity_ranker}")
    print(f"   get_default_symbols: {get_default_symbols}")
except ImportError as e:
    print(f"❌ Factory import failed: {e}")
    import traceback
    traceback.print_exc()
    OPPORTUNITY_RANKER_AVAILABLE = False
else:
    OPPORTUNITY_RANKER_AVAILABLE = True

# Step 2: Check routes import
if OPPORTUNITY_RANKER_AVAILABLE:
    try:
        from backend.routes import opportunity_routes
        print("✅ Routes import successful")
        print(f"   opportunity_routes: {opportunity_routes}")
        print(f"   router: {opportunity_routes.router}")
    except ImportError as e:
        print(f"❌ Routes import failed: {e}")
        import traceback
        traceback.print_exc()
        OPPORTUNITY_RANKER_AVAILABLE = False

print("=" * 80)
print(f"FINAL STATUS: OPPORTUNITY_RANKER_AVAILABLE = {OPPORTUNITY_RANKER_AVAILABLE}")
print("=" * 80)
