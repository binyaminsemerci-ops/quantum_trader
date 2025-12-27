"""Test if TP trigger logic works correctly."""
import sys

# Simulate LONG position
side = "LONG"
current_price = 137.60  # Above TP0
tp_levels = [(137.53, 0.40), (138.59, 0.35), (140.18, 0.25)]
triggered_legs = set()

print(f"\n=== TP TRIGGER TEST ===")
print(f"Side: {side}")
print(f"Current Price: ${current_price}")
print(f"TP Levels: {tp_levels}")
print(f"Already Triggered: {triggered_legs}")
print("\nChecking each TP:")

triggerable = []
for i, (tp_price, size_pct) in enumerate(tp_levels):
    already_triggered = i in triggered_legs
    
    if side == "LONG":
        should_trigger = current_price >= tp_price
    else:
        should_trigger = current_price <= tp_price
    
    print(f"\n  TP{i}: price=${tp_price:.2f}, size={size_pct:.1%}")
    print(f"    Already triggered: {already_triggered}")
    print(f"    Should trigger: {should_trigger} (price {current_price} >= {tp_price})")
    
    if should_trigger and not already_triggered:
        triggerable.append((i, tp_price, size_pct))
        print(f"    ✅ WILL TRIGGER!")
    else:
        print(f"    ❌ No trigger")

print(f"\n📊 Result: {len(triggerable)} triggerable TPs")
if triggerable:
    print(f"First to trigger: TP{triggerable[0][0]} @ ${triggerable[0][1]}")
else:
    print("⚠️  NO TPs would trigger!")
