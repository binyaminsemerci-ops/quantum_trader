# Code Diff - MIN_NOTIONAL Fix

## File: exit_order_gateway.py

### Location: Line 275-299 (After quantization, before submission)

```diff
         # Quantize quantity (for all orders)
         if 'quantity' in order_params:
             original_qty = float(order_params['quantity'])
             quantized_qty = quantize_quantity(symbol, original_qty, client)
             if quantized_qty == 0.0:
                 logger.error(
                     f"[EXIT_GATEWAY] {symbol}: Quantity {original_qty} too small after quantization. "
                     f"Order NOT submitted."
                 )
                 return None
             order_params['quantity'] = str(quantized_qty)
             logger.debug(
                 f"[EXIT_GATEWAY] {symbol} quantity: {original_qty:.8f} -> {quantized_qty:.8f}"
             )

+        # 🔥 FIX -4164 MIN_NOTIONAL: Set reduceOnly=True for ALL exit orders
+        # This allows exit orders with notional < $5 to succeed
+        order_params['reduceOnly'] = True
+        
+        # Calculate and log notional for monitoring
+        if 'quantity' in order_params and 'price' in order_params:
+            try:
+                qty = float(order_params['quantity'])
+                price = float(order_params['price'])
+                notional = qty * price
+                
+                if notional < 5.0:
+                    logger.warning(
+                        f"[EXIT_GATEWAY] ⚠️  {symbol} notional ${notional:.2f} < $5 MIN_NOTIONAL. "
+                        f"reduceOnly=True allows this exit order to proceed."
+                    )
+                else:
+                    logger.debug(
+                        f"[EXIT_GATEWAY] {symbol} notional ${notional:.2f} (reduceOnly=True set)"
+                    )
+            except (ValueError, TypeError) as e:
+                logger.debug(f"[EXIT_GATEWAY] Could not calculate notional: {e}")

         # Log order submission attempt (include positionSide if present for hedge mode visibility)
         position_side_str = f", positionSide={order_params.get('positionSide', 'N/A')}" if 'positionSide' in order_params else ""
         logger.info(
             f"[EXIT_GATEWAY] 📤 Submitting {order_kind} order: "
             f"module={module_name}, symbol={symbol}, type={order_type}{position_side_str}, "
-            f"params_keys={list(order_params.keys())}"
+            f"params_keys={list(order_params.keys())}, reduceOnly={order_params.get('reduceOnly', False)}"
         )

         # Forward to actual exchange client
         result = client.futures_create_order(**order_params)
```

## Summary of Changes

**Lines Added:** 24  
**Lines Modified:** 1  
**Total Diff:** 25 lines

### Added Features

1. **Line 276:** `order_params['reduceOnly'] = True` - Core fix
2. **Lines 279-292:** Notional calculation and logging
   - Calculates `notional = quantity × price`
   - Warns if notional < $5 (special attention)
   - Debug logs for notional ≥ $5 (normal case)
   - Error handling for calculation failures
3. **Line 299:** Enhanced log to show `reduceOnly` status

### Modified Features

- **Line 299:** Added `reduceOnly={order_params.get('reduceOnly', False)}` to submission log

### No Changes To

- ❌ Order sizing logic
- ❌ Quantization logic
- ❌ Module ownership checks
- ❌ Error handling flow
- ❌ Order submission mechanism

**Impact:** Minimal, surgical fix - only adds `reduceOnly=True` parameter to all exit orders.
