s="/home/qt/quantum_trader/services/exit_monitor_service.py"
c=open(s).read()
g="""
_exit_processed=set()
_exit_cooldown={}
def check_exit_dedup(sym,oid):
 k=f"{sym}_{oid}"
 if k in _exit_processed:logger.info(f"EXIT_DEDUP {k}");return True
 _exit_processed.add(k);return False
def check_exit_cooldown(sym,side):
 from datetime import datetime,timedelta;k=f"{sym}_{side}";n=datetime.utcnow()
 if k in _exit_cooldown and n-_exit_cooldown[k]<timedelta(seconds=30):logger.info(f"EXIT_COOLDOWN {sym}");return True
 _exit_cooldown[k]=n;return False

"""
c=c.replace("async def send_close_order(position: TrackedPosition, reason: str):",g+"async def send_close_order(position: TrackedPosition, reason: str):")
c=c.replace('"""Send close order to execution service"""\n    try:','"""Send close order to execution service"""\n    if check_exit_dedup(position.symbol,position.order_id):return\n    if check_exit_cooldown(position.symbol,position.side):return\n    try:')
c=c.replace('f"🎯 EXIT TRIGGERED:','f"📤 EXIT_PUBLISH:')
open(s,"w").write(c)
print("Patched")
