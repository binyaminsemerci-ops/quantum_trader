from backend.database import get_db

def calculate_pnl():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT side, qty, price FROM trades")
    trades = cursor.fetchall()
    pnl = 0
    for side, qty, price in trades:
        if side.upper() == "BUY":
            pnl -= qty * price
        elif side.upper() == "SELL":
            pnl += qty * price
    return round(pnl, 2)

def calculate_pnl_per_symbol():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT symbol, side, qty, price FROM trades")
    trades = cursor.fetchall()
    pnl_by_symbol = {}
    for symbol, side, qty, price in trades:
        if symbol not in pnl_by_symbol:
            pnl_by_symbol[symbol] = 0
        if side.upper() == "BUY":
            pnl_by_symbol[symbol] -= qty * price
        elif side.upper() == "SELL":
            pnl_by_symbol[symbol] += qty * price
    return {sym: round(val, 2) for sym, val in pnl_by_symbol.items()}
