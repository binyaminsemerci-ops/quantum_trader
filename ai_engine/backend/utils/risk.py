import pandas as pd


class RiskManager:
    def __init__(self):
        self.stop_loss = None
        self.take_profit = None

    def set_stop_take(
        self, entry_price: float, stop_loss_pct=0.02, take_profit_pct=0.04, side: str = "LONG"
    ):
        """Calculate stop-loss and take-profit levels.
        
        Args:
            entry_price: Entry price for the position
            stop_loss_pct: Stop-loss percentage (e.g., 0.02 = 2%)
            take_profit_pct: Take-profit percentage (e.g., 0.04 = 4%)
            side: Position side - "LONG" or "SHORT"
        
        For LONG: SL below entry, TP above entry
        For SHORT: SL above entry, TP below entry
        """
        if side.upper() == "SHORT":
            # SHORT: Stop-loss ABOVE entry (price goes up = loss)
            self.stop_loss = entry_price * (1 + stop_loss_pct)
            # Take-profit BELOW entry (price goes down = profit)
            self.take_profit = entry_price * (1 - take_profit_pct)
        else:  # LONG
            # LONG: Stop-loss BELOW entry (price goes down = loss)
            self.stop_loss = entry_price * (1 - stop_loss_pct)
            # Take-profit ABOVE entry (price goes up = profit)
            self.take_profit = entry_price * (1 + take_profit_pct)

    def check_exit(self, current_price: float, position: int) -> bool:
        if position == 1:  # long
            return current_price <= self.stop_loss or current_price >= self.take_profit
        elif position == -1:  # short
            return current_price >= self.take_profit or current_price <= self.stop_loss
        return False

    def failsafe_filter(self, df: pd.DataFrame) -> bool:
        return not df.empty and "Close" in df.columns
