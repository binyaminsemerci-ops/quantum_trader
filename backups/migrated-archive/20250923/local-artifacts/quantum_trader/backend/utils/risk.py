class RiskManager:
    def __init__(self, max_position: float = 0.1, max_loss_pct: float = 0.02):
        """
        :param max_position: Maks posisjon som andel av balansen (f.eks. 0.1 = 10%)
        :param max_loss_pct: Maks prosentvis tap pr. trade (f.eks. 0.02 = 2%)
        """
        self.max_position = max_position
        self.max_loss_pct = max_loss_pct

    def validate_order(
        self, balance: float, qty: float, price: float, stop_loss: float
    ) -> (bool, str):
        """
        Validerer en ordre basert på risikoregler.
        :param balance: Kontoens balanse
        :param qty: Antall enheter
        :param price: Kjøpspris
        :param stop_loss: Stop-loss pris
        :return: (True/False, begrunnelse)
        """
        position_size = qty * price

        # 1. Sjekk maks posisjon
        if position_size > balance * self.max_position:
            return False, "Position size exceeds max allowed"

        # 2. Sjekk gyldig stop-loss
        if stop_loss <= 0 or stop_loss >= price:
            return False, "Invalid stop-loss"

        # 3. Sjekk risiko per trade
        risk_amount = (price - stop_loss) * qty
        if risk_amount > balance * self.max_loss_pct:
            return False, "Risk exceeds maximum allowed per trade"

        return True, "Order is valid"
