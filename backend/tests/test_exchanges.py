import sys
import types
from backend.utils.exchanges import get_exchange_client


def test_binance_adapter_mocked():
    client = get_exchange_client('binance')
    bal = client.spot_balance()
    assert isinstance(bal, dict)
    assert 'asset' in bal and 'free' in bal


def test_coinbase_adapter_mocked():
    client = get_exchange_client('coinbase')
    bal = client.spot_balance()
    assert isinstance(bal, dict)
    assert 'asset' in bal and 'free' in bal
    trades = client.fetch_recent_trades('ETHUSDC', limit=3)
    assert isinstance(trades, list) and len(trades) == 3


def test_kucoin_adapter_mocked():
    client = get_exchange_client('kucoin')
    bal = client.spot_balance()
    assert isinstance(bal, dict)
    assert 'asset' in bal and 'free' in bal


def test_ccxt_integration_monkeypatch(monkeypatch):
    # Create a fake minimal ccxt module with coinbasepro and kucoin attributes
    fake_ccxt = types.SimpleNamespace()

    class FakeExchange:
        def __init__(self, conf):
            self.conf = conf

        def fetch_balance(self, params=None):
            return {'total': {'USDC': 123.45}, 'free': {'USDC': 123.45}}

        def fetch_trades(self, symbol, limit=5):
            # return trades in ccxt style
            return [{'timestamp': 1690000000000, 'amount': 0.01, 'price': 100.0, 'side': 'buy'} for _ in range(limit)]

        def create_order(self, symbol, type, side, amount):
            return {'id': 'order123', 'status': 'ok', 'symbol': symbol, 'side': side, 'amount': amount}

    fake_ccxt.coinbasepro = lambda conf=None: FakeExchange(conf)
    fake_ccxt.kucoin = lambda conf=None: FakeExchange(conf)

    monkeypatch.setitem(sys.modules, 'ccxt', fake_ccxt)

    # Now instantiate adapters with fake creds and assert they use the fake exchange
    client = get_exchange_client('coinbase', api_key='k', api_secret='s')
    bal = client.spot_balance()
    assert bal.get('asset') == 'USDC' and bal.get('free') == 123.45
    trades = client.fetch_recent_trades('ETHUSDC', limit=2)
    assert len(trades) == 2
    order = client.create_order('ETHUSDC', 'BUY', 0.1)
    assert order.get('status') == 'ok' or 'id' in order

    client2 = get_exchange_client('kucoin', api_key='k', api_secret='s')
    bal2 = client2.spot_balance()
    assert bal2.get('free') == 123.45
