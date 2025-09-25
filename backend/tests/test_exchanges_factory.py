from backend.utils.exchanges import get_exchange_client


def test_get_exchange_client_binance_mock():
    # call without name to exercise DEFAULT_EXCHANGE behavior
    client = get_exchange_client()
    assert client.spot_balance()['asset'] in ('USDC', 'USDT')
    assert isinstance(client.fetch_recent_trades('BTCUSDC', limit=3), list)


def test_get_exchange_client_coinbase_mock():
    client = get_exchange_client('coinbase')
    assert client.spot_balance()['asset'] == 'USDC'
    assert client.create_order('BTCUSDC', 'BUY', 0.001)['status'] == 'mock'


def test_get_exchange_client_kucoin_mock():
    client = get_exchange_client('kucoin')
    assert client.spot_balance()['asset'] == 'USDC'
    trades = client.fetch_recent_trades('ETHUSDC', limit=2)
    assert len(trades) == 2
