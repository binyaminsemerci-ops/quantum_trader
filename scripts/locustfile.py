from locust import HttpUser, TaskSet, task, between
import random

class UserBehavior(TaskSet):
    @task(2)
    def get_signals(self):
        self.client.get("/api/ai/signals/latest?limit=10")

    @task(2)
    def get_prices(self):
        symbol = random.choice(["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        self.client.get(f"/api/prices/latest?symbol={symbol}")

    @task(1)
    def get_stats(self):
        self.client.get("/api/stats/overview")

    @task(1)
    def get_trade_logs(self):
        self.client.get("/api/trade_logs?limit=10")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)
