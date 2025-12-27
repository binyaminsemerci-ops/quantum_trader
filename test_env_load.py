from dotenv import load_dotenv
import os

load_dotenv()

print(f"QT_MAX_NOTIONAL_PER_TRADE = {os.getenv('QT_MAX_NOTIONAL_PER_TRADE')}")
print(f"QT_MAX_DAILY_LOSS = {os.getenv('QT_MAX_DAILY_LOSS')}")
print(f"QT_MAX_GROSS_EXPOSURE = {os.getenv('QT_MAX_GROSS_EXPOSURE')}")
print(f"QT_MAX_POSITION_PER_SYMBOL = {os.getenv('QT_MAX_POSITION_PER_SYMBOL')}")
print(f"QT_DEFAULT_LEVERAGE = {os.getenv('QT_DEFAULT_LEVERAGE')}")
