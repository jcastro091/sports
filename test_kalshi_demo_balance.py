# test_kalshi_demo_balance.py
import kalshi_python
from kalshi_python import Configuration, KalshiClient

configuration = Configuration(host="https://demo-api.kalshi.co/trade-api/v2")

with open(r"C:\Users\Jcast\OneDrive\Documents\sports-repo\keys\kalshi-demo-key.pem", "r") as f:
    configuration.private_key_pem = f.read()

configuration.api_key_id = "6983d0be-674b-455d-b8fd-91c42a657a47"  # your DEMO key id

client = KalshiClient(configuration)
bal = client.get_balance()
print(bal)
