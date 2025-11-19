# execution/adapters/kalshi_adapter.py

import time, base64, requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

@dataclass
class Order:
    row_id: str
    ticker: str
    side: str           # "YES" | "NO"
    price: float        # dollars
    size: float         # dollars (we convert to count)
    price_floor: float
    tif: str = "GTC"
    notes: str = ""

class KalshiAdapter:
    def __init__(self, base_url: str, key_id: str, private_key_file: str):
        self.base_url = base_url.rstrip("/")
        self.key_id = key_id
        self._priv = self._load_private_key(private_key_file)
        self._sess = requests.Session()
        self._sess.headers["Content-Type"] = "application/json"

    # ---------- signing ----------
    def _load_private_key(self, pem_path: str):
        p = Path(pem_path)
        data = p.read_bytes()
        return serialization.load_pem_private_key(data, password=None)

    def _signed_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Kalshi signature: sign timestamp + METHOD + path_without_query
        Using RSA-PSS(SHA256).
        """
        ts_ms = str(int(time.time() * 1000))
        path_wo_q = path.split("?", 1)[0]
        msg = f"{ts_ms}{method.upper()}{path_wo_q}".encode("utf-8")
        sig = self._priv.sign(
            msg,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        sig_b64 = base64.b64encode(sig).decode("ascii")
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
        }

    # ---------- HTTP ----------
    def _request(self, method: str, path: str,
                 params: Optional[Dict[str, Any]] = None,
                 json: Optional[Dict[str, Any]] = None) -> requests.Response:
        if not path.startswith("/"):
            path = "/" + path
        url = self.base_url + path
        # IMPORTANT: sign the path as seen by the server (without query string)
        headers = self._signed_headers(method, path)
        r = self._sess.request(method.upper(), url, params=params, json=json,
                               headers=headers, timeout=15)
        return r

    # ---------- high-level helpers ----------
    def place(self, order: Order):
        """
        Place a limit order. We send count (contracts) and per-side price in cents.
        size (dollars) / price (dollars) ≈ count
        """
        count = max(1, int(order.size // max(order.price, 0.01)))
        px_cents = int(round(order.price * 100))

        payload = {
            "ticker": order.ticker,
            "type": "limit",
            "tif": order.tif,
            "count": count,
            "yes_price": px_cents if order.side == "YES" else 0,
            "no_price":  px_cents if order.side == "NO"  else 0,
        }

        r = self._request("POST", "/trade-api/v2/portfolio/orders", json=payload)

        r.raise_for_status()
        # return a light object with what watcher expects
        data = r.json()
        class Resp: pass
        resp = Resp()
        resp.id = data.get("id") or data.get("order_id")
        resp.order_id = resp.id
        resp.status = data.get("status","SUBMITTED")
        resp.filled = data.get("filled") or data.get("count") or 0
        return resp
