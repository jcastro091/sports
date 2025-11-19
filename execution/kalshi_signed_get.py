# kalshi_signed_get.py (fixed)
import time, base64, requests
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

BASE = "https://api.elections.kalshi.com"
KEY_ID = "b1947b44-d40e-4afd-8e85-fdca0426e40a"  # <-- your Key ID from the screenshot
PEM_PATH = r"C:\Users\Jcast\OneDrive\Documents\sports-repo\keys\kalshi-demo-keyPROD.pem"  # <-- point to the real file

def _load_private_key(pem_path: str):
    pem = Path(pem_path)
    if not pem.exists():
        raise FileNotFoundError(f"Private key not found: {pem}")
    data = pem.read_bytes()
    return serialization.load_pem_private_key(data, password=None)

def _kalshi_headers(method: str, path: str, key_id: str, privkey) -> dict:
    ts_ms = str(int(time.time() * 1000))
    # sign path WITHOUT query per docs
    path_wo_q = path.split("?", 1)[0]
    msg = f"{ts_ms}{method.upper()}{path_wo_q}".encode("utf-8")
    # RSA-PSS + SHA256 per docs
    sig = privkey.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(sig).decode("ascii")
    print("SIGNING STRING:", (ts_ms + method.upper() + path_wo_q))
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": sig_b64,
    }

def signed_get(path: str):
    priv = _load_private_key(PEM_PATH)
    hdrs = _kalshi_headers("GET", path, KEY_ID, priv)
    r = requests.get(BASE + path, headers=hdrs, timeout=15)
    print("STATUS", r.status_code)
    print(r.text[:400])
    r.raise_for_status()
    return r.json()

if __name__ == "__main__":
    # 1) Prove auth
    signed_get("/trade-api/v2/portfolio/balance")
    # 2) Event
    signed_get("/trade-api/v2/events/KXELONMARS-99")
    # 3) Markets for that event
    signed_get("/trade-api/v2/markets?event_ticker=KXELONMARS-99")
