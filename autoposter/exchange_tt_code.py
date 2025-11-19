#!/usr/bin/env python3
import os, json, requests, urllib.parse, hashlib, base64
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

CLIENT_KEY    = os.getenv("TT_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TT_CLIENT_SECRET")
REDIRECT_URI  = "http://127.0.0.1:8383/callback"
TOKEN_URL     = "https://open.tiktokapis.com/v2/oauth/token/"  # trailing slash
TIMEOUT       = 60

def s256(verifier: str) -> str:
    dig = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(dig).decode("ascii").rstrip("=")

print("Paste the FULL ?code= value from the callback URL:")
raw_code = input("> ").strip()
code = urllib.parse.unquote(raw_code)  # <-- decode once

print("Paste the PKCE code_verifier printed by get_tt_token.py:")
verifier = input("> ").strip()

print("Derived code_challenge from verifier:", s256(verifier))

payload = {
    "client_key": CLIENT_KEY,
    "client_secret": CLIENT_SECRET,
    "grant_type": "authorization_code",
    "code": code,                         # decoded code
    "redirect_uri": REDIRECT_URI,
    "code_verifier": verifier,
}

s = requests.Session(); s.trust_env = False
def post_once(url):
    r = s.post(url, data=payload, timeout=TIMEOUT, allow_redirects=False,
               headers={"Accept":"application/json",
                        "Content-Type":"application/x-www-form-urlencoded"})
    print("Status:", r.status_code)
    print("Resp Content-Type:", r.headers.get("Content-Type"))
    print("Resp Location:", r.headers.get("Location"))
    print("Raw body (first 400):", r.text[:400])
    return r

r = post_once(TOKEN_URL)
if r.is_redirect or r.status_code in (301,302,303,307,308):
    r = post_once(r.headers.get("Location") or TOKEN_URL)

try:
    data = r.json()
except Exception:
    data = {"status_code": r.status_code, "text": r.text[:400]}

print(json.dumps(data, indent=2))
with open("tiktok_tokens.json","w",encoding="utf-8") as f:
    json.dump(data,f,indent=2)
print("Saved tokens to tiktok_tokens.json")
