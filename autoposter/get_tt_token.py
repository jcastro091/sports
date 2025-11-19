#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import base64, hashlib, os, secrets, sys, json, urllib.parse, webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
import requests

CLIENT_KEY    = os.getenv("TT_CLIENT_KEY",    "").strip()
CLIENT_SECRET = os.getenv("TT_CLIENT_SECRET", "").strip()
REDIRECT_URI  = os.getenv("TT_REDIRECT_URI",  "http://127.0.0.1:8383/callback").strip()
SCOPES        = os.getenv("TT_SCOPES", "user.info.basic,video.upload,video.publish")

AUTH_URL  = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"  # x-www-form-urlencoded

STATE = "sharpssignal"

def b64url_no_pad(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")

def gen_pkce():
    # 43–128 chars from the permitted set
    code_verifier = secrets.token_urlsafe(64).rstrip("=")
    digest = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = hashlib.sha256(code_verifier.encode()).hexdigest()
    return code_verifier, code_challenge

code_verifier, code_challenge = gen_pkce()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith("/callback"):
            self.send_response(404); self.end_headers(); return
        q = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(q)
        code  = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]

        if not code or state != STATE:
            self.send_response(400); self.end_headers()
            self.wfile.write(b"Bad request: missing code or bad state")
            return

        # Exchange authorization code -> tokens (MUST include code_verifier)
        data = {
            "client_key":     CLIENT_KEY,
            "client_secret":  CLIENT_SECRET,   # optional for public apps; ok for confidential
            "grant_type":     "authorization_code",
            "code":           code,
            "redirect_uri":   REDIRECT_URI,
            "code_verifier":  code_verifier,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        print("Status: 200")
        print("Request Content-Type:", headers["Content-Type"])
        print("Request body preview:", "&".join([f"{k}={str(v)[:64]}" for k,v in data.items()]))

        r = requests.post(TOKEN_URL, headers=headers, data=data, timeout=30)
        try:
            payload = r.json()
        except Exception:
            payload = {"raw": r.text}

        print("\n🔑 TOKEN RESPONSE:\n", json.dumps(payload, indent=2))

        # Save tokens if present
        out = {
            "access_token":  payload.get("access_token", ""),
            "refresh_token": payload.get("refresh_token", ""),
            "expires_in":    payload.get("expires_in", 0),
            "scopes":        payload.get("scope", ""),
            "token_type":    payload.get("token_type", ""),
        }
        with open("tiktok_tokens.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

        self.send_response(200); self.end_headers()
        self.wfile.write(b"OK - You can close this tab.")
        # Stop server after one callback
        def shutdown(server): server.shutdown()
        import threading; threading.Thread(target=shutdown, args=(httpd,)).start()

if __name__ == "__main__":
    if not CLIENT_KEY or not CLIENT_SECRET:
        print("Please set TT_CLIENT_KEY and TT_CLIENT_SECRET in your environment.")
        sys.exit(1)

    print("PKCE code_verifier:", code_verifier)
    params = {
        "client_key":            CLIENT_KEY,
        "response_type":         "code",
        "scope":                 SCOPES,  # comma-separated is fine
        "redirect_uri":          REDIRECT_URI,
        "state":                 STATE,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    }
    url = AUTH_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    print("🌐 Opening OAuth URL:\n", url)

    # start local receiver
    server_address = ("127.0.0.1", int(urllib.parse.urlparse(REDIRECT_URI).port or 8383))
    httpd = HTTPServer(server_address, Handler)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print(f"📡 Listening on {REDIRECT_URI} ...")
    httpd.serve_forever()
