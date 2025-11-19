# scripts/test_request.py
import requests, json
url = "http://127.0.0.1:8099/context/text"
payload = {
    "event_id": "123456",
    "sport": "baseball_mlb",
    "league": "MLB",
    "teams": {"home": "BOS", "away": "NYY"},
    "start_time_utc": "2025-09-21T23:05:00Z",
    "pro": False,
}
resp = requests.post(url, json=payload)
print("STATUS:", resp.status_code)
print("CT:", resp.headers.get("content-type"))
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print("RAW:", resp.text[:1000])
