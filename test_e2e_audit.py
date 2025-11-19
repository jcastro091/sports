import requests, json, time

BASE = "http://localhost:8099"

def post_log():
    payload = {
        "user": "john",
        "model": "test",
        "input": {"a": 1},
        "output": {"b": 2},
        "decision": "approve",
        "status": "ok",
        "meta": {"note": "e2e"}
    }
    r = requests.post(f"{BASE}/api/audit_log", json=payload, timeout=10)
    print("POST status:", r.status_code, "body:", r.text)
    return r

def get_logs():
    r = requests.get(f"{BASE}/api/audit_log?user=john&limit=5", timeout=10)
    print("GET status:", r.status_code)
    try:
        data = r.json()
        print(json.dumps(data, indent=2)[:1000])
    except Exception as e:
        print("GET parse error:", e, r.text)

if __name__ == "__main__":
    post_log()
    time.sleep(0.5)
    get_logs()
