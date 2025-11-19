import requests

url = "http://localhost:8099/api/audit_log"
payload = {
    "user": "john",
    "model": "test",
    "input": {"a": 1},
    "output": {"b": 2},
    "decision": "approve",
    "status": "ok"
}

r = requests.post(url, json=payload)
print(r.status_code, r.text)
