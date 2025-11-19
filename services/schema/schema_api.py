from fastapi import FastAPI

app = FastAPI(title="Schema API", version="0.1.0")

@app.get("/api/schema")
def get_schema():
    return {
        "title": "Audit Log Filter",
        "fields": [
            {"name": "user", "label": "User", "type": "text"},
            {"name": "decision", "label": "Decision", "type": "text"},
            {"name": "since", "label": "Since (ISO 8601)", "type": "text"}
        ]
    }
