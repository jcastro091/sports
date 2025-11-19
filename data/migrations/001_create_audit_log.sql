CREATE TABLE IF NOT EXISTS audit_log (
  id BIGSERIAL PRIMARY KEY,
  ts_utc TIMESTAMPTZ NOT NULL,
  user_id TEXT NOT NULL,
  model TEXT NOT NULL,
  input JSONB NOT NULL,
  output JSONB NOT NULL,
  decision TEXT NOT NULL,
  status TEXT NOT NULL,
  meta JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts_utc DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_decision ON audit_log(decision);
