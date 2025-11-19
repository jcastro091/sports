# Promotion & CI policy (short)
- All changes via PR with green CI (schemachange + dbt tests)
- Stage deploy on merge to main
- Manual approval required for Prod
- RBAC-as-code lives in migrations/001_init/010_rbac_roles_users.sql
