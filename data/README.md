# SharpsSignal Snowflake CI/CD Starter

This repo contains a minimal, production-style pipeline for Snowflake:
- **schemachange** for ordered DDL/RBAC migrations
- **dbt-snowflake** for transformations and tests
- **GitHub Actions** for CI (PR ephemeral schema) and CD (Stage → Prod with approval)
- **Zero-copy clones** / ephemeral schemas for fast previews

## Folder layout
```
migrations/            # schemachange change scripts (ordered)
dbt/                   # dbt project (models + tests)
.github/workflows/     # CI/CD pipelines
ops/                   # policies, helpers
```

## Quick start
1) Create Snowflake objects (names can be changed):
   - Databases: `DEV_DB`, `STAGE_DB`, `PROD_DB`
   - Warehouses: `WH_CI_SMALL`, `WH_TRANSFORM_MED`
   - Roles: `ROLE_CI` (least privilege for CI), plus your human dev role.
   - User: `CI_BOT` assigned to `ROLE_CI`.

2) Generate a Snowflake key pair for CI_BOT, store **private key** in GitHub Secrets (base64):
   - `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PRIVATE_KEY`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_WAREHOUSE`

3) In GitHub, add repo secrets for dbt targets (Stage/Prod) if needed:
   - `SNOWFLAKE_DB_STAGE`, `SNOWFLAKE_DB_PROD` (or keep the defaults in workflows).

4) Push a branch and open a PR. CI will:
   - create `pr_<number>` schema in `DEV_DB`
   - run **schemachange** (DDL) into that schema
   - run **dbt build** with tests

5) Merge to `main`. CD will deploy to `STAGE_DB`, then wait for **manual approval** to deploy to `PROD_DB`.

## Demo notes
- Change `dbt/models/marts/fact_bets.sql` in a PR (add a column or calc).
- Show CI summary, then merge and approve to deploy.
- Query in Snowflake: `select * from PROD_DB.ANALYTICS.FACT_BETS limit 5;`.
