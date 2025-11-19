-- change: RBAC baseline (adjust to org policy)
create role if not exists ROLE_CI;
grant usage on database DEV_DB to role ROLE_CI;
grant usage on all schemas in database DEV_DB to role ROLE_CI;

grant usage on database STAGE_DB to role ROLE_CI;
grant usage on all schemas in database STAGE_DB to role ROLE_CI;

grant usage on database PROD_DB to role ROLE_CI;
grant usage on schema PROD_DB.ANALYTICS to role ROLE_CI;

grant create table, create view on schema DEV_DB.RAW to role ROLE_CI;
grant usage on warehouse WH_CI_SMALL to role ROLE_CI;
grant usage on warehouse WH_TRANSFORM_MED to role ROLE_CI;
