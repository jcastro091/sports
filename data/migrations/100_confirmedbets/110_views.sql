-- change: ANALYTICS view (points to provided schema)
create or replace view ${db_name}.ANALYTICS.VW_CONFIRMEDBETS as
select * from ${db_name}.${schema}.CONFIRMEDBETS;
