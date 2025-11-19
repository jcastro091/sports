{{ config(materialized='table') }}
select distinct team
from (
  select team_home as team from {{ ref('confirmedbets__stg') }}
  union all
  select team_away as team from {{ ref('confirmedbets__stg') }}
)
where team is not null
