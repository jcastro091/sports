{{ config(materialized='table') }}
with s as (
  select * from {{ ref('confirmedbets__stg') }}
)
select
  bet_id,
  sport, league, market, pick_side,
  odds, stake, kickoff_ts,
  1.0 / (1 + (iff(odds>=0, odds/100, 100/abs(odds)))) as implied_prob,
  {{ dbt_utils.generate_surrogate_key(['bet_id']) }} as bet_sk
  , current_timestamp() as demo_added_col
from s
