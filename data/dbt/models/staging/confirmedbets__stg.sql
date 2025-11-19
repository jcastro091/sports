{{ config(materialized='view') }}
select
  bet_id,
  upper(sport)   as sport,
  upper(league)  as league,
  market,
  team_home,
  team_away,
  pick_side,
  try_to_decimal(odds) as odds,
  try_to_decimal(stake) as stake,
  to_timestamp_tz(kickoff_ts) as kickoff_ts,
  created_ts
from {{ source('raw','confirmedbets') }}
