-- change: base table for confirmed bets (DEV/PR schemas)
create table if not exists ${db_name}.${schema}.CONFIRMEDBETS like DEV_DB.RAW.CONFIRMEDBETS;

-- ensure base exists in DEV for local work
create table if not exists DEV_DB.RAW.CONFIRMEDBETS (
  bet_id string,
  sport string,
  league string,
  market string,
  team_home string,
  team_away string,
  pick_side string,
  odds number(10,2),
  stake number(10,2),
  kickoff_ts timestamp_tz,
  created_ts timestamp_tz default current_timestamp()
);
