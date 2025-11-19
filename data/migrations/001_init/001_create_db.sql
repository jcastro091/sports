-- change: create databases and schemas (idempotent)
create database if not exists DEV_DB;
create database if not exists STAGE_DB;
create database if not exists PROD_DB;

create schema if not exists DEV_DB.RAW;
create schema if not exists DEV_DB.ANALYTICS;

create schema if not exists STAGE_DB.RAW;
create schema if not exists STAGE_DB.ANALYTICS;

create schema if not exists PROD_DB.RAW;
create schema if not exists PROD_DB.ANALYTICS;
