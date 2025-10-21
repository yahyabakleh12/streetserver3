-- camera_reports hypertable setup for TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS camera_reports (
  id BIGSERIAL PRIMARY KEY,
  ts timestamptz NOT NULL,
  camera_id text NOT NULL,
  event_type text NOT NULL,
  occupancy int,
  bbox jsonb,
  image_key text,
  payload jsonb NOT NULL
);

SELECT create_hypertable('camera_reports','ts', if_not_exists=>true);
CREATE INDEX IF NOT EXISTS camera_reports_camera_id_ts_idx ON camera_reports (camera_id, ts DESC);
