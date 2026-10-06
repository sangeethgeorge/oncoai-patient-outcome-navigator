# tests/test_db_utils.py
# Integration tests against the local MIMIC-III Postgres. Run with: pytest -m db

import pytest
import duckdb

from oncoai_prototype.utils.db_utils import connect_to_postgres, query_postgres_duckdb, run_postgres_sql, redact

pytestmark = pytest.mark.db


pg = run_postgres_sql


def test_redact_masks_passwords():
    assert redact("dbname=x user=u password=s3cret host=h") == "dbname=x user=u password=*** host=h"
    assert redact("postgresql://u:s3cret@h/db") == "postgresql://u:***@h/db"


def test_db_connection(db_conn_str):
    con = connect_to_postgres(db_conn_str)
    try:
        assert con.sql("SELECT 1").fetchone()[0] == 1
    finally:
        con.close()


def test_invalid_connection_raises_error():
    bad = 'dbname=invalid_db user=bad password=wrong host=localhost port=5432'
    with pytest.raises(duckdb.Error):
        query_postgres_duckdb(f"SELECT * FROM postgres_scan('{bad}', 'public', 'oncology_icu_base')")


def test_cohort_has_one_row_per_icu_stay(db_conn_str):
    row = pg(db_conn_str, "SELECT COUNT(*) AS n, COUNT(DISTINCT icustay_id) AS n_stays FROM oncology_icu_base")
    assert row.loc[0, "n"] == row.loc[0, "n_stays"] > 0


def test_cohort_stays_are_at_least_48h(db_conn_str):
    row = pg(db_conn_str, "SELECT MIN(EXTRACT(EPOCH FROM (outtime - intime)) / 3600) AS min_hours FROM oncology_icu_base")
    assert row.loc[0, "min_hours"] >= 48


def test_vitals_keep_time_of_day(db_conn_str):
    row = pg(db_conn_str, "SELECT AVG((charttime::time <> '00:00')::int) AS frac FROM all_vitals_48h")
    assert row.loc[0, "frac"] > 0.5


def test_extraction_window_is_first_48h(db_conn_str):
    for view in ("all_labs_48h", "all_vitals_48h"):
        row = pg(db_conn_str, f"""
            SELECT COUNT(*) AS outside FROM {view} v JOIN oncology_icu_base c USING (icustay_id)
            WHERE v.charttime < c.intime OR v.charttime >= c.intime + INTERVAL '48 hours'""")
        assert row.loc[0, "outside"] == 0, view
