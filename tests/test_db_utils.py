#tests/test_db_utils.py

import pytest
import pandas as pd
from dotenv import load_dotenv
import os

from oncoai_prototype.utils.db_utils import connect_to_postgres, query_postgres_duckdb

# Load environment variables
load_dotenv()
PG_CONN_STR = os.getenv("ONCOAI_POSTGRES_CONN_STR")

# --- Fixtures ---

@pytest.fixture(scope="module")
def postgres_conn_str():
    if PG_CONN_STR is None:
        pytest.skip("Environment variable ONCOAI_POSTGRES_CONN_STR is not set.")
    return PG_CONN_STR

@pytest.fixture(scope="module")
def valid_duckdb_query(postgres_conn_str):
    return f"""
        SELECT *
        FROM postgres_scan('{postgres_conn_str}', 'public', 'oncology_icu_base')
        LIMIT 5
    """

@pytest.fixture(scope="module")
def invalid_duckdb_query():
    return """
        SELECT *
        FROM postgres_scan('dbname=invalid_db user=bad password=wrong host=localhost port=5432',
                           'public', 'non_existent_table')
    """

# --- Tests ---

def test_db_connection(postgres_conn_str):
    con = connect_to_postgres(postgres_conn_str)
    assert con is not None
    con.close()

def test_valid_query_execution(valid_duckdb_query):
    df = query_postgres_duckdb(valid_duckdb_query)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "subject_id" in df.columns or "hadm_id" in df.columns  # Adjust based on table used

def test_invalid_connection_raises_error(invalid_duckdb_query):
    with pytest.raises(Exception):
        query_postgres_duckdb(invalid_duckdb_query)


