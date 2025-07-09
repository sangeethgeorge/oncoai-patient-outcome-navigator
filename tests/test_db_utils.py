#tests/test_db_utils.py

import pytest
import pandas as pd
from oncoai_prototype.utils.db_utils import connect_to_postgres, query_postgres_duckdb

# Use a test-friendly or real local Postgres instance
PG_CONN_STR = "dbname='mimic-iii' user='sangeethgeorge' password='12345' host='localhost' port='5432'"
VALID_QUERY = """
    SELECT *
    FROM postgres_scan('dbname=mimic-iii user=sangeethgeorge password=12345 host=localhost port=5432', 
                       'public', 'patients')
    LIMIT 5
"""

INVALID_QUERY = """
    SELECT *
    FROM postgres_scan('dbname=invalid user=bad password=wrong host=localhost port=5432', 
                       'public', 'patients')
"""

def test_db_connection():
    con = connect_to_postgres(PG_CONN_STR)
    assert con is not None

def test_valid_query_execution():
    df = query_postgres_duckdb(VALID_QUERY)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert "subject_id" in df.columns

def test_invalid_connection_raises_error():
    with pytest.raises(Exception):
        query_postgres_duckdb(INVALID_QUERY)
