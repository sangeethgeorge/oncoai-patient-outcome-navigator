# src/oncoai_prototype/utils/db_utils.py
# Returns a connection string for DuckDB to scan Postgres via postgres_scanner extension.

import duckdb

def connect_to_postgres(conn_str: str):
    duckdb.sql("INSTALL postgres_scanner;")
    duckdb.sql("LOAD postgres_scanner;")
    return duckdb.connect()

def query_postgres_duckdb(query: str):
    con = duckdb.connect()
    return con.execute(query).df()
