# src/oncoai_prototype/utils/db_utils.py
# Returns a connection string for DuckDB to scan Postgres via postgres_scanner extension.

import duckdb

def connect_to_postgres(conn_str: str):
    """Establish a DuckDB connection with postgres_scanner extension loaded."""
    try:
        duckdb.sql("INSTALL postgres_scanner;")
    except duckdb.CatalogException:
        pass  # Already installed

    duckdb.sql("LOAD postgres_scanner;")
    return duckdb.connect()

def query_postgres_duckdb(query: str):
    """Run a SQL query against Postgres using DuckDB's postgres_scanner."""
    con = duckdb.connect()
    try:
        con.sql("LOAD postgres_scanner;")
        return con.execute(query).df()
    finally:
        con.close()

