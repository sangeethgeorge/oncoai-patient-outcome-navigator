# src/oncoai_prototype/utils/db_utils.py
# DuckDB connections that can scan Postgres via the postgres_scanner extension.

import re
import duckdb

def connect_to_postgres(conn_str: str):
    """Return a DuckDB connection with postgres_scanner loaded (conn_str is used in postgres_scan calls)."""
    con = duckdb.connect()
    con.install_extension("postgres_scanner")
    con.load_extension("postgres_scanner")
    return con

def query_postgres_duckdb(query: str):
    """Run a SQL query against Postgres using DuckDB's postgres_scanner."""
    con = duckdb.connect()
    try:
        con.install_extension("postgres_scanner")
        con.load_extension("postgres_scanner")
        return con.execute(query).df()
    finally:
        con.close()

def run_postgres_sql(conn_str: str, sql: str):
    """Run native Postgres SQL (read-only) through DuckDB and return a DataFrame."""
    con = connect_to_postgres(conn_str)
    try:
        quoted = conn_str.replace("'", "''")
        con.execute(f"ATTACH '{quoted}' AS pg (TYPE postgres, READ_ONLY)")
        return con.execute("SELECT * FROM postgres_query('pg', ?)", [sql]).df()
    finally:
        con.close()

def redact(text: str) -> str:
    """Mask passwords in libpq-style or URL connection strings before printing errors."""
    return re.sub(r"(password=)\S+|(://[^:/@]+:)[^@]+@", lambda m: (m.group(1) or m.group(2)) + "***" + ("" if m.group(1) else "@"), text)
