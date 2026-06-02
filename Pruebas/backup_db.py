"""
backup_db.py — Genera un backup SQL de todas las tablas de la base de datos.
Uso: python backup_db.py
Requiere: .env en ManejoDeDatos/ con la variable base_datos
"""

import os
import sys
from datetime import datetime

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "ManejoDeDatos", ".env"))

import psycopg2
from psycopg2 import sql
from urllib.parse import urlparse

DB_URL = os.getenv("base_datos")
if not DB_URL:
    print("ERROR: variable 'base_datos' no encontrada en el .env")
    sys.exit(1)

parsed = urlparse(DB_URL)
conn_params = {
    "host":     parsed.hostname,
    "port":     parsed.port or 5432,
    "user":     parsed.username,
    "password": parsed.password,
    "dbname":   parsed.path.lstrip("/"),
}

TABLAS = None  # se detectan automáticamente desde la BD

fecha = datetime.now().strftime("%Y%m%d_%H%M%S")
archivo = os.path.join(_ROOT, f"backup_bioeslaonda_{fecha}.sql")


def escape_val(val):
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (dict, list)):
        import json
        return "'" + json.dumps(val, ensure_ascii=False).replace("'", "''") + "'"
    if isinstance(val, (bytes, memoryview)):
        hex_str = bytes(val).hex()
        return f"'\\x{hex_str}'"
    return "'" + str(val).replace("'", "''") + "'"


def backup_tabla(cur, tabla: str, f):
    cur.execute(f"SELECT * FROM {tabla}")
    filas = cur.fetchall()
    cols = [desc[0] for desc in cur.description]

    if not filas:
        f.write(f"-- Tabla '{tabla}' vacia\n\n")
        return

    f.write(f"-- Tabla: {tabla} ({len(filas)} filas)\n")
    for fila in filas:
        valores = ", ".join(escape_val(v) for v in fila)
        columnas = ", ".join(cols)
        f.write(f"INSERT INTO {tabla} ({columnas}) VALUES ({valores});\n")
    f.write("\n")
    print(f"  OK {tabla}: {len(filas)} filas")


def obtener_tablas(cur):
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
    """)
    return [row[0] for row in cur.fetchall()]


def main():
    print(f"Conectando a {conn_params['host']}:{conn_params['port']} / {conn_params['dbname']}...")
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    tablas = obtener_tablas(cur)
    print(f"Tablas encontradas: {', '.join(tablas)}\n")

    with open(archivo, "w", encoding="utf-8") as f:
        f.write(f"-- Backup: {conn_params['dbname']}\n")
        f.write(f"-- Fecha: {datetime.now().isoformat()}\n")
        f.write(f"-- Host: {conn_params['host']}:{conn_params['port']}\n\n")
        f.write("SET client_encoding = 'UTF8';\n")
        f.write("SET standard_conforming_strings = on;\n\n")

        for tabla in tablas:
            try:
                backup_tabla(cur, tabla, f)
            except Exception as e:
                f.write(f"-- ERROR en tabla '{tabla}': {e}\n\n")
                print(f"  ERROR {tabla}: {e}")
                conn.rollback()

    cur.close()
    conn.close()
    size_kb = os.path.getsize(archivo) / 1024
    print(f"\nBackup guardado: {archivo} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
