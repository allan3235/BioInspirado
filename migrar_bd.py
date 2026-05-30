"""
migrar_bd.py — Agrega columnas faltantes a tablas existentes sin borrar datos.
Ejecutar UNA SOLA VEZ desde tu PC antes de volver a lanzar Docker.

Uso: python migrar_bd.py
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, "ManejoDeDatos", ".env"))

import psycopg2
from urllib.parse import urlparse

DB_URL = os.getenv("base_datos")
parsed = urlparse(DB_URL)
conn_params = {
    "host":     parsed.hostname,
    "port":     parsed.port or 5432,
    "user":     parsed.username,
    "password": parsed.password,
    "dbname":   parsed.path.lstrip("/"),
}

MIGRACIONES = [
    # escenarios_busqueda — columnas de resultado que faltaban
    "ALTER TABLE escenarios_busqueda ADD COLUMN IF NOT EXISTS mejor_fitness DOUBLE PRECISION",
    "ALTER TABLE escenarios_busqueda ADD COLUMN IF NOT EXISTS mejor_individuo_json JSON",
    "ALTER TABLE escenarios_busqueda ADD COLUMN IF NOT EXISTS fitness_history_json JSON",
    "ALTER TABLE escenarios_busqueda ADD COLUMN IF NOT EXISTS mensaje_error TEXT",
    "ALTER TABLE escenarios_busqueda ADD COLUMN IF NOT EXISTS experimento_id INTEGER REFERENCES experimentos(id)",

    # tareas_evaluacion — por si también tiene columnas viejas
    "ALTER TABLE tareas_evaluacion ADD COLUMN IF NOT EXISTS fitness DOUBLE PRECISION",
    "ALTER TABLE tareas_evaluacion ADD COLUMN IF NOT EXISTS mensaje_error TEXT",
    "ALTER TABLE tareas_evaluacion ADD COLUMN IF NOT EXISTS nodo_id VARCHAR(100)",
    "ALTER TABLE tareas_evaluacion ADD COLUMN IF NOT EXISTS fecha_inicio TIMESTAMP",
    "ALTER TABLE tareas_evaluacion ADD COLUMN IF NOT EXISTS fecha_fin TIMESTAMP",
    "ALTER TABLE tareas_evaluacion ADD COLUMN IF NOT EXISTS generacion INTEGER",
    "ALTER TABLE tareas_evaluacion ADD COLUMN IF NOT EXISTS individuo_idx INTEGER",

    # imagenes — columna nombre_archivo por si no existe
    "ALTER TABLE imagenes ADD COLUMN IF NOT EXISTS nombre_archivo VARCHAR(255)",

    # resultados_diagnostico — tabla nueva, se crea si no existe
    """
    CREATE TABLE IF NOT EXISTS resultados_diagnostico (
        id SERIAL PRIMARY KEY,
        modelo_id INTEGER REFERENCES modelos(id),
        curva_perdida BYTEA,
        curva_precision BYTEA,
        matriz_confusion BYTEA,
        curva_roc BYTEA,
        formato VARCHAR(10) DEFAULT 'png',
        fecha TIMESTAMP DEFAULT NOW()
    )
    """,
]


def main():
    print(f"Conectando a {conn_params['host']}:{conn_params['port']} / {conn_params['dbname']}...")
    conn = psycopg2.connect(**conn_params)
    conn.autocommit = True
    cur = conn.cursor()

    for sql in MIGRACIONES:
        stmt = sql.strip().splitlines()[0].strip()
        try:
            cur.execute(sql)
            print(f"  OK  {stmt}")
        except Exception as e:
            print(f"  ERROR {stmt}: {e}")

    cur.close()
    conn.close()
    print("\nMigracion completada. Ahora puedes lanzar Docker.")


if __name__ == "__main__":
    main()
