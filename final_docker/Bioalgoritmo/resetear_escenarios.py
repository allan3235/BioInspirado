"""
resetear_escenarios.py — Limpia la BD y recarga los escenarios nuevos.

Borra en orden correcto (respetando FK):
  resultados_diagnostico → metricas → hiperparametros →
  tareas_evaluacion → modelos → escenarios_busqueda → experimentos

NO toca: imagenes, imagenes_uso

Uso (desde la carpeta final_docker):
    docker compose run --rm --no-deps --entrypoint python worker Bioalgoritmo/resetear_escenarios.py
"""

import sys
import os
from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

load_dotenv(os.path.join(_REPO_ROOT, "ManejoDeDatos", ".env"))

from sqlalchemy import text
from ManejoDeDatos.basededatos import (
    engine, iniciar_bd, obtener_sesion, cargar_escenarios,
)
from cargar_escenarios import ESCENARIOS


def contar_filas(conn, tabla: str) -> int:
    return conn.execute(text(f"SELECT COUNT(*) FROM {tabla}")).scalar()


def mostrar_estado(conn, titulo: str):
    tablas_seguras  = ["escenarios_busqueda", "experimentos", "modelos",
                       "metricas", "hiperparametros", "resultados_diagnostico",
                       "tareas_evaluacion"]
    tablas_intocables = ["imagenes", "imagenes_uso"]
    print(f"\n{titulo}")
    print("-" * 50)
    for t in tablas_seguras:
        print(f"  {t:<35} {contar_filas(conn, t):>6} filas")
    for t in tablas_intocables:
        print(f"  {t:<35} {contar_filas(conn, t):>6} filas  << intacta")


def limpiar(conn):
    orden = [
        "resultados_diagnostico",
        "metricas",
        "hiperparametros",
        "tareas_evaluacion",
        "modelos",
        "escenarios_busqueda",
        "experimentos",
    ]
    for tabla in orden:
        n = contar_filas(conn, tabla)
        conn.execute(text(f"DELETE FROM {tabla}"))
        print(f"  Eliminados {n:>6} registros de '{tabla}'")
    conn.commit()


def main():
    print("=" * 50)
    print("RESET DE ESCENARIOS")
    print("=" * 50)

    iniciar_bd()

    with engine.begin() as conn:
        mostrar_estado(conn, "Estado ANTES:")
        print("\nEliminando datos (imagenes intactas)...")
        limpiar(conn)

    with engine.connect() as conn:
        mostrar_estado(conn, "Estado DESPUÉS de limpiar:")

    print(f"\nCargando {len(ESCENARIOS)} escenarios nuevos...")
    sesion = obtener_sesion()
    cargar_escenarios(sesion, ESCENARIOS)
    sesion.close()

    with engine.connect() as conn:
        n = contar_filas(conn, "escenarios_busqueda")
        ni = contar_filas(conn, "imagenes")
        niu = contar_filas(conn, "imagenes_uso")
        print(f"\nResultado final:")
        print(f"  escenarios_busqueda : {n} escenarios listos")
        print(f"  imagenes            : {ni} (sin cambios)")
        print(f"  imagenes_uso        : {niu} (sin cambios)")

    print("\nListo. Ahora lanza los contenedores:")
    print("  docker compose up --scale worker=40")


if __name__ == "__main__":
    main()
