"""
main.py — Punto de entrada para ejecución distribuida en Docker.

Cada contenedor:
  1. Se conecta a PostgreSQL (variable base_datos)
  2. Toma el siguiente escenario pendiente (SELECT FOR UPDATE SKIP LOCKED)
  3. Corre el optimizador bio-inspirado (GA o PSO)
  4. Guarda resultado en la BD y repite hasta no haber más escenarios

Variables de entorno:
  base_datos  → postgresql://user:pass@host:5432/dbname
  HOSTNAME    → identificador del nodo (lo asigna Docker automáticamente)
  GPU         → 1/true activa GPU (default), 0/false la desactiva (CPU only)
"""

import os
import sys
import time
from datetime import timedelta

# Variable de entorno GPU=1/true activa GPU, GPU=0/false la apaga
_gpu_env = os.environ.get("GPU", "1").strip().lower()
if _gpu_env in ("0", "false", "no"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sqlalchemy import func, select
from ManejoDeDatos.basededatos import (
    EscenarioBusqueda,
    iniciar_bd,
    obtener_sesion,
    tomar_escenario,
    completar_escenario,
    fallar_escenario,
    reintentar_escenarios_colgados,
)
from ManejoDeDatos.data_loader import load_dataset_bd
from bio_optimizer_core import (
    set_seeds,
    validar_parametros_ga,
    GeneticAlgorithm,
    ParticleSwarmOptimizer,
    guardar_individuo_bd,
    serializar_modelo,
    build_model,
    crear_experimento,
    preparar_datasets,
)


def _progreso(sesion, nodo_id: str, inicio_global: float) -> str:
    total = sesion.execute(select(func.count()).select_from(EscenarioBusqueda)).scalar()
    pendientes = sesion.execute(
        select(func.count()).select_from(EscenarioBusqueda)
        .where(EscenarioBusqueda.estado == "pendiente")
    ).scalar()
    completados = sesion.execute(
        select(func.count()).select_from(EscenarioBusqueda)
        .where(EscenarioBusqueda.estado == "completado")
    ).scalar()

    procesados = total - pendientes
    pct = (completados / total * 100) if total else 0
    elapsed = time.time() - inicio_global

    if completados > 0:
        eta_seg = (elapsed / completados) * (total - completados)
        eta_str = str(timedelta(seconds=int(eta_seg)))
    else:
        eta_str = "calculando..."

    return (
        f"[{nodo_id}] Progreso: {completados}/{total} completados "
        f"({pct:.1f}%) | en proceso: {procesados - completados} | "
        f"ETA: {eta_str}"
    )


def ejecutar_escenario(escenario, sesion) -> tuple:
    set_seeds(escenario.seed)

    train_ds, val_ds, class_names = load_dataset_bd()
    num_classes = len(class_names)

    experimento_id = crear_experimento(sesion, escenario.nombre)
    search_space = escenario.search_space_json

    if escenario.algoritmo == "GA":
        validar_parametros_ga(escenario.prob_mutacion, escenario.prob_cruce)
        optimizer = GeneticAlgorithm(
            search_space=search_space,
            poblacion_size=escenario.poblacion_size,
            num_generaciones=escenario.num_generaciones,
            prob_mutacion=escenario.prob_mutacion,
            prob_cruce=escenario.prob_cruce,
            seed=escenario.seed,
        )
    elif escenario.algoritmo == "PSO":
        optimizer = ParticleSwarmOptimizer(
            search_space=search_space,
            poblacion_size=escenario.poblacion_size,
            num_generaciones=escenario.num_generaciones,
            w_inercia=escenario.w_inercia,
            c1_cognitivo=escenario.c1_cognitivo,
            c2_social=escenario.c2_social,
            seed=escenario.seed,
        )
    else:
        raise ValueError(f"Algoritmo desconocido: {escenario.algoritmo}")

    mejor_individuo, fitness_history = optimizer.run(
        train_ds, val_ds, num_classes, experimento_id, sesion
    )
    mejor_fitness = fitness_history[-1]

    mejor_modelo = build_model(mejor_individuo, num_classes)
    ds_train_final, ds_val_final = preparar_datasets(mejor_individuo, train_ds, val_ds)
    mejor_history = mejor_modelo.fit(
        ds_train_final,
        validation_data=ds_val_final,
        epochs=mejor_individuo["epochs"],
        verbose=1,
    )
    print("[main] Serializando mejor modelo...")
    mejor_modelo_bytes = serializar_modelo(mejor_modelo)
    guardar_individuo_bd(
        sesion, experimento_id, mejor_individuo, mejor_history,
        modelo_bytes=mejor_modelo_bytes,
    )

    return experimento_id, mejor_fitness, mejor_individuo, fitness_history


def main():
    nodo_id = os.environ.get("HOSTNAME", "nodo-local")
    print(f"[{nodo_id}] Iniciando nodo...")

    iniciar_bd()
    sesion = obtener_sesion()

    rescatados = reintentar_escenarios_colgados(sesion, timeout_minutos=120)
    if rescatados:
        print(f"[{nodo_id}] {rescatados} escenario(s) colgado(s) reencolado(s).")

    inicio_global = time.time()
    escenarios_procesados = 0

    while True:
        escenario = tomar_escenario(sesion, nodo_id)

        if escenario is None:
            print(f"[{nodo_id}] No hay escenarios pendientes. Nodo terminado.")
            break

        print(f"[{nodo_id}] Escenario #{escenario.id}: '{escenario.nombre}' ({escenario.algoritmo})")
        print(_progreso(sesion, nodo_id, inicio_global))

        t0 = time.time()
        try:
            experimento_id, mejor_fitness, mejor_individuo, fitness_history = ejecutar_escenario(
                escenario, sesion
            )
            completar_escenario(
                sesion,
                escenario_id=escenario.id,
                experimento_id=experimento_id,
                mejor_fitness=mejor_fitness,
                mejor_individuo=mejor_individuo,
                fitness_history=fitness_history,
            )
            escenarios_procesados += 1
            duracion = str(timedelta(seconds=int(time.time() - t0)))
            print(
                f"[{nodo_id}] Escenario #{escenario.id} completado en {duracion}. "
                f"Fitness: {mejor_fitness:.4f}"
            )
            print(_progreso(sesion, nodo_id, inicio_global))

        except Exception as e:
            fallar_escenario(sesion, escenario.id, str(e))
            print(f"[{nodo_id}] Error en escenario #{escenario.id}: {e}")

    sesion.close()
    total_tiempo = str(timedelta(seconds=int(time.time() - inicio_global)))
    print(f"[{nodo_id}] Total escenarios procesados: {escenarios_procesados} en {total_tiempo}")


if __name__ == "__main__":
    main()
