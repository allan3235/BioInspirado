"""
cargar_escenarios.py — Correr UNA SOLA VEZ desde tu PC antes de lanzar Docker.

Inserta los escenarios en la BD. Luego cada contenedor toma uno automáticamente.

Uso:
    python Bioalgoritmo/cargar_escenarios.py

Luego:
    docker compose up --scale worker=3
"""

import sys
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ManejoDeDatos.basededatos import iniciar_bd, obtener_sesion, cargar_escenarios

ESCENARIOS = [
    # ── Escenario 1: GA agresivo, learning rates altos ───────────────────────
    {
        "nombre": "GA_lr_alto_poblacion_grande",
        "algoritmo": "GA",
        "seed": 42,
        "poblacion_size": 20,
        "num_generaciones": 10,
        "prob_mutacion": 0.3,
        "prob_cruce": 0.8,
        "search_space_json": {
            "learning_rate":    [1e-3, 5e-3, 1e-2],
            "batch_size":       [16, 32],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [20, 30],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── Escenario 2: GA conservador, learning rates bajos ────────────────────
    {
        "nombre": "GA_lr_bajo_dropout_alto",
        "algoritmo": "GA",
        "seed": 7,
        "poblacion_size": 15,
        "num_generaciones": 8,
        "prob_mutacion": 0.1,
        "prob_cruce": 0.9,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "rmsprop"],
            "epochs":           [10, 20],
            "dropout_rate":     [0.4, 0.5],
            "dense_units":      [128, 256],
            "filters":          [32, 64, 128],
            "use_augmentation": [True, False],
        },
    },
    # ── Escenario 3: PSO estándar ─────────────────────────────────────────────
    {
        "nombre": "PSO_estandar",
        "algoritmo": "PSO",
        "seed": 99,
        "poblacion_size": 20,
        "num_generaciones": 10,
        "w_inercia":    0.5,
        "c1_cognitivo": 1.5,
        "c2_social":    1.5,
        "search_space_json": {
            "learning_rate":    [1e-4, 1e-3, 5e-3],
            "batch_size":       [16, 32, 64],
            "optimizer":        ["adam", "sgd", "rmsprop"],
            "epochs":           [10, 20, 30],
            "dropout_rate":     [0.2, 0.3, 0.5],
            "dense_units":      [128, 256, 512],
            "filters":          [32, 64, 128],
            "use_augmentation": [True, False],
        },
    },
    # ── Escenario 4: PSO explorador (inercia alta) ────────────────────────────
    {
        "nombre": "PSO_exploracion_alta",
        "algoritmo": "PSO",
        "seed": 13,
        "poblacion_size": 25,
        "num_generaciones": 12,
        "w_inercia":    0.9,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size":       [16, 32],
            "optimizer":        ["adam", "adamw", "sgd"],
            "epochs":           [15, 25, 35],
            "dropout_rate":     [0.1, 0.3, 0.5],
            "dense_units":      [128, 256, 512],
            "filters":          [32, 64, 128],
            "use_augmentation": [True],
        },
    },
]


if __name__ == "__main__":
    print("Inicializando tablas...")
    iniciar_bd()

    sesion = obtener_sesion()
    cargar_escenarios(sesion, ESCENARIOS)
    sesion.close()

    print(f"{len(ESCENARIOS)} escenarios cargados.")
    print("Ahora lanza los contenedores: docker compose up --scale worker=3")
