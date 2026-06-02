"""
cargar_escenarios.py — Correr UNA SOLA VEZ desde tu PC antes de lanzar Docker.

Inserta los escenarios en la BD. Luego cada contenedor toma uno automáticamente.

Uso:
    python Bioalgoritmo/cargar_escenarios.py

Variables de entorno (.env):
    test=1  → inserta solo el escenario de prueba (1 epoch, 2 individuos)
    test=0  → inserta todos los escenarios reales

Luego:
    docker compose up --scale worker=40
"""

import sys
import os
from dotenv import load_dotenv

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

load_dotenv(os.path.join(_REPO_ROOT, "ManejoDeDatos", ".env"))

from ManejoDeDatos.basededatos import iniciar_bd, obtener_sesion, cargar_escenarios

ESCENARIO_TEST = [
    {
        "nombre": "TEST_verificacion_bd",
        "algoritmo": "GA",
        "seed": 0,
        "poblacion_size": 2,
        "num_generaciones": 1,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.8,
        "search_space_json": {
            "learning_rate":    [1e-3],
            "batch_size":       [32],
            "optimizer":        ["adam"],
            "epochs":           [1],
            "dropout_rate":     [0.3],
            "dense_units":      [64],
            "filters":          [16],
            "use_augmentation": [False],
        },
    },
]

ESCENARIOS = [
    # ══════════════════════════════════════════════════════════════════════════
    # ── GENETIC ALGORITHM (GA) — 20 escenarios ────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # ── GA-01: AdamW óptimo — espacio centrado en lr=1e-3 ────────────────────
    {
        "nombre": "GA_adamw_optimo_100gen",
        "algoritmo": "GA",
        "seed": 42,
        "poblacion_size": 20,
        "num_generaciones": 100,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [5e-4, 1e-3, 2e-3],
            "batch_size":       [32],
            "optimizer":        ["adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-02: AdamW lr fino — rejilla fina alrededor del óptimo ─────────────
    {
        "nombre": "GA_adamw_lr_fino_120gen",
        "algoritmo": "GA",
        "seed": 7,
        "poblacion_size": 20,
        "num_generaciones": 120,
        "prob_mutacion": 0.18,
        "prob_cruce": 0.88,
        "search_space_json": {
            "learning_rate":    [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.25, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-03: Todos los optimizadores — comparación equilibrada ─────────────
    {
        "nombre": "GA_todos_optim_equilibrado_150gen",
        "algoritmo": "GA",
        "seed": 99,
        "poblacion_size": 20,
        "num_generaciones": 150,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw", "sgd", "rmsprop"],
            "epochs":           [20, 30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-04: Mutación alta — exploración agresiva ───────────────────────────
    {
        "nombre": "GA_mutacion_alta_100gen",
        "algoritmo": "GA",
        "seed": 13,
        "poblacion_size": 25,
        "num_generaciones": 100,
        "prob_mutacion": 0.4,
        "prob_cruce": 0.75,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size":       [16, 32, 64],
            "optimizer":        ["adam", "adamw", "sgd", "rmsprop"],
            "epochs":           [20, 30, 40],
            "dropout_rate":     [0.1, 0.2, 0.3, 0.5],
            "dense_units":      [128, 256, 512],
            "filters":          [32, 64, 128],
            "use_augmentation": [True, False],
        },
    },
    # ── GA-05: Mutación baja, cruce alto — convergencia rápida ───────────────
    {
        "nombre": "GA_convergencia_cruce_alto_150gen",
        "algoritmo": "GA",
        "seed": 21,
        "poblacion_size": 20,
        "num_generaciones": 150,
        "prob_mutacion": 0.08,
        "prob_cruce": 0.95,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-06: Población grande — diversidad máxima ───────────────────────────
    {
        "nombre": "GA_poblacion_grande_80gen",
        "algoritmo": "GA",
        "seed": 55,
        "poblacion_size": 50,
        "num_generaciones": 80,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [25, 35, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-07: Población pequeña, muchas generaciones — refinamiento largo ────
    {
        "nombre": "GA_poblacion_pequena_200gen",
        "algoritmo": "GA",
        "seed": 77,
        "poblacion_size": 8,
        "num_generaciones": 200,
        "prob_mutacion": 0.25,
        "prob_cruce": 0.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True, False],
        },
    },
    # ── GA-08: Arquitectura grande — más capacidad ────────────────────────────
    {
        "nombre": "GA_arquitectura_grande_120gen",
        "algoritmo": "GA",
        "seed": 33,
        "poblacion_size": 20,
        "num_generaciones": 120,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-09: Arquitectura compacta — modelo ligero ──────────────────────────
    {
        "nombre": "GA_arquitectura_compacta_120gen",
        "algoritmo": "GA",
        "seed": 101,
        "poblacion_size": 20,
        "num_generaciones": 120,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [20, 30, 40],
            "dropout_rate":     [0.1, 0.2, 0.3],
            "dense_units":      [64, 128, 256],
            "filters":          [16, 32],
            "use_augmentation": [True],
        },
    },
    # ── GA-10: Sin augmentation — comparación base ────────────────────────────
    {
        "nombre": "GA_sin_augmentation_150gen",
        "algoritmo": "GA",
        "seed": 202,
        "poblacion_size": 20,
        "num_generaciones": 150,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [False],
        },
    },
    # ── GA-11: Dropout alto — regularización fuerte ───────────────────────────
    {
        "nombre": "GA_dropout_alto_130gen",
        "algoritmo": "GA",
        "seed": 303,
        "poblacion_size": 20,
        "num_generaciones": 130,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.4, 0.5],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-12: Dropout bajo — red menos regularizada ──────────────────────────
    {
        "nombre": "GA_dropout_bajo_130gen",
        "algoritmo": "GA",
        "seed": 404,
        "poblacion_size": 20,
        "num_generaciones": 130,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.88,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.0, 0.1, 0.2],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-13: Batch grande — gradiente estable ───────────────────────────────
    {
        "nombre": "GA_batch_grande_110gen",
        "algoritmo": "GA",
        "seed": 505,
        "poblacion_size": 20,
        "num_generaciones": 110,
        "prob_mutacion": 0.22,
        "prob_cruce": 0.83,
        "search_space_json": {
            "learning_rate":    [5e-4, 1e-3, 2e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-14: Epochs largos por individuo ────────────────────────────────────
    {
        "nombre": "GA_epochs_largos_100gen",
        "algoritmo": "GA",
        "seed": 606,
        "poblacion_size": 15,
        "num_generaciones": 100,
        "prob_mutacion": 0.18,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [50, 60, 80],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-15: Espacio amplio — máxima exploración ────────────────────────────
    {
        "nombre": "GA_espacio_amplio_200gen",
        "algoritmo": "GA",
        "seed": 707,
        "poblacion_size": 25,
        "num_generaciones": 200,
        "prob_mutacion": 0.25,
        "prob_cruce": 0.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size":       [16, 32, 64],
            "optimizer":        ["adam", "adamw", "sgd", "rmsprop"],
            "epochs":           [20, 30, 40, 50],
            "dropout_rate":     [0.1, 0.2, 0.3, 0.4, 0.5],
            "dense_units":      [128, 256, 512],
            "filters":          [32, 64, 128],
            "use_augmentation": [True, False],
        },
    },
    # ── GA-16: SGD — línea base de comparación ────────────────────────────────
    {
        "nombre": "GA_sgd_baseline_100gen",
        "algoritmo": "GA",
        "seed": 808,
        "poblacion_size": 20,
        "num_generaciones": 100,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-3, 5e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["sgd"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-17: Balance total — espacio medio, 180 gen ─────────────────────────
    {
        "nombre": "GA_balance_total_180gen",
        "algoritmo": "GA",
        "seed": 909,
        "poblacion_size": 22,
        "num_generaciones": 180,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 2e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw", "rmsprop"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-18: AdamW + arquitectura intermedia — refinamiento 160 gen ─────────
    {
        "nombre": "GA_adamw_arquitectura_media_160gen",
        "algoritmo": "GA",
        "seed": 1010,
        "poblacion_size": 20,
        "num_generaciones": 160,
        "prob_mutacion": 0.22,
        "prob_cruce": 0.82,
        "search_space_json": {
            "learning_rate":    [5e-4, 1e-3, 2e-3],
            "batch_size":       [32],
            "optimizer":        ["adamw"],
            "epochs":           [40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 384, 512],
            "filters":          [32, 48, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-19: Búsqueda lr × batch — interacción entre los dos ───────────────
    {
        "nombre": "GA_lr_batch_interaccion_140gen",
        "algoritmo": "GA",
        "seed": 1111,
        "poblacion_size": 20,
        "num_generaciones": 140,
        "prob_mutacion": 0.2,
        "prob_cruce": 0.85,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
            "batch_size":       [16, 32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── GA-20: Rápido — exploración inicial ──────────────────────────────────
    {
        "nombre": "GA_rapido_60gen",
        "algoritmo": "GA",
        "seed": 1212,
        "poblacion_size": 30,
        "num_generaciones": 60,
        "prob_mutacion": 0.3,
        "prob_cruce": 0.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [20, 30, 40],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True, False],
        },
    },

    # ══════════════════════════════════════════════════════════════════════════
    # ── PARTICLE SWARM OPTIMIZATION (PSO) — 20 escenarios ─────────────────────
    # ══════════════════════════════════════════════════════════════════════════

    # ── PSO-01: Clásico Clerc-Kennedy, adamw ──────────────────────────────────
    {
        "nombre": "PSO_clasico_adamw_120gen",
        "algoritmo": "PSO",
        "seed": 2001,
        "poblacion_size": 30,
        "num_generaciones": 120,
        "w_inercia":    0.729,
        "c1_cognitivo": 1.494,
        "c2_social":    1.494,
        "search_space_json": {
            "learning_rate":    [5e-4, 1e-3, 2e-3],
            "batch_size":       [32],
            "optimizer":        ["adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-02: Clásico Clerc-Kennedy, todos los optimizadores ───────────────
    {
        "nombre": "PSO_clasico_todos_optim_150gen",
        "algoritmo": "PSO",
        "seed": 2002,
        "poblacion_size": 30,
        "num_generaciones": 150,
        "w_inercia":    0.729,
        "c1_cognitivo": 1.494,
        "c2_social":    1.494,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw", "sgd", "rmsprop"],
            "epochs":           [20, 30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-03: Exploración alta — inercia alta ────────────────────────────────
    {
        "nombre": "PSO_exploracion_alta_100gen",
        "algoritmo": "PSO",
        "seed": 2003,
        "poblacion_size": 25,
        "num_generaciones": 100,
        "w_inercia":    0.9,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size":       [16, 32, 64],
            "optimizer":        ["adam", "adamw", "sgd", "rmsprop"],
            "epochs":           [20, 30, 40],
            "dropout_rate":     [0.1, 0.2, 0.3, 0.5],
            "dense_units":      [128, 256, 512],
            "filters":          [32, 64, 128],
            "use_augmentation": [True, False],
        },
    },
    # ── PSO-04: Explotación alta — inercia baja ────────────────────────────────
    {
        "nombre": "PSO_explotacion_alta_150gen",
        "algoritmo": "PSO",
        "seed": 2004,
        "poblacion_size": 20,
        "num_generaciones": 150,
        "w_inercia":    0.2,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-05: Cognitivo dominante — cada partícula sigue su mejor ───────────
    {
        "nombre": "PSO_cognitivo_dominante_130gen",
        "algoritmo": "PSO",
        "seed": 2005,
        "poblacion_size": 20,
        "num_generaciones": 130,
        "w_inercia":    0.5,
        "c1_cognitivo": 3.0,
        "c2_social":    1.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw", "rmsprop"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-06: Social dominante — el enjambre converge al global ─────────────
    {
        "nombre": "PSO_social_dominante_130gen",
        "algoritmo": "PSO",
        "seed": 2006,
        "poblacion_size": 20,
        "num_generaciones": 130,
        "w_inercia":    0.5,
        "c1_cognitivo": 1.0,
        "c2_social":    3.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw", "rmsprop"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-07: Balance c1=c2=2.0, inercia media ─────────────────────────────
    {
        "nombre": "PSO_balance_c2_150gen",
        "algoritmo": "PSO",
        "seed": 2007,
        "poblacion_size": 20,
        "num_generaciones": 150,
        "w_inercia":    0.6,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 2e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-08: Enjambre grande — amplia cobertura ────────────────────────────
    {
        "nombre": "PSO_enjambre_grande_80gen",
        "algoritmo": "PSO",
        "seed": 2008,
        "poblacion_size": 50,
        "num_generaciones": 80,
        "w_inercia":    0.6,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [25, 35, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-09: Enjambre pequeño, largo — refinamiento lento ─────────────────
    {
        "nombre": "PSO_enjambre_pequeno_200gen",
        "algoritmo": "PSO",
        "seed": 2009,
        "poblacion_size": 8,
        "num_generaciones": 200,
        "w_inercia":    0.7,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True, False],
        },
    },
    # ── PSO-10: AdamW lr fino ─────────────────────────────────────────────────
    {
        "nombre": "PSO_adamw_lr_fino_130gen",
        "algoritmo": "PSO",
        "seed": 2010,
        "poblacion_size": 20,
        "num_generaciones": 130,
        "w_inercia":    0.55,
        "c1_cognitivo": 2.2,
        "c2_social":    1.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 2e-4, 5e-4, 1e-3, 2e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.25, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-11: Sin augmentation ──────────────────────────────────────────────
    {
        "nombre": "PSO_sin_augmentation_150gen",
        "algoritmo": "PSO",
        "seed": 2011,
        "poblacion_size": 20,
        "num_generaciones": 150,
        "w_inercia":    0.6,
        "c1_cognitivo": 1.5,
        "c2_social":    1.5,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4, 0.5],
            "dense_units":      [256, 512],
            "filters":          [32, 64, 128],
            "use_augmentation": [False],
        },
    },
    # ── PSO-12: Arquitectura grande ───────────────────────────────────────────
    {
        "nombre": "PSO_arquitectura_grande_130gen",
        "algoritmo": "PSO",
        "seed": 2012,
        "poblacion_size": 20,
        "num_generaciones": 130,
        "w_inercia":    0.65,
        "c1_cognitivo": 1.8,
        "c2_social":    1.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-13: Arquitectura compacta ─────────────────────────────────────────
    {
        "nombre": "PSO_arquitectura_compacta_130gen",
        "algoritmo": "PSO",
        "seed": 2013,
        "poblacion_size": 20,
        "num_generaciones": 130,
        "w_inercia":    0.65,
        "c1_cognitivo": 1.8,
        "c2_social":    1.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [20, 30, 40],
            "dropout_rate":     [0.1, 0.2, 0.3],
            "dense_units":      [64, 128, 256],
            "filters":          [16, 32],
            "use_augmentation": [True],
        },
    },
    # ── PSO-14: Espacio amplio ─────────────────────────────────────────────────
    {
        "nombre": "PSO_espacio_amplio_180gen",
        "algoritmo": "PSO",
        "seed": 2014,
        "poblacion_size": 30,
        "num_generaciones": 180,
        "w_inercia":    0.6,
        "c1_cognitivo": 1.8,
        "c2_social":    2.2,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size":       [16, 32, 64],
            "optimizer":        ["adam", "adamw", "sgd", "rmsprop"],
            "epochs":           [20, 30, 40, 50],
            "dropout_rate":     [0.1, 0.2, 0.3, 0.4, 0.5],
            "dense_units":      [128, 256, 512],
            "filters":          [32, 64, 128],
            "use_augmentation": [True, False],
        },
    },
    # ── PSO-15: Dropout variado — toda la gama ────────────────────────────────
    {
        "nombre": "PSO_dropout_variado_140gen",
        "algoritmo": "PSO",
        "seed": 2015,
        "poblacion_size": 20,
        "num_generaciones": 140,
        "w_inercia":    0.6,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [5e-4, 1e-3],
            "batch_size":       [32],
            "optimizer":        ["adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-16: Epochs largos por individuo ───────────────────────────────────
    {
        "nombre": "PSO_epochs_largos_110gen",
        "algoritmo": "PSO",
        "seed": 2016,
        "poblacion_size": 15,
        "num_generaciones": 110,
        "w_inercia":    0.55,
        "c1_cognitivo": 2.2,
        "c2_social":    1.8,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [50, 60, 80],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-17: Batch variado — interacción lr × batch ────────────────────────
    {
        "nombre": "PSO_batch_variado_140gen",
        "algoritmo": "PSO",
        "seed": 2017,
        "poblacion_size": 20,
        "num_generaciones": 140,
        "w_inercia":    0.7,
        "c1_cognitivo": 1.5,
        "c2_social":    2.5,
        "search_space_json": {
            "learning_rate":    [5e-4, 1e-3, 2e-3, 5e-3],
            "batch_size":       [16, 32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-18: SGD — línea base ──────────────────────────────────────────────
    {
        "nombre": "PSO_sgd_baseline_110gen",
        "algoritmo": "PSO",
        "seed": 2018,
        "poblacion_size": 20,
        "num_generaciones": 110,
        "w_inercia":    0.7,
        "c1_cognitivo": 1.5,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-3, 5e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["sgd"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-19: Largo convergencia ─────────────────────────────────────────────
    {
        "nombre": "PSO_largo_convergencia_200gen",
        "algoritmo": "PSO",
        "seed": 2019,
        "poblacion_size": 25,
        "num_generaciones": 200,
        "w_inercia":    0.5,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 2e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw", "rmsprop"],
            "epochs":           [30, 40, 50],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True],
        },
    },
    # ── PSO-20: Rápido — exploración inicial ──────────────────────────────────
    {
        "nombre": "PSO_rapido_60gen",
        "algoritmo": "PSO",
        "seed": 2020,
        "poblacion_size": 30,
        "num_generaciones": 60,
        "w_inercia":    0.65,
        "c1_cognitivo": 2.0,
        "c2_social":    2.0,
        "search_space_json": {
            "learning_rate":    [1e-4, 5e-4, 1e-3, 5e-3],
            "batch_size":       [32, 64],
            "optimizer":        ["adam", "adamw"],
            "epochs":           [20, 30, 40],
            "dropout_rate":     [0.2, 0.3, 0.4],
            "dense_units":      [256, 512],
            "filters":          [32, 64],
            "use_augmentation": [True, False],
        },
    },
]


if __name__ == "__main__":
    modo_test = os.getenv("test", "0").strip() == "1"

    print("Inicializando tablas...")
    iniciar_bd()

    sesion = obtener_sesion()

    if modo_test:
        cargar_escenarios(sesion, ESCENARIO_TEST)
        sesion.close()
        print("Modo TEST activo — escenario de prueba cargado: 2 individuos, 1 época.")
        print("Verifica en la BD que se guardó el modelo y luego lanza los escenarios reales con test=0.")
    else:
        cargar_escenarios(sesion, ESCENARIOS)
        sesion.close()
        print(f"{len(ESCENARIOS)} escenarios cargados.")
        print("Ahora lanza los contenedores: docker compose up --scale worker=40")
