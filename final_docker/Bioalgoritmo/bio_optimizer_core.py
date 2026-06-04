import io
import queue
import random
import sys
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validar_parametros_ga(prob_mutacion: float, prob_cruce: float) -> None:
    if not (0.0 <= prob_mutacion <= 1.0):
        raise ValueError(f"PROB_MUTACION debe estar en [0.0, 1.0], valor recibido: {prob_mutacion}")
    if not (0.0 <= prob_cruce <= 1.0):
        raise ValueError(f"PROB_CRUCE debe estar en [0.0, 1.0], valor recibido: {prob_cruce}")


_OPTIMIZERS_MAP = {
    "adam":    torch.optim.Adam,
    "sgd":     torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adamw":   torch.optim.AdamW,
}


# ─── Modelos PyTorch ──────────────────────────────────────────────────────────

class _CNN(nn.Module):
    """CNN propia: tres bloques Conv→BN→ReLU→MaxPool, GlobalAvgPool, Dense, Dropout, salida."""

    def __init__(self, filters: int, dense_units: int, dropout_rate: float, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(filters, filters * 2, 3, padding=1),
            nn.BatchNorm2d(filters * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(filters * 2, filters * 4, 3, padding=1),
            nn.BatchNorm2d(filters * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(filters * 4, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


class _VGG16Transfer(nn.Module):
    """VGG16 transfer learning — modelo de comparación/baseline."""

    def __init__(self, dense_units: int, dropout_rate: float, num_classes: int):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def _make_optimizer(optimizer_name: str, model: nn.Module, learning_rate: float):
    opt_cls = _OPTIMIZERS_MAP[optimizer_name]
    kwargs = {"lr": learning_rate}
    if optimizer_name == "adamw":
        kwargs["weight_decay"] = 1e-4
    elif optimizer_name == "sgd":
        kwargs["momentum"] = 0.9
    return opt_cls(model.parameters(), **kwargs)


def build_model(individuo: dict, num_classes: int, device=None):
    """Retorna (model, optimizer, criterion) para la CNN propia."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_name = individuo["optimizer"]
    if optimizer_name not in _OPTIMIZERS_MAP:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no soportado. Válidos: {set(_OPTIMIZERS_MAP.keys())}"
        )
    model = _CNN(
        filters=individuo.get("filters", 32),
        dense_units=individuo["dense_units"],
        dropout_rate=individuo["dropout_rate"],
        num_classes=num_classes,
    ).to(device)
    optimizer = _make_optimizer(optimizer_name, model, individuo["learning_rate"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, optimizer, criterion


def build_model_vgg16(individuo: dict, num_classes: int, device=None):
    """Retorna (model, optimizer, criterion) para VGG16 transfer learning."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer_name = individuo["optimizer"]
    if optimizer_name not in _OPTIMIZERS_MAP:
        raise ValueError(
            f"Optimizador '{optimizer_name}' no soportado. Válidos: {set(_OPTIMIZERS_MAP.keys())}"
        )
    model = _VGG16Transfer(
        dense_units=individuo["dense_units"],
        dropout_rate=individuo["dropout_rate"],
        num_classes=num_classes,
    ).to(device)
    optimizer = _make_optimizer(optimizer_name, model, individuo["learning_rate"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    return model, optimizer, criterion


# ─── Bucle de entrenamiento ───────────────────────────────────────────────────

def _train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total_correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return (total_loss / total, total_correct / total) if total else (0.0, 0.0)


def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            total_correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return (total_loss / total, total_correct / total) if total else (0.0, 0.0)


def entrenar_modelo(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    """Entrena el modelo y retorna un dict con el historial de métricas."""
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    for epoch in range(epochs):
        tr_loss, tr_acc = _train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = _val_epoch(model, val_loader, criterion, device)
        hist["loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["accuracy"].append(tr_acc)
        hist["val_accuracy"].append(vl_acc)
        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"loss: {tr_loss:.4f}  acc: {tr_acc:.4f} | "
            f"val_loss: {vl_loss:.4f}  val_acc: {vl_acc:.4f}"
        )
    return hist


# ─── Persistencia ────────────────────────────────────────────────────────────

def _import_bd_models():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from ManejoDeDatos.basededatos import Experimento, Modelo, Metrica, Hiperparametro
    return Experimento, Modelo, Metrica, Hiperparametro


def _nueva_sesion():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from ManejoDeDatos.basededatos import obtener_sesion
    return obtener_sesion()


# ─── Cola de reintentos ───────────────────────────────────────────────────────

@dataclass
class _TareaGuardado:
    experimento_id: int
    individuo: dict
    historia: dict
    arquitectura: str
    modelo_bytes: Optional[bytes] = None


_cola_reintentos: queue.Queue = queue.Queue()
_worker_iniciado = False
_worker_lock = threading.Lock()


def _insertar_en_bd(sesion, tarea: _TareaGuardado) -> int:
    """Lógica de inserción compartida entre la ruta normal y los reintentos."""
    _, Modelo, Metrica, Hiperparametro = _import_bd_models()

    modelo = Modelo(
        experimento_id=tarea.experimento_id,
        arquitectura=tarea.arquitectura,
        fecha=datetime.utcnow(),
    )
    if tarea.modelo_bytes:
        modelo.archivo = tarea.modelo_bytes
        modelo.formato = "pt"
    sesion.add(modelo)
    sesion.flush()

    sesion.add(Hiperparametro(
        modelo_id=modelo.id,
        tasa_aprendizaje=tarea.individuo.get("learning_rate"),
        tamano_lote=tarea.individuo.get("batch_size"),
        optimizador=tarea.individuo.get("optimizer"),
        epocas=tarea.individuo.get("epochs"),
        aumento_datos=tarea.individuo.get("use_augmentation"),
        neuronas_densas=tarea.individuo.get("dense_units"),
        tasa_dropout=tarea.individuo.get("dropout_rate"),
        filtros_conv=tarea.individuo.get("filters"),
    ))

    hist = tarea.historia
    for epoch_idx in range(len(hist.get("loss", []))):
        sesion.add(Metrica(
            modelo_id=modelo.id,
            epoca=epoch_idx + 1,
            perdida_entrenamiento=_safe_get(hist, "loss", epoch_idx),
            perdida_validacion=_safe_get(hist, "val_loss", epoch_idx),
            precision_entrenamiento=_safe_get(hist, "accuracy", epoch_idx),
            precision_validacion=_safe_get(hist, "val_accuracy", epoch_idx),
        ))

    sesion.commit()
    return modelo.id


def _worker_reintentos():
    pendientes: list = []
    while True:
        time.sleep(60)
        while True:
            try:
                pendientes.append(_cola_reintentos.get_nowait())
            except queue.Empty:
                break

        if not pendientes:
            continue

        print(f"[reintento BD] {len(pendientes)} modelo(s) pendiente(s) — intentando guardar...")
        exitosas = []
        for i, tarea in enumerate(pendientes):
            sesion = None
            try:
                sesion = _nueva_sesion()
                modelo_id = _insertar_en_bd(sesion, tarea)
                exitosas.append(i)
                tipo = "mejor modelo" if tarea.modelo_bytes else "individuo"
                print(f"[reintento BD] OK ({tipo}) — experimento_id={tarea.experimento_id} modelo_id={modelo_id}")
            except Exception as e:
                tipo = "mejor modelo" if tarea.modelo_bytes else "individuo"
                print(f"[reintento BD] Error ({tipo}) experimento_id={tarea.experimento_id} — reintentando en 60s: {e}")
                if sesion:
                    try:
                        sesion.rollback()
                    except Exception:
                        pass
            finally:
                if sesion:
                    try:
                        sesion.close()
                    except Exception:
                        pass

        for i in sorted(exitosas, reverse=True):
            pendientes.pop(i)


def _iniciar_worker():
    global _worker_iniciado
    with _worker_lock:
        if not _worker_iniciado:
            t = threading.Thread(target=_worker_reintentos, daemon=True, name="bd-retry-worker")
            t.start()
            _worker_iniciado = True


def _encolar(tarea: _TareaGuardado):
    _cola_reintentos.put(tarea)
    _iniciar_worker()
    tipo = "mejor modelo" if tarea.modelo_bytes else "individuo"
    print(f"[guardar BD] Encolado para reintento ({tipo}) — experimento_id={tarea.experimento_id}")


# ─── API pública de persistencia ─────────────────────────────────────────────

def crear_experimento(sesion, nombre: str) -> int:
    Experimento, _, _, _ = _import_bd_models()
    experimento = Experimento(nombre=nombre, fecha=datetime.utcnow())
    sesion.add(experimento)
    sesion.commit()
    sesion.refresh(experimento)
    return experimento.id


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generar_diagnostico_imagenes(model, history, val_loader, num_classes: int) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score, recall_score

    hist = history
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, ax = plt.subplots()
    ax.plot(epochs, hist["loss"], label="Entrenamiento")
    ax.plot(epochs, hist["val_loss"], label="Validación")
    ax.set_title("Curva de Pérdida")
    ax.set_xlabel("Época")
    ax.set_ylabel("Pérdida")
    ax.legend()
    plt.tight_layout()
    curva_perdida = _fig_to_bytes(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs, hist["accuracy"], label="Entrenamiento")
    ax.plot(epochs, hist["val_accuracy"], label="Validación")
    ax.set_title("Curva de Precisión")
    ax.set_xlabel("Época")
    ax.set_ylabel("Precisión")
    ax.legend()
    plt.tight_layout()
    curva_precision = _fig_to_bytes(fig)

    device = next(model.parameters()).device
    model.eval()
    y_pred_list, y_true_list = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            y_pred_list.extend(out.argmax(1).cpu().tolist())
            if isinstance(labels, torch.Tensor):
                y_true_list.extend(labels.tolist())
            else:
                y_true_list.extend(labels)
    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_title("Matriz de Confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.tight_layout()
    matriz_confusion_bytes = _fig_to_bytes(fig)

    labels_range = list(range(num_classes))
    recall_vals = recall_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    f1_vals = f1_score(y_true, y_pred, labels=labels_range, average=None, zero_division=0)
    x = np.arange(num_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, num_classes * 0.8), 5))
    ax.bar(x - width / 2, recall_vals, width, label="Recall")
    ax.bar(x + width / 2, f1_vals, width, label="F1-Score")
    ax.set_title("Recall y F1-Score por Clase")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Clase {i}" for i in range(num_classes)], rotation=45)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    curva_recall_f1 = _fig_to_bytes(fig)

    return {
        "curva_perdida": curva_perdida,
        "curva_precision": curva_precision,
        "matriz_confusion": matriz_confusion_bytes,
        "curva_recall_f1": curva_recall_f1,
    }


def guardar_diagnostico_bd(modelo_id: int, model, history, val_loader, num_classes: int) -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from ManejoDeDatos.basededatos import guardar_diagnostico

    diags = generar_diagnostico_imagenes(model, history, val_loader, num_classes)
    sesion = _nueva_sesion()
    try:
        guardar_diagnostico(sesion, modelo_id, **diags)
    finally:
        sesion.close()


def serializar_modelo(model: nn.Module) -> Optional[bytes]:
    """Serializa un modelo PyTorch a bytes .pt. Retorna None si falla."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(model, tmp_path)
            size = os.path.getsize(tmp_path)
            if size == 0:
                print("[serializar_modelo] Error: el archivo guardado está vacío")
                return None
            print(f"[serializar_modelo] Archivo temporal: {size / 1024:.1f} KB en {tmp_path}")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        print(f"[serializar_modelo] Error al guardar modelo: {e}")
        return None


def guardar_individuo_bd(sesion, experimento_id: int, individuo: dict, history,
                         arquitectura: str = "CNN",
                         modelo_bytes: Optional[bytes] = None) -> int:
    hist = history if isinstance(history, dict) else {}
    tarea = _TareaGuardado(
        experimento_id=experimento_id,
        individuo=dict(individuo),
        historia=dict(hist),
        arquitectura=arquitectura,
        modelo_bytes=modelo_bytes,
    )
    sesion_nueva = None
    try:
        sesion_nueva = _nueva_sesion()
        resultado = _insertar_en_bd(sesion_nueva, tarea)
        return resultado
    except Exception as e:
        print(f"[guardar_individuo_bd] Error: {e}")
        try:
            if sesion_nueva:
                sesion_nueva.rollback()
        except Exception:
            pass
        _encolar(tarea)
        return 0
    finally:
        try:
            if sesion_nueva:
                sesion_nueva.close()
        except Exception:
            pass


def _safe_get(hist: dict, key: str, idx: int):
    vals = hist.get(key, [])
    return float(vals[idx]) if idx < len(vals) else None


# ─── Dataset con augmentación ─────────────────────────────────────────────────

class _AugDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self._ds = base_dataset
        self._tf = transform

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        img, label = self._ds[idx]
        return self._tf(img), label


# ─── Preparación de DataLoaders ───────────────────────────────────────────────

def preparar_datasets(individuo: dict, train_dataset, val_dataset):
    """
    Aplica batch_size y use_augmentation del individuo sobre los datasets crudos
    y retorna (train_loader, val_loader, steps_per_epoch).
    """
    batch_size = individuo.get("batch_size", 32)
    use_aug = individuo.get("use_augmentation", True)

    if use_aug:
        augment = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=36, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        train_ds_final = _AugDataset(train_dataset, augment)
    else:
        train_ds_final = train_dataset

    train_loader = DataLoader(
        train_ds_final, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    steps_per_epoch = len(train_loader)
    return train_loader, val_loader, steps_per_epoch


# ─── Evaluación ──────────────────────────────────────────────────────────────

def evaluate_individual(individuo, train_dataset, val_dataset, num_classes, experimento_id, sesion) -> float:
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_loader, val_loader, _ = preparar_datasets(individuo, train_dataset, val_dataset)
        model, optimizer, criterion = build_model(individuo, num_classes, device)
        history = entrenar_modelo(model, train_loader, val_loader, optimizer, criterion,
                                  individuo["epochs"], device)
        try:
            modelo_id = guardar_individuo_bd(sesion, experimento_id, individuo, history)
            if modelo_id:
                try:
                    guardar_diagnostico_bd(modelo_id, model, history, val_loader, num_classes)
                except Exception as diag_err:
                    print(f"[evaluate_individual] Advertencia diagnóstico: {diag_err}")
        except Exception as bd_err:
            print(f"[evaluate_individual] Advertencia BD: {bd_err}")
        return float(history["val_accuracy"][-1])
    except Exception as err:
        print(f"[evaluate_individual] Error: {err}")
        return 0.0


# ─── Algoritmo Genético ───────────────────────────────────────────────────────

class GeneticAlgorithm:

    def __init__(self, search_space, poblacion_size, num_generaciones,
                 prob_mutacion, prob_cruce, seed=42):
        self.search_space = search_space
        self.poblacion_size = poblacion_size
        self.num_generaciones = num_generaciones
        self.prob_mutacion = prob_mutacion
        self.prob_cruce = prob_cruce
        self.seed = seed
        self._rng = random.Random(seed)

    def inicializar_poblacion(self):
        rng = random.Random(self.seed)
        return [
            {key: rng.choice(vals) for key, vals in self.search_space.items()}
            for _ in range(self.poblacion_size)
        ]

    def evaluar_poblacion(self, poblacion, train_dataset, val_dataset, num_classes, experimento_id, sesion):
        return [
            evaluate_individual(ind, train_dataset, val_dataset, num_classes, experimento_id, sesion)
            for ind in poblacion
        ]

    def seleccion_torneo(self, poblacion, fitness_list, k=2):
        indices = self._rng.sample(range(len(poblacion)), min(k, len(poblacion)))
        best_idx = max(indices, key=lambda i: fitness_list[i])
        return dict(poblacion[best_idx])

    def cruce_uniforme(self, padre1, padre2):
        hijo1, hijo2 = {}, {}
        for key in padre1:
            if self._rng.random() < self.prob_cruce:
                hijo1[key] = padre1[key]; hijo2[key] = padre2[key]
            else:
                hijo1[key] = padre2[key]; hijo2[key] = padre1[key]
        return hijo1, hijo2

    def mutacion(self, individuo):
        return {
            key: (self._rng.choice(self.search_space[key])
                  if self._rng.random() < self.prob_mutacion else val)
            for key, val in individuo.items()
        }

    def run(self, train_dataset, val_dataset, num_classes, experimento_id, sesion):
        poblacion = self.inicializar_poblacion()
        fitness_list = self.evaluar_poblacion(
            poblacion, train_dataset, val_dataset, num_classes, experimento_id, sesion
        )

        mejor_idx = int(np.argmax(fitness_list))
        mejor_individuo = dict(poblacion[mejor_idx])
        mejor_fitness = fitness_list[mejor_idx]
        fitness_history = [mejor_fitness]

        for _ in range(self.num_generaciones - 1):
            nueva_poblacion = [dict(mejor_individuo)]
            while len(nueva_poblacion) < self.poblacion_size:
                p1 = self.seleccion_torneo(poblacion, fitness_list)
                p2 = self.seleccion_torneo(poblacion, fitness_list)
                h1, h2 = self.cruce_uniforme(p1, p2)
                nueva_poblacion.append(self.mutacion(h1))
                if len(nueva_poblacion) < self.poblacion_size:
                    nueva_poblacion.append(self.mutacion(h2))

            poblacion = nueva_poblacion[:self.poblacion_size]
            fitness_list = self.evaluar_poblacion(
                poblacion, train_dataset, val_dataset, num_classes, experimento_id, sesion
            )

            gen_mejor_idx = int(np.argmax(fitness_list))
            if fitness_list[gen_mejor_idx] > mejor_fitness:
                mejor_fitness = fitness_list[gen_mejor_idx]
                mejor_individuo = dict(poblacion[gen_mejor_idx])
            fitness_history.append(mejor_fitness)

        return mejor_individuo, fitness_history


# ─── PSO ─────────────────────────────────────────────────────────────────────

class ParticleSwarmOptimizer:

    def __init__(self, search_space, poblacion_size, num_generaciones,
                 w_inercia=0.5, c1_cognitivo=1.5, c2_social=1.5, seed=42):
        self.search_space = search_space
        self.poblacion_size = poblacion_size
        self.num_generaciones = num_generaciones
        self.w_inercia = w_inercia
        self.c1_cognitivo = c1_cognitivo
        self.c2_social = c2_social
        self.seed = seed
        self._rng = random.Random(seed)

    def inicializar_enjambre(self):
        rng = random.Random(self.seed)
        enjambre = []
        for _ in range(self.poblacion_size):
            pos = {key: rng.choice(vals) for key, vals in self.search_space.items()}
            vel = {key: rng.uniform(-1.0, 1.0) for key in self.search_space}
            enjambre.append({"pos": dict(pos), "vel": dict(vel),
                             "pbest_pos": dict(pos), "pbest_fit": 0.0})
        return enjambre

    def discretizar_posicion(self, pos_continua):
        discreto = {}
        for key, val in pos_continua.items():
            opciones = self.search_space[key]
            try:
                discreto[key] = min(opciones, key=lambda v: abs(float(v) - float(val)))
            except (TypeError, ValueError):
                discreto[key] = opciones[0]
        return discreto

    def actualizar_velocidad(self, particula, gbest_pos):
        nueva_vel = {}
        for key in self.search_space:
            v = particula["vel"][key]
            try:
                pos   = float(particula["pos"][key])
                pbest = float(particula["pbest_pos"][key])
                gbest = float(gbest_pos[key])
                r1, r2 = self._rng.random(), self._rng.random()
                nueva_vel[key] = (self.w_inercia * v
                                  + self.c1_cognitivo * r1 * (pbest - pos)
                                  + self.c2_social   * r2 * (gbest - pos))
            except (TypeError, ValueError):
                nueva_vel[key] = 0.0
        return nueva_vel

    def actualizar_posicion(self, particula):
        nueva_pos_continua = {}
        for key in self.search_space:
            try:
                nueva_pos_continua[key] = float(particula["pos"][key]) + particula["vel"][key]
            except (TypeError, ValueError):
                nueva_pos_continua[key] = particula["pos"][key]
        return self.discretizar_posicion(nueva_pos_continua)

    def run(self, train_dataset, val_dataset, num_classes, experimento_id, sesion):
        enjambre = self.inicializar_enjambre()

        gbest_pos = None
        gbest_fit = -1.0
        for particula in enjambre:
            fit = evaluate_individual(
                particula["pos"], train_dataset, val_dataset, num_classes, experimento_id, sesion
            )
            particula["pbest_fit"] = fit
            if fit > gbest_fit:
                gbest_fit = fit
                gbest_pos = dict(particula["pos"])

        fitness_history = [gbest_fit]

        for _ in range(self.num_generaciones - 1):
            for particula in enjambre:
                particula["vel"] = self.actualizar_velocidad(particula, gbest_pos)
                nueva_pos = self.actualizar_posicion(particula)
                particula["pos"] = nueva_pos

                fit = evaluate_individual(
                    nueva_pos, train_dataset, val_dataset, num_classes, experimento_id, sesion
                )
                if fit > particula["pbest_fit"]:
                    particula["pbest_fit"] = fit
                    particula["pbest_pos"] = dict(nueva_pos)
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_pos = dict(nueva_pos)

            fitness_history.append(gbest_fit)

        return gbest_pos, fitness_history


# ─── Visualización ───────────────────────────────────────────────────────────

def plot_fitness_history(fitness_history: list, algoritmo: str) -> None:
    generaciones = list(range(1, len(fitness_history) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(generaciones, fitness_history, marker="o", linewidth=2, color="steelblue")
    plt.xlabel("Generación / Iteración")
    plt.ylabel("Fitness máximo (val_accuracy)")
    plt.title(f"Evolución del Fitness — {algoritmo}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history) -> None:
    hist = history if isinstance(history, dict) else {}
    epochs = range(1, len(hist.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, hist.get("accuracy", []),     label="Entrenamiento", color="steelblue")
    ax1.plot(epochs, hist.get("val_accuracy", []), label="Validación",    color="orange")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy por época"); ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.6)

    ax2.plot(epochs, hist.get("loss", []),     label="Entrenamiento", color="steelblue")
    ax2.plot(epochs, hist.get("val_loss", []), label="Validación",    color="orange")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Loss")
    ax2.set_title("Loss por época"); ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, val_loader, class_names: list) -> None:
    if len(class_names) != 6:
        raise ValueError(f"Se esperaban 6 clases, se detectaron {len(class_names)}.")

    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            y_pred.extend(out.argmax(1).cpu().tolist())
            if isinstance(labels, torch.Tensor):
                y_true.extend(labels.tolist())
            else:
                y_true.extend(labels)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicción"); plt.ylabel("Real")
    plt.title("Matriz de Confusión — Mejor Individuo")
    plt.tight_layout()
    plt.show()
