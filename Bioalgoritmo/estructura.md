# Dataset: Oral Diseases

## Fuente
**Kaggle:** `salmansajid05/oral-diseases`

Conjunto de imagenes de enfermedades bucales para clasificacion con CNN y algoritmos bioinspirados.

---

## Clases (6 enfermedades)

| Clase | Fuente de datos |
|---|---|
| **Calculus** | Solo carpeta de clasificacion |
| **Gingivitis** | Solo carpeta de clasificacion |
| **Hypodontia** | Solo carpeta de clasificacion |
| **Caries** | Carpeta de clasificacion + crops YOLO |
| **Mouth Ulcer** | Carpeta de clasificacion + crops YOLO |
| **Tooth Discoloration** | Carpeta de clasificacion + crops YOLO |

---

## Estructura del raw dataset (antes del procesamiento)

```
dataset/oral-diseases/
├── Calculus/Calculus/                          <- imagenes de Calculus
├── Gingivitis/Gingivitis/                      <- imagenes de Gingivitis
├── hypodontia/hypodontia/                      <- imagenes de Hypodontia
├── Data caries/Data caries/                    <- imagenes de Caries
├── Mouth Ulcer/Mouth Ulcer/                    <- imagenes de Mouth Ulcer
├── Tooth Discoloration/Tooth Discoloration/    <- imagenes de Tooth Discoloration
└── Caries_Gingivitus_ToothDiscoloration_Ulcer-yolo_annotated-Dataset/
    └── .../Data/
        ├── images/
        │   ├── train/   <- imagenes YOLO
        │   └── val/     <- imagenes YOLO
        └── labels/
            ├── train/   <- anotaciones YOLO (.txt)
            └── val/     <- anotaciones YOLO (.txt)
```

---

## Estructura del flat dataset (listo para entrenar)

```
dataset/oral-diseases-flat/
├── train/
│   ├── Calculus/
│   ├── Caries/
│   ├── Gingivitis/
│   ├── Hypodontia/
│   ├── Mouth Ulcer/
│   └── Tooth Discoloration/
└── val/
    ├── Calculus/
    ├── Caries/
    ├── Gingivitis/
    ├── Hypodontia/
    ├── Mouth Ulcer/
    └── Tooth Discoloration/
```

---

## Pipeline de construccion

```
Raw dataset
    |
    +-- Clases grandes (Calculus, Gingivitis, Hypodontia)
    |       └── _collect_folder_images()
    |
    └── Clases pequeñas (Caries, Mouth Ulcer, Tooth Discoloration)
            +-- _collect_folder_images()   <- imagenes de la carpeta
            └── _generate_yolo_crops()     <- recortes de bounding boxes YOLO
                    |
                    └── Crops guardados temporalmente en _yolo_crops_tmp/
                            |
                            └── (eliminados al finalizar)
            |
            └── _shuffle_split_copy()      <- mezcla y divide 80/20
                    |
                    +-- train/{clase}/clase_00001.jpg
                    └── val/{clase}/clase_00001.jpg
```

---

## Anotaciones YOLO

El dataset YOLO contiene 4 clases. Solo se usan 3:

| YOLO class ID | Clase |
|---|---|
| `0` | Caries |
| `1` | Mouth Ulcer |
| `2` | Tooth Discoloration |
| ~~Gingivitis~~ | Ignorada (ya existe en carpeta, evita duplicados) |

Cada `.txt` tiene lineas con formato:
```
<class_id> <cx> <cy> <width> <height>   <- coordenadas normalizadas [0, 1]
```
El codigo recorta el bounding box de la imagen original con OpenCV y guarda cada crop como imagen independiente.

---

## Configuracion de entrenamiento

| Parametro | Valor |
|---|---|
| Tamaño de imagen | `224 x 224` px |
| Batch size | `32` |
| Split train/val | `80% / 20%` |
| Seed | `42` |
| Formatos aceptados | `.jpg`, `.jpeg`, `.png`, `.bmp` |

---

## Data Augmentation (solo en train)

| Transformacion | Valor |
|---|---|
| Rotacion | +-20 grados |
| Desplazamiento horizontal/vertical | 10% |
| Flip horizontal | Si |
| Zoom | +-10% |
| Normalizacion | `/255` |
