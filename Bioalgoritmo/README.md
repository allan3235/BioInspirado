# CNN para Detección de Enfermedades Orales

Algoritmos bioinspirados aplicados a clasificación de imágenes médicas.

## Requisitos previos

- Python 3.10 o superior
- pip

## Configuración del entorno

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd BioInspirado/Bioalgoritmo
```

### 2. Crear el entorno virtual

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> Sabrás que el entorno está activo cuando veas `(.venv)` al inicio de tu terminal.

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar el dataset de Kaggle

Primero configura tu API key de Kaggle:

1. Ve a [kaggle.com](https://www.kaggle.com) → tu perfil → **Settings** → **Create New Token**
2. Descarga el archivo `kaggle.json` y colócalo en:
   - **Windows:** `C:\Users\TuUsuario\.kaggle\kaggle.json`
   - **Mac/Linux:** `~/.kaggle/kaggle.json`

Luego descarga el dataset:

```bash
kaggle datasets download -d salmansajid05/oral-diseases -p dataset/ --unzip
```

### 5. Abrir el notebook

```bash
jupyter notebook cnn_oral_diseases.ipynb
```

## Desactivar el entorno virtual

Cuando termines de trabajar:

```bash
deactivate
```
