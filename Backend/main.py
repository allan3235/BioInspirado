from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

# ==========================
# CORS
# ==========================

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Clases del modelo
# ==========================

CLASES = {
    0: "Caries",
    1: "Gingivitis",
    2: "Sano"
}

# ==========================
# Transformaciones
# ==========================

transformacion = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ==========================
# Cargar modelo
# ==========================

try:

    ckpt = torch.load(
        "oral_disease_model.pth",
        map_location="cpu"
    )

    print("TIPO:")
    print(type(ckpt))

    if isinstance(ckpt, dict):
        print("KEYS:")
        print(ckpt.keys())

    modelo = None

except Exception as e:

    print(f"Error cargando modelo: {e}")

    modelo = None

# ==========================
# Endpoint de predicción
# ==========================

@app.post("/predecir/")
async def predecir(file: UploadFile = File(...)):

    if modelo is None:
        raise HTTPException(
            status_code=500,
            detail="El modelo no fue cargado correctamente."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo enviado no es una imagen."
        )

    try:

        # Leer imagen enviada
        contenido = await file.read()

        imagen = Image.open(
            io.BytesIO(contenido)
        ).convert("RGB")

        # Aplicar transformaciones
        imagen_tensor = transformacion(imagen)

        # Agregar dimensión batch
        imagen_batch = imagen_tensor.unsqueeze(0)

        # Inferencia
        with torch.no_grad():

            salida = modelo(imagen_batch)

            probabilidades = torch.softmax(
                salida,
                dim=1
            )

            clase_predicha = torch.argmax(
                probabilidades,
                dim=1
            ).item()

            confianza = probabilidades[
                0,
                clase_predicha
            ].item()

        return {
            "filename": file.filename,
            "diagnostico": CLASES[clase_predicha],
            "confianza": round(confianza * 100, 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )