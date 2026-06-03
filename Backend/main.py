from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_small
from PIL import Image
import io
from pathlib import Path

BASE_DIR = Path(__file__).parent

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
    0: "Calculus",
    1: "Caries",
    2: "Gingivitis",
    3: "Hypodontia",
    4: "Tooth Discoloration",
    5: "Ulcers"
}

# ==========================
# Transformaciones
# ==========================

transformacion = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ==========================
# Arquitectura del modelo
# ==========================

class OralDiseaseModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = convnext_small()
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ==========================
# Cargar modelo
# ==========================

try:
    ckpt = torch.load(
        BASE_DIR / "oral_disease_model.pth",
        map_location="cpu",
        weights_only=False
    )

    num_classes = ckpt.get("num_classes", 6)

    modelo = OralDiseaseModel(num_classes=num_classes)
    modelo.load_state_dict(ckpt["model_state_dict"])
    modelo.eval()

    print(f"Modelo cargado correctamente ({num_classes} clases)")

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

        contenido = await file.read()

        imagen = Image.open(
            io.BytesIO(contenido)
        ).convert("RGB")

        imagen_tensor = transformacion(imagen)
        imagen_batch = imagen_tensor.unsqueeze(0)

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

        probabilidades_por_clase = {
            CLASES[i]: round(probabilidades[0, i].item() * 100, 2)
            for i in range(len(CLASES))
        }

        return {
            "filename": file.filename,
            "diagnostico": CLASES[clase_predicha],
            "confianza": round(confianza * 100, 2),
            "probabilidades": probabilidades_por_clase
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error procesando la imagen: {str(e)}"
        )
