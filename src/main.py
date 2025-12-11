import os
from typing import Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load

# ------------------------------------------------------------
# Configuración de rutas
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_churn_rf_v1.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}. "
                            f"Ejecuta primero 'python src/train_model.py'.")

# ------------------------------------------------------------
# Carga del artefacto de modelo
# ------------------------------------------------------------
artifact = load(MODEL_PATH)
model = artifact["model"]
feature_cols = artifact["feature_cols"]
threshold_top10 = artifact["threshold_top10"]

# ------------------------------------------------------------
# Definición de la API
# ------------------------------------------------------------
app = FastAPI(
    title="API Churn Emporio Vinos y Licores",
    description=(
        "Servicio de inferencia para predecir probabilidad de no recompra (churn 30 días).\n"
        "El endpoint /predict recibe un diccionario de features y devuelve la probabilidad de churn "
        "y si el cliente está en el Top 10% de riesgo."
    ),
    version="1.0.0",
)


class FeaturesPayload(BaseModel):
    """
    Payload de entrada para /predict.

    Ejemplo:
    {
      \"features\": {
        \"freq_180\": 5,
        \"spend_total\": 120000,
        \"spend_avg\": 25000,
        \"coupon_rate\": 0.3,
        \"recency_days\": 15,
        \"tenure_days\": 200,
        \"arpu\": 60000,
        \"share_Vino\": 0.7,
        \"share_Cerveza\": 0.2,
        \"share_Destilado\": 0.1,
        \"region_Centro\": 1,
        \"region_Sur\": 0,
        \"canal_Web\": 1,
        \"canal_WhatsApp\": 0
      }
    }
    """
    features: Dict[str, float]


@app.get("/health", tags=["status"])
def health_check() -> Dict[str, Any]:
    """
    Endpoint simple para verificar que el servicio está arriba.
    """
    return {
        "status": "ok",
        "model_loaded": True,
        "n_features_expected": len(feature_cols),
        "threshold_top10": threshold_top10,
    }


@app.post("/predict", tags=["inferencia"])
def predict(payload: FeaturesPayload) -> Dict[str, Any]:
    """
    Realiza la predicción de churn para un cliente.

    - Recibe un diccionario de features (nombre_columna -> valor).
    - Ajusta el orden de columnas según feature_cols.
    - Completa con 0 las columnas faltantes.
    - Devuelve:
        - probabilidad de churn (entre 0 y 1)
        - flag si está en el Top 10% de riesgo
    """
    # Convertir a DataFrame de una fila
    input_features = payload.features

    if not input_features:
        raise HTTPException(status_code=400, detail="El diccionario 'features' no puede venir vacío.")

    # DataFrame con una sola fila
    df_input = pd.DataFrame([input_features])

    # Asegurar el orden y presencia de todas las columnas esperadas
    # Las columnas faltantes se rellenan con 0
    df_input_aligned = df_input.reindex(columns=feature_cols, fill_value=0)

    # Verificación básica: que al menos alguna columna de las esperadas tenga valor distinto de 0
    if (df_input_aligned.sum(axis=1) == 0).all():
        raise HTTPException(
            status_code=400,
            detail=(
                "Todas las features esperadas son 0. "
                "Revisa que los nombres de las columnas coincidan con las del modelo."
            ),
        )

    # Predicción de probabilidad de churn
    proba_churn = float(model.predict_proba(df_input_aligned)[0, 1])
    is_top10 = bool(proba_churn >= threshold_top10)

    return {
        "proba_churn": proba_churn,
        "is_top10_risk": is_top10,
        "threshold_top10": threshold_top10,
        "model_info": {
            "type": type(model).__name__,
            "best_model_name": getattr(model, "__class__", type(model)).__name__,
        },
        "input_used_features": list(df_input_aligned.columns),
    }
