import os
from joblib import load
import pandas as pd
import numpy as np

# Ruta del modelo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "modelo_churn_rf_v1.joblib")


def test_model_file_exists():
    """Verifica que el modelo exista en /models."""
    assert os.path.exists(MODEL_PATH), f"No se encontró el modelo en {MODEL_PATH}"


def test_model_can_load():
    """Prueba que el modelo se pueda cargar sin errores."""
    artifact = load(MODEL_PATH)
    assert "model" in artifact
    assert "feature_cols" in artifact
    assert "threshold_top10" in artifact


def test_model_predict_proba():
    """Prueba que el modelo pueda predecir usando features dummy."""
    artifact = load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]

    # Crear una fila vacía (todas las columnas = 0)
    df_test = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)

    proba = model.predict_proba(df_test)[0, 1]

    assert 0 <= proba <= 1, "La probabilidad no está entre 0 y 1"
