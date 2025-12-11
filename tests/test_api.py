import os
from fastapi.testclient import TestClient
from joblib import load

import sys
# Para importar correctamente src.main
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from main import app  # Importa la API

client = TestClient(app)


def test_health_endpoint():
    """Prueba el endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] == True


def test_predict_endpoint():
    """Prueba /predict enviando datos válidos."""
    # Request mínimo: solo unas pocas columnas; las demás se llenan con 0.
    payload = {
        "features": {
            "freq_180": 5,
            "spend_total": 10000,
            "recency_days": 20
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert "proba_churn" in body
    assert "is_top10_risk" in body
    assert 0 <= body["proba_churn"] <= 1


def test_predict_missing_features():
    """Prueba comportamiento cuando request viene vacío o incorrecto."""
    payload = {"features": {}}

    response = client.post("/predict", json=payload)
    assert response.status_code == 400  # Debe fallar correctamente
