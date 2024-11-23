import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API d'analyse de sentiment !"}

def test_predict_positive():
    response = client.post("/predict", json={"text": "I love this!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"

def test_predict_negative():
    response = client.post("/predict", json={"text": "I hate this!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"

def test_empty_text():
    """Test pour une entrée vide."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Le texte fourni est vide."
