import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from main import app
from fastapi.testclient import TestClient

client = TestClient(app)

# Test de la route principale
def test_root():
    """Test de la route racine (GET /)."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API d'analyse de sentiment !"}

# Test de prédiction positive
def test_predict_positive():
    """Test pour un texte à sentiment positif."""
    payload = {"text": "I absolutely love this project!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

# Test de prédiction négative
def test_predict_negative():
    """Test pour un texte à sentiment négatif."""
    payload = {"text": "I absolutely hate this project!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

# Test pour une entrée vide
def test_empty_text():
    """Test pour une entrée vide."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Le texte fourni est vide."

# Test pour une entrée avec caractères spéciaux
def test_special_characters():
    """Test pour un texte contenant uniquement des caractères spéciaux."""
    payload = {"text": "!@#$%^&*()"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Le texte nettoyé est vide après prétraitement."

# Test pour un texte très long
def test_long_text():
    """Test pour un texte très long."""
    long_text = "I love this! " * 1000  # Répéter une phrase positive
    payload = {"text": long_text}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

# Test pour un texte avec des emojis
def test_text_with_emojis():
    """Test pour un texte contenant des emojis."""
    payload = {"text": "I love this project! 😍✨"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

# Test pour vérifier le nettoyage
def test_text_cleaning():
    """Test pour vérifier si le texte est correctement nettoyé avant la prédiction."""
    payload = {"text": "@user http://example.com I LOVE THIS!!!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0
