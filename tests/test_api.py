import sys
import os
from unittest.mock import patch
from fastapi.testclient import TestClient

# ============================
# Préconfiguration pour les tests
# ============================

# Définir les variables d'environnement avant d'importer l'application
os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"] = "InstrumentationKey=00000000-0000-0000-0000-000000000000"
os.environ["TEST_ENV"] = "1"  # Indique que nous sommes en mode test

# Ajouter le répertoire racine au chemin système pour permettre les imports locaux
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

# ============================
# Importer l'application et configurer le client de test
# ============================

from main import app  # Import de l'application FastAPI locale

# Initialiser le client de test pour interagir avec l'application
client = TestClient(app)

# ================================
# Tests des routes principales
# ================================

def test_root():
    """Test de la route racine (GET /)."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API d'analyse de sentiment !"}

# ==========================
# Tests de prédictions
# ==========================

def test_predict_positive():
    """Test pour un texte à sentiment positif."""
    payload = {"text": "I absolutely love this project!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

def test_predict_negative():
    """Test pour un texte à sentiment négatif."""
    payload = {"text": "I absolutely hate this project!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

# ==========================
# Tests d'erreurs et cas limites
# ==========================

def test_empty_text():
    """Test pour une entrée vide."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Le texte fourni est vide."

def test_special_characters():
    """Test pour un texte contenant uniquement des caractères spéciaux."""
    payload = {"text": "!@#$%^&*()"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Le texte nettoyé est vide après prétraitement."

def test_long_text():
    """Test pour un texte très long."""
    long_text = "I love this! " * 1000  # Répéter une phrase positive
    payload = {"text": long_text}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

def test_uppercase_text():
    """Test pour un texte entièrement en majuscules."""
    payload = {"text": "I LOVE THIS!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

def test_mixed_case_text():
    """Test pour un texte avec un mélange de majuscules et minuscules."""
    payload = {"text": "I LoVe ThiS ProJect!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

def test_text_with_emojis():
    """Test pour un texte contenant des emojis."""
    payload = {"text": "I love this project! 😍✨"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

# ==========================
# Tests supplémentaires
# ==========================

@patch("main.model.predict", side_effect=Exception("Erreur interne"))
def test_model_error(mock_predict):
    """Test pour simuler une erreur interne dans le modèle."""
    payload = {"text": "I love this project!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "Erreur interne lors de la prédiction."

def test_text_cleaning():
    """Test pour vérifier si le texte est correctement nettoyé avant la prédiction."""
    payload = {"text": "@user http://example.com I LOVE THIS!!!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert "confidence" in response.json()
    assert 0.0 <= float(response.json()["confidence"]) <= 1.0

def test_foreign_language():
    """Test pour un texte dans une langue étrangère."""
    payload = {"text": "Je déteste ce projet."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "sentiment" in response.json()
    assert "confidence" in response.json()

