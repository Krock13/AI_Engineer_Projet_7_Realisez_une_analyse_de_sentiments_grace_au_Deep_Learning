from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
from utils import clean_tweet, preprocess_text

# Initialiser l'application FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.1")

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le modèle et le vectorizer
model_path = "saved_models/logistic_regression_tf-idf.pkl"
vectorizer_path = "saved_models/tf-idf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    logger.error("Le modèle ou le vectorizer n'a pas été trouvé. Vérifiez les chemins spécifiés.")
    raise FileNotFoundError("Le modèle ou le vectorizer n'a pas été trouvé. Vérifiez les chemins spécifiés.")

logger.info("Chargement du modèle et du vectorizer...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
logger.info("Modèle et vectorizer chargés avec succès.")

# Définir le modèle de requête
class SentimentRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Bienvenue dans l'API d'analyse de sentiment !"}

@app.post("/predict")
def predict(request: SentimentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Le texte fourni est vide.")

    # Étape 1 : Nettoyage du tweet
    logger.info("Nettoyage du tweet...")
    tweet = clean_tweet(request.text)
    logger.info(f"Tweet nettoyé : {tweet}")

    # Étape 2 : Prétraitement du tweet
    logger.info("Prétraitement du tweet...")
    tweet = preprocess_text(tweet)
    logger.info(f"Tweet prétraité : {tweet}")

    if not tweet.strip():
        raise HTTPException(status_code=400, detail="Le texte nettoyé est vide après prétraitement.")

    # Étape 3 : Vectorisation
    logger.info("Vectorisation du tweet...")
    text_vectorized = vectorizer.transform([tweet])

    # Étape 4 : Prédiction
    logger.info("Prédiction du sentiment...")
    prediction = model.predict(text_vectorized)
    probabilities = model.predict_proba(text_vectorized)

    # Décoder la prédiction
    sentiment_label = "positive" if prediction[0] == 1 else "negative"
    confidence = probabilities[0][prediction[0]]

    logger.info(f"Sentiment : {sentiment_label}, Confiance : {confidence:.2f}")

    return {"sentiment": sentiment_label, "confidence": f"{confidence:.2f}"}
