from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
from utils import clean_tweet, preprocess_text

# Récupérer la clé d'instrumentation à partir des variables d'environnement
INSTRUMENTATION_KEY = os.getenv("INSTRUMENTATION_KEY")

# Configurer Application Insights
logger = logging.getLogger("sentiment_analysis_app")
logger.setLevel(logging.INFO)

# Détecter si les tests sont en cours d'exécution
IS_TESTING = os.getenv("TEST_ENV", False)

# Configurer le logger pour afficher les messages localement aussi
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Ajouter un gestionnaire pour Application Insights
if not IS_TESTING:
    if not INSTRUMENTATION_KEY:
        raise ValueError("INSTRUMENTATION_KEY est manquant dans les variables d'environnement.")
    azure_handler = AzureLogHandler(connection_string=INSTRUMENTATION_KEY)
    logger.addHandler(azure_handler)

    # Log pour tester la connexion avec Application Insights
    logger.info("Test de connexion Application Insights.")

# Initialiser l'application FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.2")

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
    logger.info("Requête reçue sur la route racine.")
    return {"message": "Bienvenue dans l'API d'analyse de sentiment !"}

@app.post("/predict")
def predict(request: SentimentRequest):
    if not request.text.strip():
        logger.warning("Texte vide reçu dans la requête.")
        raise HTTPException(status_code=400, detail="Le texte fourni est vide.")

    try:
        # Étape 1 : Nettoyage du tweet
        logger.info("Nettoyage du tweet...")
        tweet = clean_tweet(request.text)
        logger.info(f"Tweet nettoyé : {tweet}")

        # Étape 2 : Prétraitement du tweet
        logger.info("Prétraitement du tweet...")
        tweet = preprocess_text(tweet)
        logger.info(f"Tweet prétraité : {tweet}")

        if not tweet.strip():
            logger.warning("Le texte est vide après le nettoyage et le prétraitement.")
            raise HTTPException(
                status_code=400, 
                detail="Le texte nettoyé est vide après prétraitement."
            )

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

        logger.info(f"Prédiction terminée. Sentiment : {sentiment_label}, Confiance : {confidence:.2f}")

        return {"sentiment": sentiment_label, "confidence": f"{confidence:.2f}"}

    except HTTPException as e:
        # Laisser passer les erreurs HTTP intentionnelles
        logger.error(f"Erreur HTTP intentionnelle : {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Erreur interne lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")
