from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
from utils import clean_tweet, preprocess_text
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace

# Configurer Azure Monitor pour OpenTelemetry
configure_azure_monitor()

# Obtenir un tracer pour générer des spans
tracer = trace.get_tracer(__name__)

# Configurer le logger
logger = logging.getLogger("sentiment_analysis_app")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Définir le modèle de requête pour /log_trace
class TraceRequest(BaseModel):
    text: str
    predicted_sentiment: str
    confidence: str

# Initialiser l'application FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.5")

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
        logger.warning("Texte vide reçu dans la requête.")
        with tracer.start_as_current_span("EmptyTextError") as span:
            span = trace.get_current_span()
            span.set_attribute("error.type", "ValidationError")
            span.set_attribute("error.message", "Texte vide reçu dans la requête")
            span.set_attribute("error.details", "Aucun texte fourni par l'utilisateur.")
        raise HTTPException(status_code=400, detail="Le texte fourni est vide.")

    try:
        # Étape 1 : Nettoyage du tweet
        tweet = clean_tweet(request.text)

        # Étape 2 : Prétraitement du tweet
        tweet = preprocess_text(tweet)

        if not tweet.strip():
            logger.warning("Le texte est vide après le nettoyage et le prétraitement.")
            with tracer.start_as_current_span("PreprocessingError") as span:
                span = trace.get_current_span()
                span.set_attribute("error.type", "PreprocessingError")
                span.set_attribute("error.message", "Le texte nettoyé est vide après prétraitement.")
                span.set_attribute("error.details", f"Texte original : {request.text}")
            raise HTTPException(status_code=400, detail="Le texte nettoyé est vide après prétraitement.")

        # Étape 3 : Vectorisation
        text_vectorized = vectorizer.transform([tweet])

        # Étape 4 : Prédiction
        prediction = model.predict(text_vectorized)
        probabilities = model.predict_proba(text_vectorized)

        # Décoder la prédiction
        sentiment_label = "positive" if prediction[0] == 1 else "negative"
        confidence = probabilities[0][prediction[0]]

        logger.info(f"Prédiction terminée. Sentiment : {sentiment_label}, Confiance : {confidence:.2f}")

        return {"sentiment": sentiment_label, "confidence": f"{confidence:.2f}"}

    except HTTPException as e:
        logger.error(f"Erreur HTTP intentionnelle : {e.detail}")
        raise e

    except Exception as e:
        logger.error(f"Erreur interne lors de la prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")

@app.post("/log_trace")
def log_trace(request: TraceRequest):
    """
    Endpoint pour enregistrer une trace en cas de prédiction incorrecte.
    """
    try:
        with tracer.start_as_current_span("PredictionErrorTrace") as span:
            # Ajouter les attributs au span
            span.set_attribute("event.type", "prediction_incorrect")
            span.set_attribute("text", request.text)
            span.set_attribute("predicted_sentiment", request.predicted_sentiment)
            span.set_attribute("confidence", request.confidence)
            span.set_attribute("message", "Prédiction signalée comme incorrecte par l'utilisateur.")

            # Log dans la console pour confirmation
            logger.warning(f"Prédiction incorrecte signalée : {request.text} "
                           f"(Sentiment : {request.predicted_sentiment}, Confiance : {request.confidence})")

        return {"message": "Trace enregistrée avec succès dans Azure Application Insight"}

    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement de la trace : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne lors de l'enregistrement de la trace.")