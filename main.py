from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
from utils import clean_tweet, preprocess_text

# Importer les outils OpenTelemetry
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter

# Configurer OpenTelemetry avec un nom de service
resource = Resource(attributes={
    "service.name": "sentiment-analysis-app"
})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer_provider = trace.get_tracer_provider()

# Obtenir une instance de tracer
tracer = trace.get_tracer(__name__)

# Configurer l'exportateur pour Azure Monitor
connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if not connection_string:
    raise ValueError("La variable d'environnement 'APPLICATIONINSIGHTS_CONNECTION_STRING' est manquante.")

# Désactiver Azure Monitor en mode test
if os.getenv("TEST_ENV") != "1":
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not connection_string:
        raise ValueError("La variable d'environnement 'APPLICATIONINSIGHTS_CONNECTION_STRING' est manquante.")

    exporter = AzureMonitorTraceExporter.from_connection_string(connection_string)
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)
else:
    # Ajout d'un exporteur fictif pour les tests
    print("Azure Monitor désactivé pour les tests.")

# Configurer le logger
logger = logging.getLogger("sentiment_analysis_app")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Initialiser l'application FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.4")

# Instrumenter FastAPI pour OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

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
        # Ajout d'une trace OpenTelemetry pour le texte vide
        with tracer.start_as_current_span("EmptyTextError"):
            span = trace.get_current_span()
            span.set_attribute("error.type", "ValidationError")
            span.set_attribute("error.message", "Texte vide reçu dans la requête")
        raise HTTPException(status_code=400, detail="Le texte fourni est vide.")

    try:
        # Étape 1 : Nettoyage du tweet
        tweet = clean_tweet(request.text)

        # Étape 2 : Prétraitement du tweet
        tweet = preprocess_text(tweet)

        if not tweet.strip():
            logger.warning("Le texte est vide après le nettoyage et le prétraitement.")
            # Ajout d'une trace OpenTelemetry pour le prétraitement échoué
            with tracer.start_as_current_span("PreprocessingError"):
                span = trace.get_current_span()
                span.set_attribute("error.type", "PreprocessingError")
                span.set_attribute("error.message", "Le texte nettoyé est vide après prétraitement.")
            raise HTTPException(
                status_code=400, 
                detail="Le texte nettoyé est vide après prétraitement."
            )

        # Étape 3 : Vectorisation
        text_vectorized = vectorizer.transform([tweet])

        # Étape 4 : Prédiction
        prediction = model.predict(text_vectorized)
        probabilities = model.predict_proba(text_vectorized)

        # Décoder la prédiction
        sentiment_label = "positive" if prediction[0] == 1 else "negative"
        confidence = probabilities[0][prediction[0]]

        logger.info(f"Prédiction terminée. Sentiment : {sentiment_label}, Confiance : {confidence:.2f}")

        # Ajout d'une trace pour une prédiction correcte
        with tracer.start_as_current_span("PredictionSuccess"):
            span = trace.get_current_span()
            span.set_attribute("prediction.sentiment", sentiment_label)
            span.set_attribute("prediction.confidence", confidence)
        return {"sentiment": sentiment_label, "confidence": f"{confidence:.2f}"}

    except HTTPException as e:
        # Laisser passer les erreurs HTTP intentionnelles
        logger.error(f"Erreur HTTP intentionnelle : {e.detail}")
        with tracer.start_as_current_span("HTTPException"):
            span = trace.get_current_span()
            span.set_attribute("error.type", "HTTPException")
            span.set_attribute("error.message", e.detail)
        raise e

    except Exception as e:
        # Capturer les erreurs inattendues
        logger.error(f"Erreur interne lors de la prédiction : {e}")
        with tracer.start_as_current_span("InternalServerError"):
            span = trace.get_current_span()
            span.set_attribute("error.type", "InternalServerError")
            span.set_attribute("error.message", str(e))
        raise HTTPException(status_code=500, detail="Erreur interne lors de la prédiction.")
