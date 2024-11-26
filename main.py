from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from utils import clean_tweet, preprocess_text

# Initialiser l'application FastAPI
app = FastAPI(title="Sentiment Analysis API", version="1.0")

# Charger le modèle et le vectorizer
model_path = "saved_models/logistic_regression_tf-idf.pkl"
vectorizer_path = "saved_models/tf-idf_vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Le modèle ou le vectorizer n'a pas été trouvé. Vérifiez les chemins spécifiés.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

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

    tweet = request.text
    tweet = clean_tweet(tweet)
    tweet = preprocess_text(tweet)
    # Transformer le texte avec le vectorizer
    text_vectorized = vectorizer.transform([tweet])

    # Faire la prédiction
    prediction = model.predict(text_vectorized)
    probabilities = model.predict_proba(text_vectorized)

    # Décoder la prédiction
    sentiment_label = "positive" if prediction[0] == 1 else "negative"
    confidence = probabilities[0][prediction[0]]

    return {"sentiment": sentiment_label, "confidence": f"{confidence:.2f}"}
