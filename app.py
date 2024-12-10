import streamlit as st
import requests
import logging

# Configuration des logs
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration de l'application
st.title("Analyse des Sentiments des Tweets")
st.subheader("Prédisez le sentiment d'un tweet avec votre modèle.")

# URL de l'API
API_URL = "https://sentiment-analysis-app-eqdcaqf6b8gwbtg3.canadacentral-01.azurewebsites.net/predict"
LOG_TRACE_URL = "https://sentiment-analysis-app-eqdcaqf6b8gwbtg3.canadacentral-01.azurewebsites.net/log_trace"

# Initialiser les états dans la session
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "tweet_input" not in st.session_state:
    st.session_state.tweet_input = ""
if "action_taken" not in st.session_state:
    st.session_state.action_taken = False

# Champ de saisie pour le texte du tweet
st.session_state.tweet_input = st.text_area("Entrez un tweet à analyser :", value=st.session_state.tweet_input, height=100)

# Bouton pour envoyer le tweet à l'API
if st.button("Analyser le tweet"):
    if st.session_state.tweet_input.strip():
        try:
            # Log avant l'appel
            logger.info(f"Envoi de la requête POST à {API_URL} avec le texte : {st.session_state.tweet_input}")

            # Appel à l'API
            with st.spinner("Analyse en cours..."):
                response = requests.post(
                    API_URL,
                    json={"text": st.session_state.tweet_input},
                    headers={"Content-Type": "application/json"}
                )

            # Gestion des réponses
            if response.status_code == 200:
                data = response.json()
                st.session_state.prediction = {
                    "sentiment": data.get("sentiment", "Inconnu"),
                    "confidence": data.get("confidence", "N/A")
                }
                st.session_state.action_taken = False
                st.success(
                    f"Sentiment prédit : **{st.session_state.prediction['sentiment'].capitalize()}** "
                    f"avec une confiance de **{st.session_state.prediction['confidence']}**"
                )
            else:
                st.error(f"Erreur : {response.status_code} - {response.text}")
                logger.error(f"Erreur lors de l'appel à l'API : {response.status_code} - {response.text}")
        except Exception as e:
            logger.exception("Erreur inattendue lors de la communication avec l'API :")
            st.error(f"Erreur lors de la communication avec l'API : {e}")
    else:
        st.warning("Veuillez entrer un texte avant de cliquer sur Analyser.")

# Afficher les actions si une prédiction existe
if st.session_state.prediction and not st.session_state.action_taken:
    if st.button("Prédiction correcte"):
        st.info("Merci pour votre validation !")
        logger.info(
            f"L'utilisateur a validé la prédiction : {st.session_state.prediction['sentiment']} "
            f"avec confiance {st.session_state.prediction['confidence']}."
        )
        st.session_state.action_taken = True

    if st.button("Prédiction incorrecte"):
        # st.error("Erreur signalée ! Trace envoyée.")
        logger.warning(f"L'utilisateur a signalé une erreur pour le texte : {st.session_state.tweet_input}")
        st.session_state.action_taken = True

        # Envoi des informations d'erreur à l'API
        try:
            logger.info("Envoi de la requête POST à /log_trace")
            trace_response = requests.post(
                LOG_TRACE_URL,
                json={
                    "text": st.session_state.tweet_input,
                    "predicted_sentiment": st.session_state.prediction["sentiment"],
                    "confidence": st.session_state.prediction["confidence"]
                },
                headers={"Content-Type": "application/json"}
            )
            logger.debug(f"Réponse de l'API /log_trace : {trace_response.status_code} - {trace_response.text}")

            if trace_response.status_code == 200:
                st.success("Trace envoyée avec succès à l'API.")
            else:
                st.error(f"Erreur lors de l'envoi de la trace : {trace_response.text}")
                logger.error(f"Erreur lors de l'appel à /log_trace : {trace_response.status_code} - {trace_response.text}")
        except Exception as trace_error:
            logger.exception("Erreur inattendue lors de l'envoi à /log_trace :")
            st.error("Une erreur est survenue lors de l'envoi de la trace.")

# Instructions pour l'utilisateur
st.markdown("""
### Instructions :
1. Saisissez un tweet dans le champ ci-dessus.
2. Cliquez sur **Analyser le tweet** pour voir le sentiment prédit.
3. Utilisez les boutons pour valider ou signaler une erreur dans la prédiction.
""")
