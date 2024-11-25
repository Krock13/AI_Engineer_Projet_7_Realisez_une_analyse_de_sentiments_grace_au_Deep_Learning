# Dockerfile optimisé
FROM python:3.9-slim

# Répertoire de travail
WORKDIR /app

# Copier uniquement ce qui est nécessaire
COPY .dockerignore /app/.dockerignore
COPY main.py /app/
COPY requirements-docker.txt /app/
COPY saved_models/logistic_regression_tf-idf.pkl /app/saved_models/
COPY saved_models/tf-idf_vectorizer.pkl /app/saved_models/

# Installer uniquement les dépendances nécessaires
RUN pip install --no-cache-dir -r requirements-docker.txt

# Exposer le port utilisé par FastAPI
EXPOSE 8000

# Commande par défaut
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
