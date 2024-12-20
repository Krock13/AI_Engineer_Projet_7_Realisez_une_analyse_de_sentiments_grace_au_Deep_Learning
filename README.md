# Analyse de sentiments avec le Deep Learning

Ce projet vise à développer un système d'analyse de sentiments utilisant des techniques de Deep Learning pour classifier les tweets selon leur tonalité (positive ou négative). Il inclut l'entraînement des modèles, une API pour effectuer des prédictions, et une interface utilisateur pour faciliter l'interaction.

## Prérequis

- **Python 3.9+**
- **pip** pour gérer les bibliothèques Python
- (Optionnel) **Docker** pour le déploiement conteneurisé
- Les dépendances listées dans `requirements.txt`

## Installation

1. **Cloner le dépôt :**

   ```bash
   git clone https://github.com/Krock13/AI_Engineer_Projet_7_Realisez_une_analyse_de_sentiments_grace_au_Deep_Learning.git
   cd AI_Engineer_Projet_7_Realisez_une_analyse_de_sentiments_grace_au_Deep_Learning

   ```

2. **Basculer sur la branche de développement :**

   ```bash
   git checkout feature/model_training

   ```

3. **Créer et activer un environnement virtuel :**

   ```bash
   python -m venv env
   source env/bin/activate  # Sur Windows : .\env\Scripts\activate
   ```

4. **Installer les dépendances :**

   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. **Lancer l'API**

   L'API, développée avec FastAPI, permet de soumettre des tweets pour analyse. Pour démarrer l'API :

   ```bash
   uvicorn main:app --reload
   ```

   Vous pouvez accéder à la documentation interactive ici : http://127.0.0.1:8000/docs.

2. **Tester l'interface utilisateur Streamlit**

   L'application Streamlit propose une interface graphique pour interagir avec le modèle. Lancez-la avec :

   ```bash
   streamlit run app.py
   ```

3. **Tests unitaires avec GitHub Actions**

   Les tests unitaires sont automatiquement exécutés lors d'un push dans la branche dev grâce à un pipeline GitHub Actions. Ce pipeline effectue les étapes suivantes :

   - Installe les dépendances nécessaires.
   - Configure un environnement Python 3.9.
   - Exécute les tests définis dans le dossier tests/.

   Pour lancer les tests localement si nécessaire :

   ```bash
   pytest
   ```

4. **Suivi avec MLFlow**

   Ce projet utilise MLFlow pour suivre les expérimentations pendant l'entraînement des modèles. Les runs MLFlow sont enregistrés dans le dossier `notebooks/mlruns`. Pour que l'interface MLFlow affiche les données correctement, exécutez MLFlow à partir du répertoire `notebooks` :

   ```bash
   cd notebooks
   mlflow ui
   ```

5. **Déploiement**

Le projet inclut un pipeline CI/CD pour le déploiement :

- Branche `main` : Un conteneur Docker est généré et déployé sur Azure.
- Les métriques et alertes de l'API sont surveillées via Azure Monitor.

## Structure du projet

- `notebooks/` : Notebooks Jupyter pour l'entraînement et l'expérimentation des modèles.
- `saved_models/` : Modèles pré-entraînés utilisés par l'API.
- `tests/` : Tests unitaires du projet.
- `main.py` : Application FastAPI.
- `app.py` : Interface Streamlit.

## Gestion des branches

- `feature/model_training` : Développement actif.
- `dev` : Intégration des modifications ; les tests unitaires sont exécutés automatiquement.
- `main` : Branche prête pour la production. Les changements validés déclenchent la génération d'un conteneur Docker et son déploiement sur Azure.
