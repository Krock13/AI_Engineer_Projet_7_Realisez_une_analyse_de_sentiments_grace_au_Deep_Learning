import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_tweet(text):
    # Enlever les mentions @pseudo
    text = re.sub(r'@\w+', '', text)
    # Enlever les liens web
    text = re.sub(r'http\S+', '', text)
    # Enlever les caractères spéciaux et ponctuation (sauf les emojis)
    text = re.sub(r'[^\w\s]', '', text)
    # Supprimer les espaces superflus
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Initialiser les outils
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Vérifier si tout le tweet est en majuscule
    if text.isupper():
        # Mettre tout en minuscule si le tweet entier est en majuscule
        text = text.lower()
    else:
        # Tokenizer le texte en mots
        words = text.split()
        
        # Conserver les mots en majuscule, mettre les autres en minuscule et enlever les stopwords
        # Appliquer la lemmatisation
        words = [lemmatizer.lemmatize(word) if word.isupper() 
                 else lemmatizer.lemmatize(word.lower())
                 for word in words if word.lower() not in stop_words]
        
        # Recréer le tweet à partir des mots traités
        text = ' '.join(words)
        
    return text
