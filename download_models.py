import nltk
import spacy

# Download nltk data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Download spaCy model
spacy.cli.download("en_core_web_lg")