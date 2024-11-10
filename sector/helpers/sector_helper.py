import spacy
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.metrics import edit_distance
from nltk import ngrams
import re
import nltk
from fuzzywuzzy import fuzz
import numpy as np


# Load spaCy model
nlp = spacy.load('en_core_web_lg')

#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')

# Stopwords
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Clean input text by removing special characters but keeping periods and converting newlines to spaces."""
    text = text.replace("\n", ".")
    text = re.sub(r'[^A-Za-z0-9\s\.\?\!]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize(text):
    """Lemmatize the sentence and remove stopwords."""
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha and token.text.lower() not in stop_words]

# Function to find a related verb for a noun using WordNet
def find_related_verb(word):
    synsets = wordnet.synsets(word, pos=wordnet.NOUN)  # Find noun synsets for the word
    for synset in synsets:
        # Get derivationally related forms
        related_forms = synset.lemmas()[0].derivationally_related_forms()
        for form in related_forms:
            if form.synset().pos() == 'v':  # Check if the related form is a verb
                if fuzz.ratio(word, form.name()) > 70 :
                    return form.name()  # Return the verb form
                else:
                    return word
    return word  # If no related verb found, return the original word

def lemmatize_dynamic(text):
    # Process the sentence using spaCy
    doc = nlp(text)
    
    lemmatized_words = []
    for token in doc:
        # Lemmatize normally using spaCy
        lemma = token.lemma_
        
        # If the word is a noun, try to find a related verb using WordNet
        if token.pos_ == "NOUN":
            lemma = find_related_verb(lemma)
        
        lemmatized_words.append(lemma)
    
    # Return the lemmatized sentence as a string
    return ' '.join(lemmatized_words)

def get_synonyms(word):
    """Get a list of synonyms for a given word using WordNet."""
    synonyms = set()

    if word not in stop_words and len(word) > 3:
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                synonyms.add(lemma.name())
        return synonyms if synonyms else {word}
    else:
        return {word}

def replace_with_synonyms(tokens):
    """Replace tokens with their most common synonym."""
    return [get_synonyms(token).pop() if get_synonyms(token) else token for token in tokens]

def generate_ngrams(tokens, n=2):
    """Generate n-grams from a list of tokens."""
    return list(ngrams(tokens, n))

def filter_content_words(tokens):
    """Filter content words (nouns, verbs, adjectives) from tokens."""
    doc = nlp(" ".join(tokens))
    return [token.text for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ']]

def jaccard_similarity(set1, set2):
    """Calculate the Jaccard Similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0

def edit_distance_similarity(tokens1, tokens2):
    """Calculate similarity based on edit distance."""
    distance = edit_distance(tokens1, tokens2)
    return 1 - (distance / max(len(tokens1), len(tokens2)))  # Normalize to get a score between 0 and 1

def key_input_coverage(input_text, reference_text):
    """Calculate how much of the input sentence is covered by the reference sentence based on key words."""
    lemmatized_input = lemmatize_dynamic(input_text).split()
    lemmatized_ref = lemmatize_dynamic(reference_text).split()

    lemmatized_input = list(set([token for token in lemmatized_input if token not in stop_words  and len(token) >= 3]))
    lemmatized_ref = list(set([token for token in lemmatized_ref if token not in stop_words and len(token) >= 3]))

    match_count = 0
    for input in lemmatized_input:
        for reference in lemmatized_ref:
            similarity = fuzz.ratio(input, reference)

            if similarity > 70:
                match_count = match_count + 1
            else:

                inp_synonym = get_synonyms(input)
                ref_synonym = get_synonyms(reference)
                if len(inp_synonym & ref_synonym) >=1:
                    match_count = match_count + 1
    coverage_score = match_count / len(lemmatized_ref) if lemmatized_input else 0

    return min(coverage_score,1.0)


def combine_sentences_simple(sentences, start, window_size):
    """Combine consecutive sentences starting from a specific index."""
    combined = ' '.join(sentences[start:start + window_size])
    return combined

def combine_sentences(sentences, combination):
    """Combine sentences based on the selected combination of indices."""
    combined = ' '.join([sentences[i] for i in combination])
    return combined

#function to pick only sequential sentences
def is_sequential(sentences):
    return all(sentences[i] + 1 == sentences[i + 1] for i in range(len(sentences)-1))

#Function to get embedding of a sentence
#currently this uses spacy embedding model
#To get better extraction accuracy switch to better embedding models or custom models

def get_embedding(sentence):
    return nlp(" ".join(sentence)).vector.reshape(1, -1)

# Calculating individual statistics

def calculate_statistics(float_values):
    mean = np.mean(float_values)
    std_dev = np.std(float_values)
    min_val = np.min(float_values)
    max_val = np.max(float_values)
    percentiles = np.percentile(float_values, [25, 50, 75, 95, 99])
    
    # Creating a dictionary to store results
    summary = {
        "Mean": mean,
        "Standard Deviation": std_dev,
        "Min": min_val,
        "Max": max_val,
        "25th Percentile": percentiles[0],
        "Median (50th Percentile)": percentiles[1],
        "75th Percentile": percentiles[2],
        "95th Percentile": percentiles[3],
        "99th Percentile": percentiles[4]
    }
    
    return summary

def process_text(text, clean_fn=None):
    if clean_fn is None:
        clean_fn = clean_text
    return clean_fn(text)

def embed_process(sentence, embed_fn=None):
    if embed_fn is None:
        embed_fn = get_embedding
    return embed_fn(sentence)
