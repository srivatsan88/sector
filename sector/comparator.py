import json
import math
import nltk
from nltk.metrics import edit_distance
from sector.helpers.sector_helper import lemmatize, replace_with_synonyms, generate_ngrams, filter_content_words, jaccard_similarity, lemmatize_dynamic, get_synonyms, stop_words, key_input_coverage
from fuzzywuzzy import fuzz

def jaccard_content_word_similarity(input_text, reference_text):
    """Calculate Jaccard similarity based on content words (nouns, verbs, adjectives)."""
    lemmatized_input = replace_with_synonyms(lemmatize(input_text))
    lemmatized_ref = replace_with_synonyms(lemmatize(reference_text))

    #lemmatized_input = lemmatize_dynamic(input_text).split()
    
    #lemmatized_ref = lemmatize_dynamic(reference_text).split()

    #print(lemmatized_input)
    #print(lemmatize_dynamic(input_text))

    #print(lemmatized_ref)
    #print(lemmatize_dynamic(reference_text))


    content_input = set(filter_content_words(lemmatized_input))
    content_reference = set(filter_content_words(lemmatized_ref))

    return jaccard_similarity(content_input, content_reference)

def ngram_fuzzy_match_score(input_text, reference_text, n=2):
    """Calculate the n-gram match score using fuzzy matching between input and reference."""
    lemmatized_input = replace_with_synonyms(lemmatize(input_text))
    lemmatized_ref = replace_with_synonyms(lemmatize(reference_text))

    #lemmatized_input = lemmatize_dynamic(input_text).split()    
    #lemmatized_ref = lemmatize_dynamic(reference_text).split()

    input_ngrams = set(generate_ngrams(lemmatized_input, n))
    reference_ngrams = set(generate_ngrams(lemmatized_ref, n))


    matched_ngrams = input_ngrams.intersection(reference_ngrams)

    return len(matched_ngrams) / len(input_ngrams) if input_ngrams else 0

def levenshtein_similarity(input_text, reference_text):
    """Calculate the normalized Levenshtein distance between input and reference text."""
    lemmatized_input = " ".join(replace_with_synonyms(lemmatize(input_text)))
    lemmatized_ref = " ".join(replace_with_synonyms(lemmatize(reference_text)))

    #lemmatized_input = lemmatize_dynamic(input_text).split()
    
    #lemmatized_ref = lemmatize_dynamic(reference_text).split()

    lev_distance = edit_distance(lemmatized_input, lemmatized_ref)
    max_length = max(len(lemmatized_input), len(lemmatized_ref))

    return 1 - (lev_distance / max_length) if max_length > 0 else 0

def pos_based_alignment(input_text, reference_text):
    """Align tokens based on POS tags and measure similarity."""
    #lemmatized_input = replace_with_synonyms(lemmatize(input_text))
    #lemmatized_ref = replace_with_synonyms(lemmatize(reference_text))

    lemmatized_input = lemmatize_dynamic(input_text).split()
    
    lemmatized_ref = lemmatize_dynamic(reference_text).split()

    input_doc = nltk.pos_tag(lemmatized_input)
    ref_doc = nltk.pos_tag(lemmatized_ref)

    input_pos = [word for word, pos in input_doc if pos.startswith(('N', 'V', 'J'))]  # Nouns, Verbs, Adjectives
    ref_pos = [word for word, pos in ref_doc if pos.startswith(('N', 'V', 'J'))]

    matched_pos = set(input_pos).intersection(set(ref_pos))
    return len(matched_pos) / len(input_pos) if input_pos else 0

def input_coverage(input_text, reference_text):
    """Calculate how much of the input sentence is covered by the reference sentence."""
    #lemmatized_input = replace_with_synonyms(lemmatize(input_text))
    #lemmatized_ref = replace_with_synonyms(lemmatize(reference_text))

    lemmatized_input = lemmatize_dynamic(input_text).split()
    
    lemmatized_ref = lemmatize_dynamic(reference_text).split()

    input_tokens = set(lemmatized_input)
    reference_tokens = set(lemmatized_ref)

    matched_tokens = input_tokens.intersection(reference_tokens)
    coverage_score = len(matched_tokens) / len(input_tokens) if input_tokens else 0

    return coverage_score

def geometric_mean_top_n(scores, top_n):
    """Compute the geometric mean of the top N scores."""
    sorted_scores = sorted(scores, reverse=True)[:top_n]
    product = math.prod(sorted_scores)  # Multiply all top N scores
    return product ** (1 / top_n) if top_n > 0 else 0

def compare_sentences_flexible(input_text, reference_text, top_n=3):
    """Compare input and reference sentences using flexible semantic similarity methods and compute geometric mean."""
    
    # Calculate individual scores
    content_word_score = jaccard_content_word_similarity(input_text, reference_text)
    ngram_score = ngram_fuzzy_match_score(input_text, reference_text, n=2)  # Bi-gram fuzzy match
    levenshtein_score = levenshtein_similarity(input_text, reference_text)
    pos_alignment_score = pos_based_alignment(input_text, reference_text)
    overall_coverage_score = input_coverage(input_text, reference_text)
    key_coverage_score = key_input_coverage(input_text, reference_text)

    # List of all scores for geometric mean
    scores = [
        content_word_score,
        ngram_score,
        levenshtein_score,
        pos_alignment_score,
        overall_coverage_score,
        key_coverage_score
    ]

    # Compute geometric mean of top N scores
    geometric_mean_score = geometric_mean_top_n(scores, top_n)

    # Combine all scores into a JSON response
    result = {
        "input_sentence": input_text,
        "matched_reference": reference_text,
        "scores": {
            "content_word_similarity": content_word_score,
            "ngram_fuzzy_match_score": ngram_score,
            "levenshtein_similarity": levenshtein_score,
            "pos_based_alignment_score": pos_alignment_score,
            "word_coverage": overall_coverage_score,
            "key_word_coverage": key_coverage_score,
            "geometric_mean_top_n": geometric_mean_score
        }
    }

    return json.dumps(result, indent=2)
