import spacy
import json
from sklearn.metrics.pairwise import cosine_similarity
from sector_helper import clean_text, replace_with_synonyms, lemmatize, lemmatize_dynamic, combine_sentences, combine_sentences_simple, nlp, jaccard_similarity
import itertools


# Load NLP model 
#nlp = spacy.load('en_core_web_lg')

# def jaccard_similarity(set1, set2):
#     """Calculate the Jaccard Similarity between two sets."""
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     return intersection / union

def match_sentence(input_sentence, reference_sentences, max_window_size, use_semantic, combine_threshold=0.6, debug=False):
    """Matches an input sentence to a set of reference sentence combinations using both ordered and unordered methods."""
    best_match = None
    best_score = 0
    match_data = {
        "input_sentence": input_sentence,
        "best_match": None,
        "best_score": 0.0,
        "combination_type": None,  # Ordered or Unordered
        "combination_size": 0,     # How many sentences were combined
    }

    # Lemmatize and apply synonym replacement (optional) to the input sentence
    lemmatized_input = lemmatize_dynamic(input_sentence)
    #lemmatized_input = replace_with_synonyms(lemmatized_input)

    if debug:
        print(f"\nInput Sentence (Original): {input_sentence}")
        #print(f"Lemmatized & Normalized Input Sentence: {' '.join(lemmatized_input)}")

    # Try all combinations of sentences from 1 to max_window_size
    for window_size in range(1, max_window_size + 1):
        for combination in itertools.combinations(range(len(reference_sentences)), window_size):
            # Try ordered combinations
            ordered_combined = combine_sentences(reference_sentences, combination)
            lemmatized_ref_ordered = lemmatize_dynamic(ordered_combined)
            #lemmatized_ref_ordered = replace_with_synonyms(lemmatized_ref_ordered)

            if use_semantic:
                input_vector = nlp(" ".join(lemmatized_input)).vector.reshape(1, -1)
                ref_vector_ordered = nlp(" ".join(lemmatized_ref_ordered)).vector.reshape(1, -1)
                score_ordered = cosine_similarity(input_vector, ref_vector_ordered).flatten()[0]
            else:
                score_ordered = jaccard_similarity(set(lemmatized_ref_ordered), set(lemmatized_input))

            if debug:
                print(f"Ordered Combination: {ordered_combined}")
                print(f"Similarity Score (Ordered): {score_ordered:.5f}")

            if score_ordered > best_score:
                best_match = ordered_combined
                best_score = score_ordered
                match_data["best_match"] = ordered_combined
                match_data["best_score"] = score_ordered
                match_data["combination_type"] = "ordered"
                match_data["combination_size"] = window_size

            # If the score exceeds the combine threshold, stop further merging
            if score_ordered >= combine_threshold:
                return match_data

            # Try unordered combinations
            for permutation in itertools.permutations(combination):
                unordered_combined = combine_sentences(reference_sentences, permutation)
                lemmatized_ref_unordered = lemmatize_dynamic(unordered_combined)
                #lemmatized_ref_unordered = replace_with_synonyms(lemmatized_ref_unordered)

                if use_semantic:
                    ref_vector_unordered = nlp(" ".join(lemmatized_ref_unordered)).vector.reshape(1, -1)
                    score_unordered = cosine_similarity(input_vector, ref_vector_unordered).flatten()[0]
                else:
                    score_unordered = jaccard_similarity(set(lemmatized_ref_unordered), set(lemmatized_input))

                if debug:
                    print(f"Unordered Combination: {unordered_combined}")
                    print(f"Similarity Score (Unordered): {score_unordered:.5f}")

                if score_unordered > best_score:
                    best_match = unordered_combined
                    best_score = score_unordered
                    match_data["best_match"] = unordered_combined
                    match_data["best_score"] = score_unordered
                    match_data["combination_type"] = "unordered"
                    match_data["combination_size"] = window_size

                # If the score exceeds the combine threshold, stop further merging
                if score_unordered >= combine_threshold:
                    return match_data

    return match_data


def extract_similar_sentences(reference_document, input_text, max_window_size=2, use_semantic=False, combine_threshold=0.6):
    """
    Extracts the most relevant sentences from the reference document by matching each input sentence to all combinations of reference sentences.
    
    :param reference_document: The reference document from which sentences are extracted.
    :param input_text: The input text (summarized or output of RAG).
    :param max_window_size: Maximum number of consecutive sentences to combine for matching.
    :param use_semantic: If True, uses semantic similarity, else uses sliding window.
    :return: A list of JSON-like objects containing matching results.
    """
    # Clean the input and reference text
    reference_document = clean_text(reference_document)
    input_text = clean_text(input_text)

    # Tokenize reference document into sentences
    reference_doc = nlp(reference_document)
    reference_sentences = [sent.text for sent in reference_doc.sents]

    # Tokenize input text into sentences
    input_doc = nlp(input_text)
    input_sentences = [sent.text for sent in input_doc.sents]
    
    matched_sentences = []

    # Match input sentences to reference sentence combinations
    for input_sentence in input_sentences:
        result = match_sentence(input_sentence, reference_sentences, max_window_size, use_semantic,combine_threshold)
        matched_sentences.append(result)

    return matched_sentences

    #return json.dumps(str(matched_sentences), indent=2)
