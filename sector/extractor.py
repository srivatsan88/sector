import spacy
import json
from sklearn.metrics.pairwise import cosine_similarity
from sector.helpers.sector_helper import clean_text, is_sequential, replace_with_synonyms, lemmatize, lemmatize_dynamic, combine_sentences, combine_sentences_simple, nlp, jaccard_similarity, get_embedding, key_input_coverage, process_text, embed_process
import itertools
from sector.utils.logging_config import logger, set_log_level


def match_sentence(input_sentence, reference_sentences, max_window_size, use_semantic, combine_threshold, debug, search, embed_fn, lexical_algo):
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

    
    logger.debug(f"\nInput Sentence (Original): {input_sentence}")
    #print(f"Lemmatized & Normalized Input Sentence: {' '.join(lemmatized_input)}")

    # Try all combinations of sentences from 1 to max_window_size
    for window_size in range(1, max_window_size + 1):
        for combination in itertools.combinations(range(len(reference_sentences)), window_size):
            distance = combination[-1] - combination[0]
            # Try sequential or ordered combinations
            if (search == 'sequential' and is_sequential(combination)) or (search == 'ordered' and distance < max_window_size*2):
                ordered_combined = combine_sentences(reference_sentences, combination)
                lemmatized_ref_ordered = lemmatize_dynamic(ordered_combined)
                #lemmatized_ref_ordered = replace_with_synonyms(lemmatized_ref_ordered)

                if use_semantic:
                    input_vector = embed_process(lemmatized_input, embed_fn=embed_fn)
                    ref_vector_ordered = embed_process(lemmatized_ref_ordered, embed_fn=embed_fn)
                    score_ordered = cosine_similarity(input_vector, ref_vector_ordered).flatten()[0]
                else:
                    if lexical_algo is None or lexical_algo == 'sentcomp':
                        score_ordered = jaccard_similarity(set(lemmatized_ref_ordered.split()), set(lemmatized_input.split()))
                    else:
                        score_ordered = key_input_coverage(lemmatized_input, lemmatized_ref_ordered)

                
                logger.debug(f"Ordered Combination: {ordered_combined}")
                logger.debug(f"Similarity Score (Ordered): {score_ordered:.5f}")

                if score_ordered > best_score:
                    best_match = ordered_combined
                    best_score = score_ordered
                    match_data["best_match"] = ordered_combined
                    match_data["best_score"] = score_ordered
                    if search == 'sequential':
                        match_data["combination_type"] = "sequential"
                    else:
                        match_data["combination_type"] = "ordered"
                    match_data["combination_size"] = window_size

                # If the score exceeds the combine threshold, stop further merging
                if score_ordered >= combine_threshold:
                    return match_data

            # Try unordered combinations
            if search in ('random'):
                for permutation in itertools.permutations(combination):
                    unordered_combined = combine_sentences(
                        reference_sentences, permutation)
                    lemmatized_ref_unordered = lemmatize_dynamic(
                        unordered_combined)
                    #lemmatized_ref_unordered = replace_with_synonyms(lemmatized_ref_unordered)

                    if use_semantic:
                        input_vector = embed_process(lemmatized_input, embed_fn=embed_fn)
                        ref_vector_unordered = embed_process(lemmatized_ref_unordered, embed_fn=embed_fn)
                        score_unordered = cosine_similarity(
                            input_vector, ref_vector_unordered).flatten()[0]
                    else:
                        if lexical_algo is None or lexical_algo == 'sentcomp':
                            score_unordered = jaccard_similarity(set(lemmatized_ref_unordered.split()), set(lemmatized_input.split()))
                        else:
                            score_unordered = key_input_coverage(lemmatized_input, lemmatized_ref_unordered)

                    
                    logger.debug(f"Unordered Combination: {unordered_combined}")
                    logger.debug(f"Similarity Score (Unordered): {score_unordered:.5f}")

                    if score_unordered > best_score:
                        best_match = unordered_combined
                        best_score = score_unordered
                        match_data["best_match"] = unordered_combined
                        match_data["best_score"] = score_unordered
                        match_data["combination_type"] = "random"
                        match_data["combination_size"] = window_size

                    # If the score exceeds the combine threshold, stop further merging
                    if score_unordered >= combine_threshold:
                        return match_data

    return match_data


def extract_similar_sentences(reference_document, input_text, max_window_size=2, use_semantic=False, combine_threshold=0.6, debug=False, search='sequential', clean_fn=None, embed_fn=None, lexical_algo=None):
    """
    Extracts the most relevant sentences from the reference document by matching each input sentence to all combinations of reference sentences.
    
    :param reference_document: The reference document from which sentences are extracted.
    :param input_text: The input text (summarized or output of RAG).
    :param max_window_size: Maximum number of consecutive sentences to combine for matching.
    :param use_semantic: If True, uses semantic similarity, else uses sliding window.
    :return: A list of JSON-like objects containing matching results.
    """

    if debug:
        set_log_level("debug")
    # Clean the input and reference text
    #reference_document = clean_text(reference_document)
    #input_text = clean_text(input_text)
    reference_document = process_text(reference_document, clean_fn=clean_fn)
    input_text = process_text(input_text, clean_fn=clean_fn)

    # Tokenize reference document into sentences
    reference_doc = nlp(reference_document)
    reference_sentences = [sent.text for sent in reference_doc.sents]

    # Tokenize input text into sentences
    input_doc = nlp(input_text)
    input_sentences = [sent.text for sent in input_doc.sents]
    
    matched_sentences = []

    # Match input sentences to reference sentence combinations
    for input_sentence in input_sentences:
        result = match_sentence(input_sentence, reference_sentences, max_window_size, use_semantic,combine_threshold, debug, search, embed_fn, lexical_algo)
        matched_sentences.append(result)

    return matched_sentences

    #return json.dumps(str(matched_sentences), indent=2)
