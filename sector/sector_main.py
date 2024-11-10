import json
from sector.comparator import compare_sentences_flexible, geometric_mean_top_n
import math
from sector.extractor import extract_similar_sentences

def process_json_list(json_list, top_n=3):
    """Process a list of input-reference pairs and calculate matcher outputs."""
    combined_scores = []
    # List to store matcher results for each JSON object
    results = []

    for entry in json_list:
        input_sentence = entry["input_sentence"]
        matched_reference = entry["best_match"]

        # Get matcher output for this input-reference pair
        matcher_result = json.loads(compare_sentences_flexible(input_sentence, matched_reference, top_n))
        results.append(matcher_result)

        # Extract the individual scores for this entry
        scores = matcher_result["scores"]
        combined_scores.append(scores)

    return combined_scores, results


def combine_scores(combined_scores, top_n=3):
    """Combine multiple scores arising from individual JSON results."""
    # Initialize variables for averaging and geometric mean
    score_sums = {
        "content_word_similarity": 0,
        "ngram_fuzzy_match_score": 0,
        "levenshtein_similarity": 0,
        "pos_based_alignment_score": 0,
        "word_coverage": 0,
        "key_word_coverage": 0,
        "geometric_mean_top_n": 0
    }

    num_entries = len(combined_scores)
    
    for scores in combined_scores:
        for key in score_sums:
            score_sums[key] += scores[key]
    
    # Calculate average scores
    average_scores = {key: score_sums[key] / num_entries for key in score_sums}

    # Calculate geometric mean across all individual geometric mean scores
    geometric_means = [score["geometric_mean_top_n"] for score in combined_scores]
    sector_context_similarity = geometric_mean_top_n(geometric_means, top_n)
    
    # Combine average and geometric mean scores
    combined_result = {
        "average_scores": average_scores,
        "sector_context_similarity": sector_context_similarity
    }

    return combined_result

def run_sector (input_text, reference_document,  max_window_size=2,  use_semantic=True, combine_threshold=0.85, top_n_individual=2, top_n_aggregated=2, debug=False, search='sequential', clean_fn=None, embed_fn=None, lexical_algo=None):
    similar_sentences_json = extract_similar_sentences(
    reference_document,
    input_text,  #LLM Response
    max_window_size=max_window_size,  # Combine consecutive sentences if needed
    use_semantic=use_semantic,  # Set to True for semantic matching or False for simple sliding window
    combine_threshold=combine_threshold,  # Threshold for combining sentences
    debug=debug,
    search=search,
    clean_fn=clean_fn,
    embed_fn=embed_fn,
    lexical_algo=lexical_algo #select algo when use_semantic is false. Options are sentcomp and keycomp
    )

    # Process the list of JSON objects
    combined_scores, sentence_results = process_json_list(similar_sentences_json, top_n_individual)

    # Combine the individual scores into an aggregate result
    combined_result = combine_scores(combined_scores, top_n_aggregated)
        
    # Output results for each input-reference pair
    #for result in sentence_results:
    #    print(json.dumps(result, indent=2))

    # Output combined score result
    #print("\nCombined Scores across all input-reference pairs:")
    #print(json.dumps(combined_result, indent=2))
    return sentence_results,combined_result
    
if __name__ == "__main__":
    # Load the JSON file (simulating loading it from a file)
    json_data = '''
    [
        {
            "input_sentence": "AI simulates human intelligence.",
            "matched_reference": "Artificial Intelligence AI refers to the simulation of human intelligence in machines that are programmed to think and act like humans.",
            "similarity_score": 0.7797716856002808,
            "is_combined": false
        },
        {
            "input_sentence": "It can recognize speech and perform tasks like humans.",
            "matched_reference": "The goal of AI is to perform tasks that would typically require human intelligence such as visual perception speech recognition decisionmaking and language translation.",
            "similarity_score": 0.842081606388092,
            "is_combined": false
        }
    ]
    '''
    
    # Simulating reading from a JSON file
    json_list = json.loads(json_data)
    
    # Process the list of JSON objects
    combined_scores, results = process_json_list(json_list)

    # Combine the individual scores into an aggregate result
    combined_result = combine_scores(combined_scores, top_n=3)
    
    # Output results for each input-reference pair
    for result in results:
        print(json.dumps(result, indent=2))

    # Output combined score result
    print("\nCombined Scores across all input-reference pairs:")
    print(json.dumps(combined_result, indent=2))
