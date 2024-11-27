from flask import Flask, request, Response, jsonify
from sector.sector_main import run_sector
from sector.helpers.sector_helper import nlp
import json

app = Flask(__name__)

# Endpoint for similarity matching
@app.route("/sector", methods=["POST"])
def match_text():
    # Parse input JSON
    data = request.json
    input_text = data.get("input_text")
    reference_text = data.get("reference_text")
    
    # Extract additional parameters with defaults
    max_window_size = data.get("max_window_size", 3)
    use_semantic = data.get("use_semantic", True)
    combine_threshold = data.get("combine_threshold", 0.996)
    top_n_individual = data.get("top_n_individual", 2)
    top_n_aggregated = data.get("top_n_aggregated", 2)
    debug = data.get("debug", False)
    search = data.get("search", "sequential")

    # Run the `run_sector` function with provided inputs and parameters
    try:
        matched_sentences, final_score = run_sector(
            input_text,
            reference_text,
            max_window_size=max_window_size,
            use_semantic=use_semantic,
            combine_threshold=combine_threshold,
            top_n_individual=top_n_individual,
            top_n_aggregated=top_n_aggregated,
            debug=debug,
            search=search,
            input_clean_fn=None,
            context_clean_fn=None,
            embed_fn=None
        )

        # Prepare JSON response with only the specified fields
        response = {
            "matched_sentences": [
                {
                    "input_sentence": match["input_sentence"],
                    "matched_reference": match["matched_reference"],
                    "sentence_similarity_score": match["scores"]["geometric_mean_top_n"]
                }
                for match in matched_sentences
            ],
            "scores": {
                "content_word_similarity": final_score["average_scores"]["content_word_similarity"],
                "ngram_fuzzy_match_score": final_score["average_scores"]["ngram_fuzzy_match_score"],
                "levenshtein_similarity": final_score["average_scores"]["levenshtein_similarity"],
                "pos_based_alignment_score": final_score["average_scores"]["pos_based_alignment_score"],
                "word_coverage": final_score["average_scores"]["word_coverage"],
                "key_word_coverage": final_score["average_scores"]["key_word_coverage"],
                "geometric_mean_top_n": final_score["average_scores"]["geometric_mean_top_n"],
                "sector_context_similarity": final_score["sector_context_similarity"]
            }
        }
        
        return Response(json.dumps(response, sort_keys=False), mimetype='application/json')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



# curl -X POST http://127.0.0.1:5000/sector \
# -H "Content-Type: application/json" \
# -d '{
#   "input_text": "AI simulates human intelligence. It can recognize speech and perform tasks like humans.",
#   "reference_text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. The goal of AI is to perform tasks that would typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
#   "max_window_size": 3,
#   "use_semantic": true,
#   "combine_threshold": 0.996,
#   "top_n_individual": 2,
#   "top_n_aggregated": 2,
#   "debug": false,
#   "search": "sequential"
# }'

