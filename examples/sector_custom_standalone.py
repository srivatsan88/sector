from sector.sector_main import run_sector
import json

import re

# This program shows how to use custom pre-processing function to account for prompt variability 

def reformat_sections(text):
    # Regex pattern to match main headings and subheadings, along with their text
    pattern = re.compile(r'(\d+\.\d*\s+)?([A-Za-z\s]+)\n([^1]+)')
    
    # Find all matches
    matches = pattern.findall(text)
    
    # Initialize a list to hold formatted output
    reformatted_text = []
    
    # Loop over each match and format as required
    for match in matches:
        heading = match[1].strip()  # Capture the main heading or subheading
        content = match[2].strip()  # Capture the content associated with the heading
        
        # Split the content by lines to separate items 
        for line in content.splitlines():
            line = line.strip()
            if line:  
                reformatted_text.append(f"{heading} - {line}")
    
    return '\n\n'.join(reformatted_text)


input_text = """
1. Impact of Rising Sea Levels
Climate change leads to rising sea levels as polar ice melts, which directly impacts coastal communities by increasing the frequency and severity of flooding.

1.1 Key Risks
Increased Flooding: Higher sea levels mean that coastal areas are more prone to flooding, particularly during storms.
Coastal Erosion: Warmer ocean temperatures accelerate coastal erosion, reducing land area and threatening buildings near shorelines.
"""
reference_doc = """
Topic: Climate Change and Its Impact on Oceans

Climate change has far-reaching impacts on the oceans, particularly affecting coastal areas and marine ecosystems. Due to rising global temperatures, polar ice caps are melting at an accelerated rate, causing sea levels to rise. This rise in sea levels increases the risk of flooding for coastal communities, especially during extreme weather events. Additionally, warmer ocean temperatures contribute to coastal erosion, which gradually reduces landmass and threatens infrastructure along shorelines. These impacts highlight the vulnerability of coastal regions and the need for adaptive measures to protect these areas.

"""

#print(reformat_sections(reference_doc))

matched_sentences,final_score = run_sector(
    input_text,
    reference_doc,
    max_window_size=3,  # Combine consecutive sentences if needed
    use_semantic=True,  # Set to True for semantic matching or False for simple sliding window
    combine_threshold=0.999,  # Threshold for combining sentences
    top_n_individual=2,
    top_n_aggregated=2,
    debug=False, 
    search='sequential',
    clean_fn=reformat_sections,
    embed_fn=None
)

response = {
    "matched_sentences": [
        {
            "input_sentence": match["input_sentence"],
            "matched_reference": match["matched_reference"]
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

print(json.dumps(response, indent=2))
