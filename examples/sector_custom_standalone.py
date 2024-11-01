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
        
        # Split the content by lines to separate items like "Increased Flooding" and "Coastal Erosion"
        for line in content.splitlines():
            line = line.strip()
            if line:  # Only process non-empty lines
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

print(reformat_sections(reference_doc))

match_sentences,final_score = run_sector(
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

print(json.dumps(match_sentences, indent=2))
print(json.dumps(final_score, indent=2))
