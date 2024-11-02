
# Sector Output Summary

This document provides an overview of the output produced by the Sector library. Each section in the output reflects different aspects of similarity between the input text and reference document, helping users evaluate alignment with the context.

Below is an example of a typical Sector output, containing sentence-level matched pairs and similarity scores.

## Example Sector Output

```json
{
  "matched_sentences": [
    {
      "input_sentence": "AI simulates human intelligence.",
      "matched_reference": "Artificial Intelligence AI refers to the simulation of human intelligence in machines that are programmed to think and act like humans.",
      "sentence_similarity_score": 1.0
    },
    {
      "input_sentence": "It can recognize speech and perform tasks like humans.",
      "matched_reference": "The goal of AI is to perform tasks that would typically require human intelligence such as visual perception speech recognition decisionmaking and language translation.",
      "sentence_similarity_score": 0.6928203230275509
    }
  ],
  "scores": {
    "content_word_similarity": 0.14285714285714285,
    "ngram_fuzzy_match_score": 0.26666666666666666,
    "levenshtein_similarity": 0.2431556948798328,
    "pos_based_alignment_score": 0.9,
    "word_coverage": 0.8,
    "key_word_coverage": 0.3666666666666667,
    "geometric_mean_top_n": 0.8464101615137755,
    "sector_context_similarity": 0.8323582900575635
  }
}
```

### Explanation of Output

1. **Matched Sentences**: 
   - The `matched_sentences` section lists the most relevant sentence matches between the input and reference document. For each input sentence, Sector identifies a corresponding sentence (or sentence fragment) in the reference that best aligns semantically. The `sentence_similarity_score` for each pair indicates the level of similarity between the input and reference sentence, allowing for a detailed sentence-by-sentence analysis.

2. **Scores**:
   - The `scores` section provides quantitative metrics, each representing a different dimension of similarity. These scores help in understanding the degree of alignment between the input and reference text at various levels, such as content words, structure, and overall relevance. All scores are on a scale from **0 to 1**, where values closer to 1 indicate a higher degree of similarity.

## Score Descriptions

| **Score Name**               | **Description**                                                                                                                                                                                                                                  | **Value** |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| **content_word_similarity**  | Calculates the **Jaccard similarity** between the sets of **content words** (nouns, verbs, adjectives) in the input and reference text. Words are lemmatized and replaced with synonyms for better generalization before comparison. A higher value indicates closer alignment in terms of essential content words. | 0 - 1             |
| **ngram_fuzzy_match_score**  | Calculates the fuzzy match between n-grams in the input and reference, emphasizing partial phrase matches. This metric captures overlapping patterns in phrasing.                                                                                 | 0 - 1             |
| **levenshtein_similarity**   | Evaluates the similarity based on the Levenshtein distance (edit distance) between input and reference, considering word rearrangements and minor edits.                                                                                          | 0 - 1             |
| **pos_based_alignment_score**| Measures similarity by aligning parts of speech (POS) tags, which ensures grammatical structure consistency between the input and reference.                                                                                                     | 0 - 1             |
| **word_coverage**            | Calculates the proportion of words in the input that are present in the reference, highlighting word-level alignment.                                                                                                                             | 0 - 1             |
| **key_word_coverage**        | Measures the **proportion of key content words** (non-stopwords with length > 3) from the input text that are present in the reference text, factoring in close matches and synonyms. Uses fuzzy matching with a similarity threshold of 70% and synonym overlap to capture aligned key terms. A higher score indicates better alignment of essential keywords. | 0 - 1             |
| **geometric_mean_top_n**     | The geometric mean of the top `n` similarity scores for the best-matching sentences, balancing various similarity metrics at a sentence level.                                                                                                   | 0 - 1             |
| **sector_context_similarity**| A final metric representing the overall context alignment. It is the geometric mean average of the top `n` sentence level similarity scores, providing an overall context alignment score.                                                                          | 0 - 1             |

## Score Ranges and Thresholds

The score ranges in Sector’s output are flexible and can be adapted to fit the unique requirements of different use cases. Factors such as the complexity of the use case, prompt variability, response specificity, and context length play an essential role in determining ideal threshold values. For instance, higher thresholds may be appropriate in domains requiring factual precision (e.g., legal or medical applications), while more lenient thresholds could be used in creative or exploratory tasks where relevance is subjective.

## Final Remarks

Together, these scores provide a comprehensive view of how well the input aligns with the reference context, capturing various aspects such as content relevance, grammatical structure, and coverage of key terms. The **sector_context_similarity** score is a holistic metric that consolidates sentence-level similarities, serving as the primary indicator of overall context alignment in the Sector output.

Additionally, the **matched_sentences** section of the output provides sentence-by-sentence mappings, showcasing the most relevant matched sentences to allow for a more detailed and contextual comparison, along with **sentence_similarity_score** for each match to quantify the similarity level at the sentence level. This combination of quantitative scores and matched pairs ensures a nuanced understanding of text alignment.



--- 
