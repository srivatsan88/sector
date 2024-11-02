import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import json

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothing_function)

# Function to calculate GLEU score 
def calculate_gleu(reference, candidate):
    return sentence_gleu([reference.split()], candidate.split())

# Function to calculate ROUGE score
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Function to calculate METEOR score
def calculate_meteor(reference, candidate):
    return meteor_score([reference.split()], candidate.split())

# Function to get all the scores and return as a JSON object
def get_scores_as_json(reference_text, candidate_text):
    # Calculate scores
    bleu_score = calculate_bleu(reference_text, candidate_text)
    gleu_score = calculate_gleu(reference_text, candidate_text)
    rouge_scores = calculate_rouge(reference_text, candidate_text)
    meteor_score_value = calculate_meteor(reference_text, candidate_text)
    
    # Prepare the JSON output
    result = {
        "BLEU": bleu_score,
        "GLEU": gleu_score,
        "ROUGE-1": rouge_scores['rouge1'].fmeasure,
        "ROUGE-2": rouge_scores['rouge2'].fmeasure,
        "ROUGE-L": rouge_scores['rougeL'].fmeasure,
        "METEOR": meteor_score_value
    }
    
    #return json.dumps(result, indent=4)
    return result