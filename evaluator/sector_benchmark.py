import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.gleu_score import sentence_gleu

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import json

from sector.extractor import extract_similar_sentences


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

if __name__=="__main__":
    
    input_text_list = ["AI simulates human intelligence. It can recognize speech and perform tasks like humans.", 
    "Supervised learning is a technique in machine learning where a model is trained on labeled data.",
    "In quantum computing, qubits are used to encode information. Unlike classical bits, qubits can exist in a superposition of both 0 and 1 states.",
    "Global warming is caused by the release of greenhouse gases like carbon dioxide, primarily from burning fossil fuels, which trap heat from the sun.",
    "The key components of blockchain include cryptographic hash functions, consensus mechanisms like proof-of-work, and smart contracts that automate transactions.",
    "The circulatory system is responsible for transporting oxygen, nutrients, and waste products throughout the body."
    ]

    reference_doc_list = ["Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and act like humans. The goal of AI is to perform tasks that would typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
    "Machine learning is a subset of artificial intelligence (AI) that allows systems to learn from data and make decisions without being explicitly programmed. It is widely used in areas such as natural language processing, computer vision, and autonomous systems. One of the most common techniques in machine learning is supervised learning, where a model is trained on labeled data. Another important approach is unsupervised learning, which allows systems to learn from unlabeled data. Reinforcement learning, a third technique, is used when systems learn by interacting with an environment to maximize a reward signal. Machine learning models are evaluated using various metrics such as accuracy, precision, recall, and F1 score. These metrics help in determining how well a model generalizes to unseen data.",
    "Quantum computing is an area of study focused on developing computer technology based on the principles of quantum theory, which explains the behavior of energy and material on the atomic and subatomic levels. Traditional computers encode information in binary (bits), where each bit is either a 0 or 1. Quantum computers use quantum bits or qubits, which can exist simultaneously in a superposition of 0 and 1 states. Quantum computing has the potential to solve complex problems much faster than classical computers, especially in areas such as cryptography, optimization, and drug discovery. One of the key principles in quantum computing is entanglement, where qubits that are entangled can influence each other, even when separated by large distances. Another important principle is quantum tunneling, which allows particles to pass through barriers that would be impossible for classical particles.",
    "Climate change refers to long-term changes in global temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle, but in recent centuries, human activities have been the dominant cause of climate change, particularly through the burning of fossil fuels like coal, oil, and gas. These burnings release greenhouse gases such as carbon dioxide (CO2) into the atmosphere. Greenhouse gases trap heat from the sun and cause global temperatures to rise. This process is known as global warming. The effects of climate change include rising sea levels, more extreme weather events such as hurricanes and droughts, and changes in biodiversity. Global efforts to combat climate change focus on reducing greenhouse gas emissions and adopting renewable energy sources.",
    "Blockchain technology is a decentralized digital ledger that records transactions across a network of computers. It is designed to be secure, transparent, and immutable, meaning once a transaction is recorded, it cannot be altered or deleted. The decentralized nature of blockchain ensures that no single entity has control over the entire network. Blockchain is the underlying technology behind cryptocurrencies like Bitcoin, but its applications go far beyond digital currencies. Blockchain can be used in supply chain management, to ensure the authenticity of goods, in healthcare to securely store patient records, and in voting systems to prevent fraud. The key components of blockchain include cryptographic hash functions, consensus mechanisms like proof-of-work, and smart contracts that automate transactions based on predefined conditions.",
    "The human body is composed of several systems that work together to maintain homeostasis and ensure survival. The circulatory system transports oxygen, nutrients, and waste products throughout the body. The respiratory system exchanges oxygen and carbon dioxide with the environment. The digestive system breaks down food and absorbs nutrients, while the nervous system coordinates the body's responses to internal and external stimuli. The musculoskeletal system provides structural support and enables movement, and the endocrine system regulates hormones that control various bodily functions. The human brain, part of the nervous system, is the control center of the body and is responsible for thoughts, emotions, and voluntary movements."
    ]

for docs in range(len(input_text_list)):

    combined_scores = []

    score_sums = {
        "BLEU": 0.0,
        "GLEU": 0.0,
        "ROUGE-1": 0.0,
        "ROUGE-2": 0.0,
        "ROUGE-L": 0.0,
        "METEOR": 0.0
    }    

    # Get the scores in JSON format
    scores_json = get_scores_as_json(reference_doc_list[docs], input_text_list[docs])
    print("Without Sector")
    print(json.dumps(scores_json, indent=4))

    similar_sentences_json = extract_similar_sentences(
        reference_doc_list[docs],
        input_text_list[docs],
        max_window_size=3,  # Combine consecutive sentences if needed
        use_semantic=True,  # Set to True for semantic matching or False for simple sliding window
        combine_threshold=0.996,  # Threshold for combining sentences
        debug=False, 
        search='sequential'
    )

    # Output the matched sentences in JSON format
    for sentence in similar_sentences_json:
        input_sentence = sentence['input_sentence']
        reference_sentence = sentence['best_match']
        print(input_sentence)
        print(reference_sentence)

        scores_json = get_scores_as_json(reference_sentence, input_sentence)

        combined_scores.append(scores_json)


    num_entries = len(combined_scores)
    for scores in combined_scores:
        for key in score_sums:
            score_sums[key] += scores[key]
    
    # Calculate average scores
    average_scores = {key: score_sums[key] / num_entries for key in score_sums}

    print("With Sector")
    print(json.dumps(scores_json, indent=4))
