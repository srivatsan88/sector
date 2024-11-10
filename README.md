
# SECTOR - Semantic Extractor and Comparator Library 

**Sector** is a Python library designed for advanced text extraction, comparison, and intelligent matching, serving as a robust `LLM response evaluation` tool. It enables users to pinpoint the most relevant sentences between an input and a reference document through sophisticated techniques like semantic alignment and customizable probabilistic matching, ensuring precise and context-aware comparisons. With a modular and highly configurable architecture, Sector empowers users to fine-tune each component to effectively evaluate the alignment, relevance, and accuracy of LLM-generated responses within complex contexts.

Traditional comparison techniques like BLEU, METEOR, and cosine similarity offer useful measures for surface-level similarity between strings but often fall short when comparing LLM responses in scenarios with extensive context and specific responses, such as retrieval-augmented generation (RAG) or concise summaries. Sector takes an engineering-driven approach to address this limitation by first extracting only the most relevant sections of the context, ensuring comparisons are focused on information crucial to the query or summary. This targeted extraction enables a precise alignment between context and response, allowing Sector to deliver a nuanced, context-aware evaluation. By complementing traditional metrics with this engineering-driven methodology, Sector provides a robust and accurate solution for LLM response evaluation in complex, context-rich AI generated content.

![Sector Architecture](https://github.com/srivatsan88/sector/raw/main/img/sector.jpg)

Fig: Architecture representing SECTOR components. For more details refer to [Architecture](https://github.com/srivatsan88/sector/blob/main/Architecture.md)

## Features

- **Context-Aware Matching**: Focuses on the most relevant sections of large contexts, ensuring precise alignment with queries or responses.
- **Advanced Semantic and Probabilistic Matching**: Utilizes semantic alignment and customizable probabilistic matching for deeper contextual accuracy.
- **Flexible Scoring Metrics**: Offers multi-dimensional scoring options matching to adapt to various evaluation needs.
- **Configurable Parameters**: Adjustable parameters for sentence matching and scoring thresholds making sector easily adaptable to diverse use cases and diverse scenario like batch and real time scoring.
- **Efficient Performance on Large Texts**: Optimizes comparisons in high-context scenarios, making it ideal for large documents and complex applications.

## Installation 

You can install the Sector library via `pip`: (Also, refer to Note section below)

```bash
pip install sector
```
OR

To install the latest version of **Sector** directly from GitHub, use:

```bash
pip install git+https://github.com/srivatsan88/sector.git
```

> **Note:** The library requires SpaCy and NLTK for NLP processing. Ensure you have the necessary model by running:
> You can download the [python file - download_models](https://github.com/srivatsan88/sector/raw/main/download_models.py) and run it to download all models
> If you are behind firewall and access is blocked then set proxy or manually download the files from respective project repo and install
>
> ```bash
> python download_models.py
> ```

## Usage

Here’s a quick example of how to use the Sector library:

For more examples refer to link [Sector Example](https://github.com/srivatsan88/sector/tree/main/examples)

```python
from sector.sector_main import run_sector

# Define input and reference texts
input_text = "Supervised learning is a technique in machine learning where a model is trained on labeled data."
reference_text = '''
Machine learning is a subset of artificial intelligence (AI) that allows systems to learn from data and make decisions without being explicitly programmed. It is widely used in areas such as natural language processing, computer vision, and autonomous systems. One of the most common techniques in machine learning is supervised learning, where a model is trained on labeled data. Another important approach is unsupervised learning, which allows systems to learn from unlabeled data. Reinforcement learning, a third technique, is used when systems learn by interacting with an environment to maximize a reward signal. Machine learning models are evaluated using various metrics such as accuracy, precision, recall, and F1 score. These metrics help in determining how well a model generalizes to unseen data.
'''

# Run the sector matching function
matched_sentences, final_score = run_sector(
    input_text,
    reference_text,
    max_window_size=2,
    use_semantic=True,
    combine_threshold=0.994,
    top_n_individual=2,
    top_n_aggregated=2,
    debug=False,
    search='ordered',
    clean_fn=None,
    embed_fn=None,
    lexical_algo=None
)

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

print(json.dumps(response, indent=2))
```
For more details on sector output `metrics` refer to [Sector Output](https://github.com/srivatsan88/sector/blob/main/Sector_Output.md)

Fastest way to explore `Sector` is via streamlit app provided in examples directory. Download the code and use below command to get started

> ```bash
> streamlit run examples/streamlit_app/app.py
> ```

## API Reference

- **`run_sector(input_text, reference_text, **params)`**
  - **Parameters**:
    - `input_text`: The sentence or paragraph to match. Typically response from LLM.
    - `reference_text`: The reference or context document against which matching is performed.
    - `max_window_size`: Maximum number of sentences to combine. See below (Sector Search Strategy Combinations) for more details.
    - `use_semantic`: Boolean indicating whether to use semantic matching. Semantic matching is defaulted to spacy embedding model but can be customized to other embedding models using embed_fn below. For better accuracy use ada or custom models.
    - `combine_threshold`: Threshold for combining sentences. See below (Sector Search Strategy Combinations) for more details.
    - `top_n_individual`: Number of top comparator match scores to pick from and compute geometric mean for individual sentences. 
    - `top_n_aggregated`: Number of top aggregated match scores across sentences to compute final geometric mean.
    - `debug`: Enables debug mode if set to `True`.
    - `search`: Specifies the search strategy (e.g., 'sequential', 'ordered', 'random'). See below (Sector Search Strategy Combinations) for more details.
    - `clean_fn`: Optional function for cleaning text. Custom pre-processing function to accomodate specific LLM response format out of prompting.
    - `embed_fn`: Optional function for custom embedding. Bring your own ebedding model for better extraction accuracy. Applicable when use_semantic is set to True.
    - `lexical_algo`: Optional parameter to set algo when use_semantic is False. Options inclide 'sentcomp' or 'keycomp'. Default is sentcomp. Sentcomp does sentence level match to pick relevant content while keycomp uses critical words to extract context.
  - **Returns**: A tuple containing matched sentences and a dictionary with scores.


### Sector Search Strategy Combinations

Sector supports three search strategies—**Sequential**, **Ordered**, and **Random**—which determine how sentence combinations are compared between input and reference texts. The total number of search space combinations (Complexities) varies based on the strategy and is calculated as follows:

#### 1. Sequential Search

In **Sequential Search**, Sector compares sentences in a strict sequence with a sliding window of sizes from 1 up to `max_window_size`.

**Formula**:
For a given number of sentences \( n \) and `max_window_size` \( w \), the total combinations \( C \) for Sequential Search is:

$$
C = \sum_{k=1}^{w} (n - k + 1)
$$

#### 2. Ordered Search

In **Ordered Search**, Sector compares sentences in any ascending order (but not necessarily consecutive). For each window size \( k \) (from 1 up to `max_window_size`), it calculates combinations of sentences while preserving order.

**Formula**:
The total combinations \( C \) for Ordered Search is:

$$
C = \sum_{k=1}^{w} \binom{n}{k} = \sum_{k=1}^{w} \frac{n!}{k!(n - k)!}
$$

#### 3. Random Search

In **Random Search**, Sector allows any combination and order of sentences for each window size up to `max_window_size`. This includes all subsets of sentences in all possible permutations.

**Formula**:
The total combinations \( C \) for Random Search is:

$$
C = \sum_{k=1}^{w} \binom{n}{k} \times k! = \sum_{k=1}^{w} \frac{n!}{(n - k)!}
$$

---

Based on above details for `max_window_size` 3 and 5 `sentences` in context, the search space for sequential is 12, ordered is 25 and random is 85. Larger the search space better is the accuracy at the cost of elevated run time. Let us see in next section on how to balance accuracy and run time based on use case need.

### Balancing Speed and Accuracy with `max_window_size`, `search`, and `combine_threshold`

Sector’s configuration parameters—`max_window_size`, `search`, and `combine_threshold`—together influence the complexity of the search space, requiring careful tuning to balance speed and accuracy based on the use case.

- **`max_window_size`**: Controls the maximum number of sentences combined in each search iteration. A larger `max_window_size` increases combinations and search space, which improves accuracy but raises computational load.

- **`search`**: Specifies the sentence arrangement strategy (e.g., "sequential," "ordered," or "random"). Flexible settings like "random" significantly increase the search space by allowing any order of sentences, while restricted settings like "sequential" and "ordered" limit combinations, providing faster but more constrained searches.

- **`combine_threshold`**: Acts as an early stopping criterion based on matching scores. A higher `combine_threshold` requires higher accuracy for matches, causing the search to continue deeper. This improves relevance in content extraction but slows down execution, as more combinations are evaluated.

### Understanding Sector output

For more details on sector output `metrics` refer to [Sector Output](https://github.com/srivatsan88/sector/blob/main/Sector_Output.md)

### Use Case Patterns

- **Real-Time Applications**: For real-time use cases, prioritize speed by choosing a lower `max_window_size`, a constrained `search` strategy (such as "sequential" or "ordered"), and a moderately low `combine_threshold` to reduce combinations and allow for quicker early stopping with adequate accuracy.

- **Batch Processing**: In batch processing scenarios where accuracy is essential, a higher `max_window_size`, "random" search strategy, and elevated `combine_threshold` are ideal. This setup increases search depth and accuracy but can afford the extra computation time required for exhaustive, high-accuracy extraction in offline analysis.

By configuring these parameters appropriately, Sector allows flexible optimization for speed or accuracy based on the needs of real-time or batch processing applications.


## Examples

The `examples` directory contains sample applications for:
- **Streamlit App**: A web-based example of text matching using Streamlit.
- **Flask App**: A web API for text matching using Flask.
- **Standalone Examples**: Scripts demonstrating different Sector functionalities.

## Benchmarking and Evaluation

Evaluating LLM output for accuracy against context is complex due to factors such as prompt variability, use-case diversity, and the inherently flexible nature of language generation. LLMs can produce a wide range of valid responses depending on subtle changes in prompts, making it difficult to standardize evaluation criteria. Different use cases like retrieval-augmented generation (RAG), summarization, question answering, or conversational agents add further complexity, as each scenario has distinct expectations for relevance and detail. For example, RAG requires precise extraction of relevant information, while summarization emphasizes capturing the essence of a larger context. Additionally, the challenge increases when the context is large, as it requires sophisticated methods to identify and align relevant segments of the context with the output. These factors make it essential to use evaluation techniques that consider meaning, context fit, and structural variations to provide a more accurate and meaningful assessment of LLM performance.

To get started quickly, the `evaluator` folder contains sample Python programs designed to assess Sector's suitability for specific use cases and to determine the appropriate threshold levels applicable to each unique scenario. As these thresholds are use-case specific, there are no predefined high or low values; instead, the evaluator helps establish optimal ranges tailored to individual application needs

## Suggestions to Improve Accuracy

To improve accuracy when using Sector, consider the following customizations:

- **Custom Pre-Processing Function**: Implement a pre-processing function specific to your prompts or responses to normalize text variations and improve consistency. Custom fucntion can be added using `clean_fn` parameter.
- **Advanced or Custom Embedding Models**: Use higher-quality embedding models or custom embeddings tailored to your domain for capturing nuanced language and specific terminology. Custom embedding model can be invoked using `embed_fn` parameter.
- **Use Case-Specific Comparator Functions**: Choose comparator functions that align best with your use case, such as those emphasizing semantic alignment, factual accuracy, or contextual similarity.
- **Adjust `max_window_size` and `combine_threshold`**: Tune these parameters to balance search depth and relevance, enhancing accuracy for your specific context.
  
These adjustments allow Sector to adapt effectively to diverse requirements, yielding precise, context-aware evaluations.


## Contributing

Thank you for your interest in contributing to the **Sector** library! Contributions are welcome and appreciated. There are many ways to get involved, and this document provides guidelines to make the process as smooth as possible.

## Ways to Contribute

- **Reporting Bugs**: If you find a bug, please open an issue.
- **Feature Requests**: Suggest new features or improvements by creating an issue.
- **Documentation**: Improve or add new documentation to help other users.
- **Code Contributions**: To be updated.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For further inquiries or support, feel free to open an issue or reach out via LinkedIn (https://www.linkedin.com/in/srivatsan-srinivasan-b8131b).
