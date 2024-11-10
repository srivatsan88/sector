import streamlit as st
import json
from sector.sector_main import run_sector
from sector.helpers.external_metrics import calculate_bleu, calculate_gleu, calculate_meteor, calculate_rouge, get_scores_as_json


# Streamlit App Title
st.title("SECTOR - Similarity Matcher")

# User Input Section
st.header("Input Text and Reference Document")
input_text = st.text_area("Enter the Input Text (LLM Response)", "Supervised learning is a technique in machine learning where a model is trained on labeled data.")
reference_text = st.text_area("Enter the Reference Document (Context Document)", "Machine learning is a subset of artificial intelligence (AI) that allows systems to learn from data and make decisions without being explicitly programmed. It is widely used in areas such as natural language processing, computer vision, and autonomous systems. One of the most common techniques in machine learning is supervised learning, where a model is trained on labeled data. Another important approach is unsupervised learning, which allows systems to learn from unlabeled data. Reinforcement learning, a third technique, is used when systems learn by interacting with an environment to maximize a reward signal. Machine learning models are evaluated using various metrics such as accuracy, precision, recall, and F1 score. These metrics help in determining how well a model generalizes to unseen data.")

# Configuration Options
st.header("Match Configuration Parameters")
max_window_size = st.slider("Max Window Size", 1, 5, 3)
use_semantic = st.checkbox("Use Semantic Matching", True)
combine_threshold = st.slider("Combine Threshold", 0.0, 1.0, 0.999, step=0.001,  format="%f")
top_n_individual = st.number_input("Top N Individual Matches", min_value=1, max_value=5, value=2)
top_n_aggregated = st.number_input("Top N Aggregated Matches", min_value=1, max_value=5, value=2)
debug = st.checkbox("Debug Mode", False)
search = st.selectbox("Search Mode", options=["sequential", "ordered","random"], index=0)
lexical_algo = st.selectbox("Lexical Algorithm", options=["sentcomp", "keycomp"], index=0)
comparator = st.selectbox("Comparator Function", options=["Sector", "External","Bring Your Own Metrics"], index=0)


# Run the function if input is provided
if st.button("Run Sector") and input_text and reference_text:
    # Run `run_sector` with the specified parameters
    sentence_results,final_score = run_sector(
        input_text,
        reference_text,
        max_window_size=max_window_size,
        use_semantic=use_semantic,
        combine_threshold=combine_threshold,
        top_n_individual=top_n_individual,
        top_n_aggregated=top_n_aggregated,
        debug=debug, 
        search=search,
        clean_fn=None,
        embed_fn=None,
        lexical_algo=lexical_algo
    )


    if comparator == 'Sector':
        # Display each sentence result in a styled card layout
        for sentence in sentence_results:
            with st.container():
                st.markdown("#### 🔍 Matched Sentence Pair")
                st.markdown(
                    f"""
                    <div style="padding:10px; background-color:#f9f9f9; border-radius:8px;">
                        <strong>Input Sentence:</strong> {sentence["input_sentence"]}<br>
                        <strong>Matched Reference:</strong> {sentence["matched_reference"]}<br>
                        <strong>Sentence Similarity Score:</strong> {sentence["scores"]["geometric_mean_top_n"]}
                    </div>
                    """, unsafe_allow_html=True
                )
                st.write("")  

        # Display Average Scores in a more structured format
        st.markdown("### 📊 Average Scores")
        cols = st.columns(2)  # Arrange average scores in two columns for better layout
        for idx, (key, value) in enumerate(final_score["average_scores"].items()):
            with cols[idx % 2]:  # Alternate columns
                st.write(f"**{key.replace('_', ' ').capitalize()}**: {value}")

        # Display Sector Context Similarity Score in a separate section with emphasis
        st.markdown("### 🎯 Sector Context Similarity Score")
        st.markdown(
            f"""
            <div style="padding:10px; background-color:#e0f7fa; border-radius:8px;">
                <h4 style="color:#00796b;">{final_score["sector_context_similarity"]}</h4>
            </div>
            """, unsafe_allow_html=True
        )
    elif comparator == "External":

        combined_scores = []

        score_sums = {
            "BLEU": 0.0,
            "GLEU": 0.0,
            "ROUGE-1": 0.0,
            "ROUGE-2": 0.0,
            "ROUGE-L": 0.0,
            "METEOR": 0.0
        }    
        for sentence in sentence_results:
            input_sentence = sentence['input_sentence']
            reference_sentence = sentence['matched_reference']

            scores_json_sector = get_scores_as_json(reference_sentence, input_sentence)

            combined_scores.append(scores_json_sector)


        num_entries = len(combined_scores)
        for scores in combined_scores:
            for key in score_sums:
                score_sums[key] += scores[key]
        
        # Calculate average scores
        average_scores = {key: score_sums[key] / num_entries for key in score_sums}

        scores_json = get_scores_as_json(reference_text, input_text)
        # st.write("Without Sector")
        # st.write(scores_json)

        # st.write("With Sector")
        # st.write(scores_json_sector)

        table_html = """
        <html>
        <head>
            <style>
                .custom-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: Arial, sans-serif;
                    margin: 20px 0;
                }
                .custom-table th, .custom-table td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }
                .custom-table th {
                    background-color: #f2f2f2;
                    color: #333;
                    font-weight: bold;
                }
                .custom-table tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .custom-table tr:hover {
                    background-color: #e6f7ff;
                }
            </style>
        </head>
        <body>
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Without Sector</th>
                        <th>With Sector</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Populate the table rows
        for metric, without_sector_value in scores_json.items():
            with_sector_value = scores_json_sector[metric]
            table_html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{without_sector_value:.4f}</td>
                    <td>{with_sector_value:.4f}</td>
                </tr>
            """

        # Close the HTML content
        table_html += """
                </tbody>
            </table>
        </body>
        </html>
        """

        # Display the styled HTML table
        st.components.v1.html(table_html, height=400, scrolling=True)

    elif comparator == "Bring Your Own Metrics":
        st.write("Streamlit version currently does not support Bring your own custom metrics. Please check examples directory on running it directly")
