import streamlit as st
import json
from sector.sector_main import run_sector

# Streamlit App Title
st.title("SECTOR - Similarity Matcher")

# User Input Section
st.header("Input Text and Reference Document")
input_text = st.text_area("Enter the Input Text (LLM Response)", "")
reference_text = st.text_area("Enter the Reference Document (Context Document)", "")

# Configuration Options
st.header("Match Configuration Parameters")
max_window_size = st.slider("Max Window Size", 1, 5, 3)
use_semantic = st.checkbox("Use Semantic Matching", True)
combine_threshold = st.slider("Combine Threshold", 0.0, 1.0, 0.999, step=0.001,  format="%f")
top_n_individual = st.number_input("Top N Individual Matches", min_value=1, max_value=5, value=2)
top_n_aggregated = st.number_input("Top N Aggregated Matches", min_value=1, max_value=5, value=2)
debug = st.checkbox("Debug Mode", False)
search = st.selectbox("Search Mode", options=["sequential", "ordered","random"], index=0)

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
        embed_fn=None
    )



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
            st.write("")  # Adds a bit of spacing

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

