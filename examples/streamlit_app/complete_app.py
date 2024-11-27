import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.mixture import GaussianMixture
from sector.comparator import compare_sentences_flexible
from sector.extractor import extract_similar_sentences
from sector.helpers.sector_helper import calculate_statistics
from sector.helpers.external_metrics import get_scores_as_json
from sector.sector_main import run_sector
from st_aggrid import AgGrid, GridOptionsBuilder
import json
from itertools import product
import random




# Utility functions for threshold determination methods
def compute_summary_statistics(geometric_means):
    return {
        'min': np.min(geometric_means),
        'max': np.max(geometric_means),
        'mean': np.mean(geometric_means),
        'median': np.median(geometric_means),
        '75th_percentile': np.percentile(geometric_means, 75),
        '90th_percentile': np.percentile(geometric_means, 90),
        '95th_percentile': np.percentile(geometric_means, 95),
        '99th_percentile': np.percentile(geometric_means, 99)
    }

def determine_cutoff_with_percentile(geometric_means, percentile=75):
    return np.percentile(geometric_means, percentile)

def determine_cutoff_with_gmm(geometric_means, n_components=2):
    geometric_means_reshaped = np.array(geometric_means).reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(geometric_means_reshaped)
    means = gmm.means_.flatten()
    means.sort()
    cutoff = (means[0] + means[1]) / 2
    return cutoff, means


# Add a complexity heuristic function
def compute_complexity(row):
    """Calculate complexity for a given parameter set."""
    search_algo_complexity = {"Sequential": 1, "Ordered": 2, "Random": 3}
    use_semantic_complexity = 1 if row["Use Semantic"] else 0
    return (
        row["Window Size"] +  # Smaller window size is less complex
        row["Top N Individual"] +  # Lower top N is less complex
        row["Top N Aggregated"] +  # Lower top N is less complex
        search_algo_complexity[row["Search Algorithm"]] +  # Sequential < Ordered < Random
        use_semantic_complexity +  # False < True
        row["Combine Threshold"]  # Lower thresholds are less complex
    )

# Hyperparameter ranges
window_size_range = [1, 2, 3]
combine_threshold_range = [0.9, 0.95, 0.996, 1.0]
search_algorithm_options = ["Sequential", "Ordered"]
use_semantic_options = [True, False]
top_n_individual_range = [2, 3]
top_n_aggregated_range = [2, 3]
lexical_algorithm_options = ["Sentcomp", "Keycomp"]

# Initialize session state for SECTOR results
if "sector_results" not in st.session_state:
    st.session_state.sector_results = None  # This will store the DataFrame with SECTOR results

# Initialize session state for sentence-level analysis
if "sentence_analysis" not in st.session_state:
    st.session_state["sentence_analysis"] = None

# Initialize session state for sentence-level analysis
if "sentence_analysis_all_rows" not in st.session_state:
    st.session_state["sentence_analysis_all_rows"] = None

 # Check if results exist in session state
if "threshold_results" not in st.session_state:
    st.session_state["threshold_results"] = None

# Check if results exist in session state
if "extractor_results" not in st.session_state:
    st.session_state["extractor_results"] = None

# Initialize session state for auto sector results
if "auto_sector_results" not in st.session_state:
    st.session_state["auto_sector_results"] = None
if "auto_sector_best_params" not in st.session_state:
    st.session_state["auto_sector_best_params"] = None

# Sidebar for file upload and SECTOR configuration
st.sidebar.header("Upload & Configure SECTOR")
uploaded_file = st.sidebar.file_uploader("Upload File (Excel or CSV)", type=["xlsx", "csv"])

if uploaded_file:
    data = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    #st.write("Data Preview", data.head())


    if not data.empty:

        # Configure AgGrid for enhanced interactivity with word wrapping
        gb = GridOptionsBuilder.from_dataframe(data)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_default_column(editable=False, filter=True, wrapText=True, autoHeight=True)  # Enable word wrap


        grid_options = gb.build()

        grid_response = AgGrid(
            data,
            gridOptions=grid_options,
            enable_enterprise_modules=True,
            theme="alpine",
            update_mode='MODEL_CHANGED'
        )

        data = pd.DataFrame(grid_response["data"])
        
    if st.button("Clear All Results"):
            st.session_state["threshold_results"] = None
            st.session_state["extractor_results"] = None
            st.session_state["sector_results"] = None
            st.session_state["sentence_analysis_all_rows"] = None
            st.session_state["sentence_analysis"] = None
            st.session_state["auto_sector_results"] = None
            st.session_state["auto_sector_best_params"] = None
    
    input_column = st.sidebar.selectbox("Select LLM Response Column", data.columns)
    reference_column = st.sidebar.selectbox("Select Context Document Column", data.columns)
    
    # SECTOR parameters
    window_size = st.sidebar.slider("Window Size", min_value=1, max_value=3, value=2)
    combine_threshold = st.sidebar.slider("Combine Threshold", min_value=0.9, max_value=1.0, value=0.996)
    search_algorithm = st.sidebar.selectbox("Search Algorithm", ["Sequential", "Ordered", "Random"])
    use_semantic = st.sidebar.checkbox("Use Semantic Matching", value=True)
    top_n_individual = st.sidebar.number_input("Top N Individual", min_value=1, max_value=4, value=2)
    top_n_aggregated = st.sidebar.number_input("Top N Aggregated", min_value=1, max_value=4, value=2)
    lexical_algo = st.sidebar.selectbox("Lexical Algorithm", ["Sentcomp", "Keycomp"])

    # Optional parameters for clean_fn and embed_fn (set to None by default)
    input_clean_fn = None
    context_clean_fn = None
    embed_fn = None
    


    # Create three tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Run SECTOR", "SECTOR Threshold Evaluator", "Combine Threshold Evaluator","Detailed Analysis","Auto SECTOR","Benchmark"])

    # Tab 1: Run SECTOR with AgGrid
    with tab1:
        st.header("Run SECTOR")

        if st.button("Run SECTOR"):
            # Run SECTOR analysis and store results in session state
            results = []
            for i in range(len(data)):
                input_text = data[input_column].iloc[i]
                reference_text = data[reference_column].iloc[i]
                
                # Assuming `run_sector` is the function you use to get results
                matched_sentences, final_score = run_sector(
                    input_text,
                    reference_text,
                    max_window_size=window_size,
                    use_semantic=use_semantic,
                    combine_threshold=combine_threshold,
                    top_n_individual=top_n_individual,
                    top_n_aggregated=top_n_aggregated,
                    debug=False,
                    search=search_algorithm.lower(),
                    lexical_algo=lexical_algo.lower(),
                    input_clean_fn=input_clean_fn,
                    context_clean_fn=context_clean_fn,
                    embed_fn=embed_fn
                )

                response = {
                    "Input Text": input_text,
                    "Reference Document": reference_text,
                    "Sector Context Similarity": final_score["sector_context_similarity"],
                    "Content Word Similarity": final_score["average_scores"]["content_word_similarity"],
                    "Ngram Fuzzy Match Score": final_score["average_scores"]["ngram_fuzzy_match_score"],
                    "Levenshtein Similarity": final_score["average_scores"]["levenshtein_similarity"],
                    "POS-Based Alignment Score": final_score["average_scores"]["pos_based_alignment_score"],
                    "Word Coverage": final_score["average_scores"]["word_coverage"],
                    "Key Word Coverage": final_score["average_scores"]["key_word_coverage"],
                    "Geometric Mean Top N": final_score["average_scores"]["geometric_mean_top_n"],
                    
                }
                results.append(response)
            
            st.session_state.sector_results = pd.DataFrame(results)  # Store in session state

        # Display results if available in session state
        if st.session_state.sector_results is not None:
            st.write("SECTOR Results")


            # Configure AgGrid for enhanced interactivity with word wrapping
            gb = GridOptionsBuilder.from_dataframe(st.session_state.sector_results)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_default_column(editable=False, filter=True, wrapText=True, autoHeight=True)  # Enable word wrap


            grid_options = gb.build()

            grid_response = AgGrid(
                st.session_state.sector_results,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                theme="alpine",
                update_mode='MODEL_CHANGED'
            )

            # Download results as Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                st.session_state.sector_results.to_excel(writer, index=False, sheet_name="SECTOR Results")
            output.seek(0)
            st.download_button(
                label="Download SECTOR Results as Excel",
                data=output,
                file_name="sector_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Threshold adjustment for Sector Context Similarity
            st.subheader("Threshold Adjustment")
            threshold = st.slider("Set Sector Context Similarity Threshold", min_value=0.0, max_value=1.0, value=0.5)

            # Calculate accuracy based on the threshold
            above_threshold = st.session_state.sector_results["Sector Context Similarity"] >= threshold
            accuracy = above_threshold.mean() * 100
            st.write(f"Accuracy based on threshold: {accuracy:.2f}%")

            # Filter the session state results for rows below the threshold
            errors_df = st.session_state.sector_results[
                st.session_state.sector_results["Sector Context Similarity"] < threshold
            ]

            # Display filtered rows
            st.write(f"**Rows with Sector Context Similarity below {threshold:.2f}:**")
            if not errors_df.empty:
                st.dataframe(errors_df)  # Display the filtered DataFrame
            else:
                st.success("No rows below the threshold!")

    # Tab 2: Threshold Determination
    with tab2:
        st.header("Threshold Determination")

        if st.button("Determine Threshold"):
            geometric_means = []

            for i in range(len(data)):
                input_text = data[input_column].iloc[i]
                reference_text = data[reference_column].iloc[i]
                
                similar_sentences_json = extract_similar_sentences(
                    reference_text,
                    input_text,
                    max_window_size=window_size,
                    use_semantic=use_semantic,
                    combine_threshold=combine_threshold,
                    debug=False,
                    search=search_algorithm.lower()
                )

                for sentence in similar_sentences_json:
                    input_sentence = sentence['input_sentence']
                    reference_sentence = sentence['best_match']
                    comparison_result = json.loads(compare_sentences_flexible(input_sentence, reference_sentence, top_n=top_n_individual))
                    geometric_mean = comparison_result["scores"]["geometric_mean_top_n"]
                    geometric_means.append(geometric_mean)

            # Compute summary statistics
            statistics = compute_summary_statistics(geometric_means)
            #st.write("Summary Statistics", statistics)

            # Calculate cutoffs using Percentile and GMM methods
            percentile_cutoff = determine_cutoff_with_percentile(geometric_means, percentile=75)
            gmm_cutoff, gmm_means = determine_cutoff_with_gmm(geometric_means)

            #st.write(f"75th Percentile Cutoff: {percentile_cutoff}")
            #st.write(f"GMM Cutoff: {gmm_cutoff}")
            #st.write(f"GMM Means: {gmm_means}")

            # Store results in session state
            st.session_state["threshold_results"] = {
                "statistics": statistics,
                "percentile_cutoff": percentile_cutoff,
                "gmm_cutoff": gmm_cutoff,
                "gmm_means": gmm_means
            }

        # Display stored results if available
        if st.session_state["threshold_results"]:
            results = st.session_state["threshold_results"]
            st.write("**Summary Statistics**", results["statistics"])
            st.write(f"**75th Percentile Cutoff**: {results['percentile_cutoff']}")
            st.write(f"**GMM Cutoff**: {results['gmm_cutoff']}")
            st.write(f"**GMM Means**: {results['gmm_means']}")        

    # Tab 3: Extractor Threshold
    with tab3:
        st.header("Evaluate Extractor Thresholds")

        if st.button("Run Extractor Threshold Evaluation"):
            input_text_list = data[input_column].tolist()
            reference_doc_list = data[reference_column].tolist()
            positive_list = []
            negative_list = []

            for i, input_text in enumerate(input_text_list):
                for j, reference_doc in enumerate(reference_doc_list):
                    similar_sentences_json = extract_similar_sentences(
                        reference_doc,
                        input_text,
                        max_window_size=4,
                        use_semantic=use_semantic,
                        combine_threshold=combine_threshold,
                        debug=False,
                        search=search_algorithm.lower(),
                        lexical_algo=lexical_algo.lower()

                    )

                    for sentence in similar_sentences_json:
                        probability = sentence['best_score']
                        if i == j:
                            positive_list.append(probability)
                        else:
                            negative_list.append(probability)

            # Calculate statistics
            positive_stats = calculate_statistics(positive_list)
            negative_stats = calculate_statistics(negative_list)

            # Display statistics
            #st.write("Positive Matches Statistics:", positive_stats)
            #st.write("Negative Matches Statistics:", negative_stats)

           # Store results in session state
            st.session_state["extractor_results"] = {
                "positive_stats": positive_stats,
                "negative_stats": negative_stats
            }

        # Display stored results if available
        if st.session_state["extractor_results"]:
            results = st.session_state["extractor_results"]
            #st.write("**Positive Matches Statistics**", results["positive_stats"])
            #st.write("**Negative Matches Statistics**", results["negative_stats"])

            # Display results in a DataFrame format
            stats_df = pd.DataFrame({
                "Statistics": list(results["positive_stats"].keys()),
                "Positive Scores": list(results["positive_stats"].values()),
                "Negative Scores": list(results["negative_stats"].values())
            })
            st.dataframe(stats_df)

    # Tab 4: Sentence-Level Analysis
    with tab4:
        st.header("Sentence-Level Analysis")
        
        data = data.reset_index(drop=True)
        # Row selection for individual analysis
        selected_row_index = st.selectbox("Select Row for Analysis", options=data.index, format_func=lambda x: f"Row {x + 1}")

        # Display selected row's input and reference text
        input_text = data[input_column].iloc[selected_row_index]
        reference_text = data[reference_column].iloc[selected_row_index]

        st.subheader(f"Selected Row: {selected_row_index + 1}")
        st.write(f"**Input Text:** {input_text}")
        st.write(f"**Reference Text:** {reference_text}")

        # Run SECTOR for the selected row
        if st.button("Run Analysis for Selected Row"):
            matched_sentences, final_score = run_sector(
                input_text,
                reference_text,
                max_window_size=window_size,  # Use value from sidebar
                use_semantic=use_semantic,  # Use value from sidebar
                combine_threshold=combine_threshold,  # Use value from sidebar
                top_n_individual=top_n_individual,  # Use value from sidebar
                top_n_aggregated=top_n_aggregated,  # Use value from sidebar
                debug=False,
                search=search_algorithm.lower(),  # Use value from sidebar
                lexical_algo=lexical_algo.lower(),
                input_clean_fn=input_clean_fn,
                context_clean_fn=context_clean_fn,
                embed_fn=None
            )

            # Process matched sentences into a DataFrame
            sentence_results = [
                {
                    "Input Sentence": match["input_sentence"],
                    "Reference Document": reference_text,
                    "Matched Reference Sentence": match["matched_reference"],
                    "Sentence Similarity Score": match["scores"]["geometric_mean_top_n"],
                }
                for match in matched_sentences
            ]
            sentence_df = pd.DataFrame(sentence_results)

            # Display sentence-level results
            st.subheader(f"Sentence-Level Matches for Row {selected_row_index + 1}")
            if not sentence_df.empty:
                # Configure AgGrid for interactive analysis
                gb = GridOptionsBuilder.from_dataframe(sentence_df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_side_bar()
                gb.configure_default_column(editable=False, filter=True, wrapText=True, autoHeight=True)
                grid_options = gb.build()

                AgGrid(
                    sentence_df,
                    gridOptions=grid_options,
                    enable_enterprise_modules=True,
                    theme="alpine",
                    height=400,
                    update_mode='MODEL_CHANGED'
                )
            else:
                st.warning("No matched sentences found for the selected row.")

            # Allow downloading the results for the selected row
            output = io.BytesIO()
            sentence_df.to_csv(output, index=False)
            output.seek(0)
            st.download_button(
                label=f"Download Analysis for Row {selected_row_index + 1}",
                data=output,
                file_name=f"sentence_analysis_row_{selected_row_index + 1}.csv",
                mime="text/csv"
            )

        # Run All and Download
        if st.button("Run Analysis for All Rows"):
            all_sentence_results = []

            for idx in data.index:
                input_text = data[input_column].iloc[idx]
                reference_text = data[reference_column].iloc[idx]
                matched_sentences, _ = run_sector(
                    input_text,
                    reference_text,
                    max_window_size=window_size,  # Use value from sidebar
                    use_semantic=use_semantic,  # Use value from sidebar
                    combine_threshold=combine_threshold,  # Use value from sidebar
                    top_n_individual=top_n_individual,  # Use value from sidebar
                    top_n_aggregated=top_n_aggregated,  # Use value from sidebar
                    debug=False,
                    search=search_algorithm.lower(),  # Use value from sidebar
                    lexical_algo=lexical_algo.lower(),
                    input_clean_fn=input_clean_fn,
                    context_clean_fn=context_clean_fn,
                    embed_fn=None
                )
                for match in matched_sentences:
                    all_sentence_results.append({
                        "Row": idx + 1,
                        "Input Sentence": match["input_sentence"],
                        "Reference Document": reference_text,
                        "Matched Reference Sentence": match["matched_reference"],
                        "Sentence Similarity Score": match["scores"]["geometric_mean_top_n"],
                    })

            # Store results in session state
            st.session_state["sentence_analysis_all_rows"] = pd.DataFrame(all_sentence_results)

        # Display aggregated results if available
        if st.session_state["sentence_analysis_all_rows"] is not None:
            st.subheader("Sentence-Level Matches for All Rows")
            aggregated_df = st.session_state["sentence_analysis_all_rows"]

            # Configure AgGrid for interactive analysis
            gb = GridOptionsBuilder.from_dataframe(aggregated_df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_default_column(editable=False, filter=True, wrapText=True, autoHeight=True)
            grid_options = gb.build()

            AgGrid(
                aggregated_df,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                theme="alpine",
                height=400,
                update_mode='MODEL_CHANGED'
            )
                
            all_sentence_df = pd.DataFrame(st.session_state["sentence_analysis_all_rows"])

            # Allow downloading the aggregated results
            output_all = io.BytesIO()
            all_sentence_df.to_csv(output_all, index=False)
            output_all.seek(0)
            st.download_button(
                label="Download Full Sentence-Level Analysis",
                data=output_all,
                file_name="sentence_level_analysis_all_rows.csv",
                mime="text/csv"
            )

    with tab5:
        st.header("Auto Sector: Randomized Hyperparameter Search")

        # Prepare parameter grid
        param_grid = list(product(
            window_size_range,
            combine_threshold_range,
            search_algorithm_options,
            use_semantic_options,
            top_n_individual_range,
            top_n_aggregated_range,
            lexical_algorithm_options
        ))

        #if st.session_state["sector_results"] is not None:
        # Configure settings
        max_trials = st.slider("Max Trials to Run", min_value=50, max_value=len(param_grid), value=50, step=1)
        sample_rows = st.slider("Number of Rows to Sample from Input", min_value=1, max_value=len(data), value=5, step=1)

        if st.button("Run Randomized Search"):
            # Sample rows
            sampled_data = data.sample(n=sample_rows, random_state=42)

            # Shuffle and limit the parameter grid for randomized search
            random.shuffle(param_grid)
            param_grid = param_grid[:max_trials]  # Limit to max_trials

            # Initialize progress display
            progress_placeholder = st.empty()
            progress_bar = st.progress(0)

            # Collect results
            hyperparameter_results = []
            for i, params in enumerate(param_grid):
                (window_size, combine_threshold, search_algorithm, use_semantic, 
                top_n_individual, top_n_aggregated, lexical_algo) = params

                # Ignore lexical_algorithm if use_semantic is True
                if use_semantic:
                    lexical_algo = None

                # Display progress
                progress_percentage = ((i + 1) / max_trials) * 100
                current_trial_info = f"""
                **Trial {i + 1}/{max_trials} ({progress_percentage:.2f}% Completed)**  
                - **Window Size**: {window_size}  
                - **Combine Threshold**: {combine_threshold}  
                - **Search Algorithm**: {search_algorithm}  
                - **Use Semantic**: {use_semantic}  
                - **Top N Individual**: {top_n_individual}  
                - **Top N Aggregated**: {top_n_aggregated}  
                - **Lexical Algorithm**: {lexical_algo if not use_semantic else 'N/A'}  
                """
                progress_placeholder.markdown(current_trial_info)

                # Simulate running Sector for all sampled rows
                total_score = 0
                for _, row in sampled_data.iterrows():
                    input_text = row[input_column]
                    reference_text = row[reference_column]

                    # Run SECTOR with the given parameters
                    matched_sentences, final_score = run_sector(
                        input_text,
                        reference_text,
                        max_window_size=window_size,
                        use_semantic=use_semantic,
                        combine_threshold=combine_threshold,
                        top_n_individual=top_n_individual,
                        top_n_aggregated=top_n_aggregated,
                        debug=False,
                        search=search_algorithm.lower(),
                        lexical_algo=lexical_algo.lower() if lexical_algo else None,
                        input_clean_fn=input_clean_fn,
                        context_clean_fn=context_clean_fn,
                        embed_fn=None
                    )

                    # Extract sector context similarity score
                    total_score += final_score["sector_context_similarity"]

                # Compute average score for the parameter combination
                avg_score = total_score / len(sampled_data)

                # Store results
                hyperparameter_results.append({
                    "Window Size": window_size,
                    "Combine Threshold": combine_threshold,
                    "Search Algorithm": search_algorithm,
                    "Use Semantic": use_semantic,
                    "Top N Individual": top_n_individual,
                    "Top N Aggregated": top_n_aggregated,
                    "Lexical Algorithm": lexical_algo if not use_semantic else "N/A",
                    "Average Score": avg_score
                })

                # Update progress bar
                progress_bar.progress((i + 1) / max_trials)

                # Stop if max_trials is reached
                if i + 1 == max_trials:
                    break

            # Clear progress display after completion
            progress_placeholder.empty()
            progress_bar.empty()

            # Convert results to DataFrame
            results_df = pd.DataFrame(hyperparameter_results)
            st.session_state["auto_sector_results"] = results_df

            # Display best parameters
            #best_params = results_df.loc[results_df["Average Score"].idxmax()]
            #st.subheader("Best Parameters")
            #st.json(best_params.to_dict())

            if not results_df.empty:
                # Add complexity to results DataFrame
                results_df["Complexity"] = results_df.apply(compute_complexity, axis=1)

                # Find the best parameters based on the score and complexity
                max_score = results_df["Average Score"].max()
                best_params = results_df[results_df["Average Score"] == max_score].sort_values("Complexity").iloc[0]
                st.session_state["auto_sector_best_params"] = best_params
                #st.subheader("Best Parameters:")
                #st.json(best_params.to_dict())

        # Display results if available
        if st.session_state["auto_sector_results"] is not None:
            st.subheader("Best Parameters")
            st.json(st.session_state["auto_sector_best_params"].to_dict())

            st.subheader("All Parameter Combinations and Scores")
            st.dataframe(st.session_state["auto_sector_results"])

            # Allow download
            csv = st.session_state["auto_sector_results"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Hyperparameter Search Results",
                data=csv,
                file_name="hyperparameter_search_results.csv",
                mime="text/csv"
            )
   # Tab 6: Benchmarking sector with external metrics
    with tab6:
        st.header("Benchmarking Sector")

        if st.button("Run Benchmark"):
            input_text_list = data[input_column].tolist()
            reference_doc_list = data[reference_column].tolist()

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

                similar_sentences_json = extract_similar_sentences(
                    reference_doc_list[docs],
                    input_text_list[docs],
                    max_window_size=4,
                    use_semantic=use_semantic,
                    combine_threshold=combine_threshold,
                    debug=False,
                    search=search_algorithm.lower(),
                    lexical_algo=lexical_algo.lower()
                )

                score_json_nosector = get_scores_as_json(reference_doc_list[docs], input_text_list[docs])

                # Output the matched sentences in JSON format
                for sentence in similar_sentences_json:
                    input_sentence = sentence['input_sentence']
                    reference_sentence = sentence['best_match']

                    scores_json_sector = get_scores_as_json(reference_sentence, input_sentence)

                    combined_scores.append(scores_json_sector)


                num_entries = len(combined_scores)
                for scores in combined_scores:
                    for key in score_sums:
                        score_sums[key] += scores[key]
                
                # Calculate average scores
                average_scores = {key: score_sums[key] / num_entries for key in score_sums}

                st.write("LLM Response : "+ input_text_list[docs])
                st.write("Context Document : "+ reference_doc_list[docs])

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
                for metric, without_sector_value in score_json_nosector.items():
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
