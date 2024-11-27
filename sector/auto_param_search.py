import pandas as pd
import random
from itertools import product
from sector.sector_main import run_sector


def auto_sector(
    file_path,
    rows_to_sample=None,
    max_trials=None,
    verbose=True
):
    """
    Perform hyperparameter search for SECTOR.

    Args:
        file_path (str): Path to the CSV or Excel file containing input data.
        rows_to_sample (int): Number of rows to sample from the input data. If None, use all rows.
        max_trials (int): Maximum number of trials for hyperparameter search. If None, run all combinations.
        verbose (bool): If True, print progress updates to the terminal.

    Returns:
        pd.DataFrame: DataFrame of all hyperparameter combinations and scores.
    """
    # Load the data
    if file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")

    # Define hyperparameter ranges
    window_size_range = [1, 2, 3]
    combine_threshold_range = [0.95, 0.996]
    search_algorithm_options = ["Sequential", "Ordered"]
    use_semantic_options = [True, False]
    top_n_individual_range = [1, 2]
    top_n_aggregated_range = [1, 2]
    lexical_algorithm_options = ["Sentcomp", "Keycomp"]

    # Calculate total parameter combinations
    param_grid = list(product(
        window_size_range,
        combine_threshold_range,
        search_algorithm_options,
        use_semantic_options,
        top_n_individual_range,
        top_n_aggregated_range,
        lexical_algorithm_options
    ))
    total_combinations = len(param_grid)
    print(f"\nTotal possible parameter combinations: {total_combinations}")

    # Limit parameter grid to max_trials if specified
    if max_trials:
        random.shuffle(param_grid)  # Randomize order for limited trials
        param_grid = param_grid[:max_trials]

    # Show data preview
    print("Data loaded successfully!")
    print(f"Columns (with indices):")
    for idx, col in enumerate(data.columns):
        print(f"  [{idx}] {col}")
    print("\nFirst 5 rows:")
    print(data.head())

    # Select columns by index or name
    def get_column(data, prompt):
        user_input = input(prompt)
        if user_input.isdigit():
            index = int(user_input)
            if index >= len(data.columns):
                raise ValueError("Invalid column index. Please check the column list.")
            return data.columns[index]
        elif user_input in data.columns:
            return user_input
        else:
            raise ValueError("Invalid column name. Please check the column list.")

    input_col = get_column(data, "\nEnter the column index or name for input text (LLM Response data): ")
    reference_col = get_column(data, "Enter the column index or name for reference text (Context data): ")

    print(f"\nSelected columns: Input Text -> {input_col}, Reference Text -> {reference_col}")

    # Sample rows if requested
    if rows_to_sample:
        data = data.sample(n=min(rows_to_sample, len(data)), random_state=42)

    # Initialize results
    results = []

    # Perform hyperparameter search
    print("\nStarting hyperparameter search...")
    for i, params in enumerate(param_grid):
        (window_size, combine_threshold, search_algorithm, use_semantic, 
         top_n_individual, top_n_aggregated, lexical_algo) = params

        # Ignore lexical_algorithm if use_semantic is True
        if use_semantic:
            lexical_algo = None

        # Display trial progress
        if verbose:
            print(f"Trial {i + 1}/{min(max_trials, total_combinations) if max_trials else total_combinations}")
            print(f"Parameters: Window Size={window_size}, Combine Threshold={combine_threshold}, "
                  f"Search Algorithm={search_algorithm}, Use Semantic={use_semantic}, "
                  f"Top N Individual={top_n_individual}, Top N Aggregated={top_n_aggregated}, "
                  f"Lexical Algorithm={lexical_algo if not use_semantic else 'N/A'}")

        # Compute total score for sampled rows
        total_score = 0
        for _, row in data.iterrows():
            input_text = row[input_col]
            reference_text = row[reference_col]

            # Run SECTOR
            _, final_score = run_sector(
                input_text,
                reference_text,
                max_window_size=window_size,
                use_semantic=use_semantic,
                combine_threshold=combine_threshold,
                top_n_individual=top_n_individual,
                top_n_aggregated=top_n_aggregated,
                debug=False,
                search=search_algorithm.lower(),
                input_clean_fn=None,
                context_clean_fn=None,
                embed_fn=None,
                lexical_algo=lexical_algo.lower() if lexical_algo else None

            )
            total_score += final_score["sector_context_similarity"]

        # Calculate average score
        avg_score = total_score / len(data)

        # Store results
        results.append({
            "Window Size": window_size,
            "Combine Threshold": combine_threshold,
            "Search Algorithm": search_algorithm,
            "Use Semantic": use_semantic,
            "Top N Individual": top_n_individual,
            "Top N Aggregated": top_n_aggregated,
            "Lexical Algorithm": lexical_algo if not use_semantic else None,
            "Average Score": avg_score
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Output best parameters
    best_params = results_df.loc[results_df["Average Score"].idxmax()]
    print("\nBest Parameters:")
    print(best_params)

    # Save results to CSV
    results_df.to_csv("hyperparameter_search_results.csv", index=False)
    print("\nResults saved to 'hyperparameter_search_results.csv'.")

    return results_df


# Main Program
if __name__ == "__main__":
    file_path = input("Enter the path to your CSV or Excel file: ")
    rows_to_sample = input("Enter the number of rows to sample (or press Enter for all rows): ")
    max_trials = input("Enter the maximum number of trials (or press Enter to run all combinations): ")

    rows_to_sample = int(rows_to_sample) if rows_to_sample.strip() else 0
    max_trials = int(max_trials) if max_trials.strip() else None

    results = auto_sector(
        file_path=file_path,
        rows_to_sample=rows_to_sample if rows_to_sample > 0 else None,
        max_trials=max_trials
    )
