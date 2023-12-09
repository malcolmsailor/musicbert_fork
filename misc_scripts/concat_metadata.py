import os
import sys

import pandas as pd


def process_csv_files(input_directory, output_file):
    # Get a list of all CSV files in the input directory
    csv_files = [file for file in os.listdir(input_directory) if file.endswith(".csv")]

    # Initialize an empty DataFrame for the final result
    final_df = pd.DataFrame()

    # Iterate over each CSV file
    for count, file in enumerate(csv_files, start=0):
        # Construct the full file path
        file_path = os.path.join(input_directory, file)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path, index_col=0)

        if count == 0:
            final_df.index = df.index
        else:
            df = df.reindex(index=final_df.index)
            assert all(final_df.index == df.index)

        # Select a subset of columns (placeholder names)
        columns_to_select = ["Num test examples", "Num test unique scores"]
        subset_df = df[columns_to_select]

        # Rename columns to make them unique
        subset_df.columns = [f"{col}_{count}" for col in subset_df.columns]

        # Append the subset DataFrame to the final DataFrame
        final_df = pd.concat([final_df, subset_df], axis=1)

    # Write the final DataFrame to a new CSV file
    final_df.to_csv(output_file)


# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_file>")
        sys.exit(1)

    input_directory = sys.argv[1]
    output_file = sys.argv[2]

    process_csv_files(input_directory, output_file)
