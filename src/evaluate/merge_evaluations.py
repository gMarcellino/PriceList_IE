import os
from argparse import ArgumentParser
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--evaluation_paths", nargs='+', required=True, help="Paths to evaluation CSV files to merge.")
parser.add_argument("--output_path", required=True, help="Path to save the merged evaluation CSV file.")
parser.add_argument("--sort_column", default="strict_micro_f1")

def merge_evaluations(evaluation_paths, output_path, sort_column=None, sort_ascending=False):
    merged_df = pd.DataFrame()

    for path in evaluation_paths:
        df = pd.read_csv(path)
        folder_name = os.path.basename(os.path.dirname(path))
        df.insert(0, "source_folder", folder_name)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    if sort_column:
        merged_df = merged_df.sort_values(by=sort_column, ascending=sort_ascending)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    args = parser.parse_args()
    merge_evaluations(args.evaluation_paths, args.output_path, sort_column=args.sort_column, sort_ascending=False)