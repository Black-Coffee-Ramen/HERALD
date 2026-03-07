import pandas as pd
from pathlib import Path

# Paths to your Excel files
raw_path = Path("data/raw/PS-02_Shortlisting_set")
processed_path = Path("data/processed")

# List of files to inspect
files_to_inspect = {
    "Shortlisting Part 1": raw_path / "Shortlisting_Data_Part_1.xlsx",
    "Shortlisting Part 2": raw_path / "Shortlisting_Data_Part_2.xlsx",
    "Processed Stage 1 Dataset": processed_path / "PS-02  Phishing Detection CSE_Domains_Dataset_for_Stage_1.xlsx"
}

def inspect_files(files_dict):
    for name, path in files_dict.items():
        print(f"\n{name} - Columns:")
        if path.exists():
            try:
                df = pd.read_excel(path, dtype=str)  # force all columns to string
                print(df.columns.tolist())
                print("First 5 rows:")
                print(df.head(5))
            except Exception as e:
                print(f"Could not read {path.name}: {e}")
        else:
            print(f"File does not exist: {path}")

if __name__ == "__main__":
    inspect_files(files_to_inspect)
