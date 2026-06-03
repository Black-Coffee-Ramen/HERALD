import pandas as pd
from pathlib import Path

# Paths to your Excel files
processed_path = Path("data/processed")

# List of files to inspect
files_to_inspect = {
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
