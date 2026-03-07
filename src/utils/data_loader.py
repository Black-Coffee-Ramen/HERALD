# src/utils/data_loader.py
import pandas as pd
import os

def load_training_data(data_dir: str = "data/raw/PS02_Training_set"):
    """
    Load the PS02 training dataset using actual column names from the Excel file.
    """
    excel_path = os.path.join(data_dir, "PS02_Training_set.xlsx")
    df = pd.read_excel(excel_path)

    print("Columns in training data:", df.columns.tolist())

    # Map to standard names for easier handling
    df = df.rename(columns={
        'Identified Phishing/Suspected Domain Name': 'domain',
        'Critical Sector Entity Name': 'cse_name',
        'Corresponding CSE Domain Name': 'cse_domain',
        'Phishing/Suspected Domains (i.e. Class Label)': 'label',
        'Evidence file name': 'evidence_filename',
        'Source of detection': 'source'
    })

    print("\nLabel distribution:")
    print(df['label'].value_counts())

    # Construct full evidence path
    evidence_dir = os.path.join(data_dir, "Evidences")
    df['evidence_path'] = df['evidence_filename'].apply(
        lambda fname: os.path.join(evidence_dir, fname) if pd.notna(fname) else None
    )

    # Check if evidence files exist
    df['evidence_exists'] = df['evidence_path'].apply(
        lambda p: os.path.exists(p) if pd.notna(p) else False
    )
    print(f"\nEvidence files found: {df['evidence_exists'].sum()}/{len(df)}")

    return df

if __name__ == "__main__":
    df = load_training_data()
    print("\nSample rows:")
    print(df[['domain', 'cse_name', 'label']].head())