"""
scripts/build_dataset_v8.py

Rebalances the HERALD v7 dataset to a 1:1 ratio.
- Loads data/processed/full_dataset_v7.csv
- Keeps all phishing samples.
- Samples an equal number of legitimate samples.
"""

import os
import pandas as pd

def main():
    print("=" * 50)
    print("HERALD v8 Dataset Rebalancer")
    print("=" * 50)
    
    input_path = 'data/processed/full_dataset_v7.csv'
    output_path = 'data/processed/full_dataset_v8.csv'
    
    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} domains from v7 dataset.")

    # Split Phishing and Legitimate
    phish_df = df[df['label'] == 'Phishing']
    legit_df = df[df['label'] == 'Legitimate']
    suspect_df = df[df['label'] == 'Suspected'] if 'Suspected' in df['label'].values else pd.DataFrame()
    
    n_phish = len(phish_df)
    print(f"Found {n_phish} phishing samples.")
    print(f"Found {len(legit_df)} legitimate samples.")

    # Rebalance: Sample Legitimate to match Phishing count
    legit_sampled = legit_df.sample(n=n_phish, random_state=42)
    print(f"Sampled {len(legit_sampled)} legitimate domains.")

    # Combine back
    v8_df = pd.concat([phish_df, legit_sampled, suspect_df], ignore_index=True)
    
    # Shuffle
    v8_df = v8_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal v8 Class Distribution:")
    print(v8_df['label'].value_counts())
    print(f"Total: {len(v8_df)}")

    os.makedirs('data/processed', exist_ok=True)
    v8_df.to_csv(output_path, index=False)
    print(f"\nFinal dataset saved to {output_path}")

if __name__ == '__main__':
    main()
