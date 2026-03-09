"""
scripts/extract_features_v4.py

Runs the v3 lexical feature pipeline over the full_dataset_v4.csv.

Input:  data/processed/full_dataset_v4.csv
Output: data/processed/full_features_v4.csv
Errors: outputs/feature_extraction_errors_v4.txt
"""

import os
import sys
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herald.features.lexical_features import extract_url_features

BATCH_SIZE = 500


def main():
    print("=" * 60)
    print("Feature Extraction v4")
    print("=" * 60)

    in_path = 'data/processed/full_dataset_v4.csv'
    out_path = 'data/processed/full_features_v4.csv'
    err_path = 'outputs/feature_extraction_errors_v4.txt'

    os.makedirs('outputs', exist_ok=True)

    df = pd.read_csv(in_path)
    print(f"Loaded {len(df)} rows from {in_path}")

    error_rows = []
    result_frames = []
    total = len(df)

    for start in range(0, total, BATCH_SIZE):
        batch = df.iloc[start:start + BATCH_SIZE].copy()
        try:
            feat_batch = extract_url_features(batch.reset_index(drop=True), domain_col='domain')
            result_frames.append(feat_batch)
        except Exception as e:
            # Fallback: process row-by-row to isolate failures
            for _, row in batch.iterrows():
                try:
                    row_df = pd.DataFrame([{'domain': row['domain'], 'label': row['label'], 'source': row.get('source', '')}])
                    feat_row = extract_url_features(row_df, domain_col='domain')
                    result_frames.append(feat_row)
                except Exception as row_e:
                    error_rows.append({'domain': row['domain'], 'error': str(row_e)})

        pct = min(start + BATCH_SIZE, total)
        print(f"  Processed {pct}/{total} domains...")

    print(f"\nFeature extraction complete.")
    print(f"  Successful: {total - len(error_rows)}")
    print(f"  Failed:     {len(error_rows)}")

    if error_rows:
        err_df = pd.DataFrame(error_rows)
        err_df.to_csv(err_path, index=False)
        print(f"  Errors logged to {err_path}")

    if result_frames:
        df_features = pd.concat(result_frames, ignore_index=True)
        # Drop object cols that aren't the label/domain/source
        for col in df_features.columns:
            if df_features[col].dtype == 'object' and col not in ('domain', 'label', 'source'):
                df_features[col] = df_features[col].astype('category').cat.codes
        df_features.to_csv(out_path, index=False)
        print(f"\nSaved {len(df_features)} rows with {len(df_features.columns)} columns to {out_path}")
    else:
        print("ERROR: No features extracted. Check inputs.")


if __name__ == '__main__':
    main()
