"""
scripts/test_v7_content_boost.py

Verifies the Stage 2 Content Analysis boost on borderline domains.
Benchmarks against v5/v6 false negatives.
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from herald.predict_with_fallback import PhishingPredictorV3

def test_borderline():
    print("=" * 60)
    print("TESTING BORDERLINE DOMAINS (Stage 2 Content Analysis)")
    print("=" * 60)
    
    predictor = PhishingPredictorV3()
    domains = [
        ("irctc-refund-claim.xyz", "IRCTC"),
        ("icici-paylink.xyz", "ICICI")
    ]
    
    for domain, brand in domains:
        print(f"\nAnalyzing: {domain} (Target: {brand})")
        res = predictor.predict(domain, cse_name=brand)
        
        orig_score = res.get('ml_confidence', 0)
        adj_score = res.get('ml_confidence_adjusted', orig_score)
        status = res.get('status')
        type_ = res.get('analysis_type')
        
        print(f"Original ML Score: {orig_score:.4f}")
        print(f"Adjusted Score:    {adj_score:.4f}")
        print(f"Final Status:      {status}")
        print(f"Analysis Type:     {type_}")
        
        if 'content_features' in res:
            feats = res['content_features']
            triggered = [k for k, v in feats.items() if v and v != -1 and v != 0]
            print(f"Content Features Detected: {triggered}")
        else:
            print("Content Analysis: Not triggered (Score outside 0.35 - 0.65)")

def benchmark_fn():
    print("\n" + "=" * 60)
    print("BENCHMARKING AGAINST FALSE NEGATIVES (Top 20)")
    print("=" * 60)
    
    fn_path = 'outputs/v5_false_negatives.csv'
    if not os.path.exists(fn_path):
        print(f"ERROR: {fn_path} not found.")
        return

    df_fn = pd.read_csv(fn_path)
    predictor = PhishingPredictorV3()
    caught_count = 0
    
    results = []
    
    sample_df = df_fn.head(20)
    
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        domain = row['domain']
        res = predictor.predict(domain)
        
        orig = res.get('ml_confidence', 0)
        adj = res.get('ml_confidence_adjusted', orig)
        
        # Check if it would be caught now
        # Note: predictor status is updated based on adj score >= threshold
        status = res['status']
        
        feats = res.get('content_features', {})
        triggered = [k for k, v in feats.items() if v and v != -1 and v != 0]
        
        results.append({
            'domain': domain,
            'orig_score': orig,
            'adj_score': adj,
            'status': status,
            'features': triggered
        })
        
        if status == 'Phishing':
            caught_count += 1
            
    print(f"\nResult: {caught_count} / {len(sample_df)} caught by Content Analysis.")
    print("\nSample Details:")
    print(f"{'Domain':30} | {'Score Change':20} | {'Status':15} | {'Triggers'}")
    print("-" * 100)
    for r in results:
        change = f"{r['orig_score']:.4f} -> {r['adj_score']:.4f}"
        print(f"{r['domain']:30} | {change:20} | {r['status']:15} | {r['features']}")

if __name__ == "__main__":
    test_borderline()
    benchmark_fn()
