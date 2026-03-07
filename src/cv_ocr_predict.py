# src/cv_ocr_predict.py
import pandas as pd
import os
import time
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.cv_ocr_analyzer import CVOCRAnalyzer

def run_cv_ocr_analysis():
    """Run standalone CV/OCR analysis on high-confidence domains"""
    print("🎯 STARTING CV/OCR ANALYSIS")
    print("=" * 60)
    
    # Check if predictions exist
    if not os.path.exists("outputs/enhanced_cse_predictions.csv"):
        print("❌ No enhanced predictions found. Run main prediction first.")
        return
    
    # Load predictions
    df_cse = pd.read_csv("outputs/enhanced_cse_predictions.csv")
    print(f"📊 Loaded {len(df_cse)} CSE-targeting domains")
    
    # Filter high-confidence domains
    high_conf_domains = df_cse[df_cse['confidence'] > 0.95].head(20)
    print(f"🔍 Analyzing {len(high_conf_domains)} high-confidence domains")
    
    # Initialize analyzer
    analyzer = CVOCRAnalyzer()
    
    # Initialize result columns
    result_columns = ['cv_ocr_confirmed', 'cv_ocr_status', 'final_confidence', 
                     'analysis_timestamp', 'phishing_indicators', 'visual_similarity']
    
    for col in result_columns:
        if col not in df_cse.columns:
            if col == 'cv_ocr_confirmed':
                df_cse[col] = False
            elif col == 'final_confidence':
                df_cse[col] = df_cse['confidence']
            else:
                df_cse[col] = ''
    
    # Analyze domains
    analyzed_count = 0
    confirmed_phishing = 0
    
    for idx, row in high_conf_domains.iterrows():
        domain = row['domain'].strip()
        cse_name = row['target_cse']
        confidence = row['confidence']
        
        print(f"\n{'='*50}")
        print(f"🔬 [{analyzed_count + 1}/{len(high_conf_domains)}] {domain}")
        print(f"🎯 Target: {cse_name}")
        print(f"📊 Confidence: {confidence:.4f}")
        print(f"{'='*50}")
        
        # Run analysis
        analysis_result = analyzer.analyze_domain(domain, cse_name, confidence)
        
        # Update dataframe
        for col in result_columns:
            df_cse.at[idx, col] = analysis_result[col]
        
        analyzed_count += 1
        if analysis_result['cv_ocr_confirmed']:
            confirmed_phishing += 1
        
        # Save progress every 2 domains
        if analyzed_count % 2 == 0:
            df_cse.to_csv("outputs/enhanced_cse_predictions_cv_verified.csv", index=False)
            print(f"💾 Progress saved: {analyzed_count} domains analyzed")
    
    # Final save
    df_cse.to_csv("outputs/enhanced_cse_predictions_cv_verified.csv", index=False)
    
    # Results summary
    print(f"\n{'='*60}")
    print("🎯 CV/OCR ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Domains analyzed: {analyzed_count}")
    print(f"🚨 Confirmed phishing: {confirmed_phishing}")
    print(f"✅ Clean domains: {analyzed_count - confirmed_phishing}")
    
    # Show confirmed domains
    confirmed = df_cse[df_cse['cv_ocr_confirmed'] == True]
    if len(confirmed) > 0:
        print(f"\n🔍 CONFIRMED PHISHING DOMAINS:")
        for _, row in confirmed.iterrows():
            print(f"   - {row['domain']} -> {row['target_cse']}")
    
    print(f"\n💾 Results saved to: outputs/enhanced_cse_predictions_cv_verified.csv")

if __name__ == "__main__":
    run_cv_ocr_analysis()