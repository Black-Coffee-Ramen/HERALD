# src/utils/post_validation.py
import os
import pandas as pd
from src.features.visual_similarity import EnhancedVisualSimilarity

class PostPredictionValidator:
    """
    Post-prediction validation to downgrade predictions without visual evidence
    """
    
    def __init__(self, evidence_dir="evidence"):
        self.evidence_dir = evidence_dir
        self.visual_similarity = EnhancedVisualSimilarity()
    
    def has_visual_evidence(self, domain, cse_name=None, min_similarity=0.7):
        """
        Check if domain has visual evidence supporting phishing classification
        """
        screenshot_path = f"{self.evidence_dir}/{domain}/screenshot.png"
        
        # Check if screenshot exists
        if not os.path.exists(screenshot_path):
            return False
        
        # Check for form/input fields
        has_login_elements = self.has_login_elements(screenshot_path)
        if not has_login_elements:
            return False
        
        # Check visual similarity if CSE is provided
        if cse_name and cse_name != 'Unknown':
            visual_score, has_high_similarity, _ = \
                self.visual_similarity.calculate_enhanced_similarity(screenshot_path, cse_name)
            return has_high_similarity and visual_score >= min_similarity
        
        return has_login_elements
    
    def has_login_elements(self, screenshot_path):
        """Check if page contains login form elements"""
        try:
            # Simple check using OCR and image analysis
            from src.core.content_classifier import ContentClassifier
            classifier = ContentClassifier()
            ui_elements = classifier.extract_ui_elements(screenshot_path)
            
            return (ui_elements['login_form'] and 
                   (ui_elements['password_field'] or ui_elements['username_field']))
        except:
            return False
    
    def downgrade_prediction(self, domain, current_prediction, current_confidence, cse_name=None):
        """
        Downgrade prediction if visual evidence is lacking
        """
        if current_prediction == "Phishing":
            if not self.has_visual_evidence(domain, cse_name):
                print(f"🔧 Downgrading {domain} from Phishing to Suspected (no visual evidence)")
                return "Suspected", current_confidence * 0.6
                
        return current_prediction, current_confidence
    
    def validate_batch_predictions(self, df_predictions):
        """
        Apply post-validation to batch predictions
        """
        print("🔍 Applying post-prediction validation...")
        
        df_validated = df_predictions.copy()
        
        for idx, row in df_validated.iterrows():
            if row['final_label'] == 'Phishing':
                domain = row['domain']
                cse_name = row.get('target_cse', 'Unknown')
                confidence = row.get('final_confidence', row.get('confidence', 0.5))
                
                new_label, new_confidence = self.downgrade_prediction(
                    domain, 'Phishing', confidence, cse_name
                )
                
                if new_label != 'Phishing':
                    df_validated.at[idx, 'final_label'] = new_label
                    df_validated.at[idx, 'final_confidence'] = new_confidence
                    df_validated.at[idx, 'validation_notes'] = 'Downgraded - no visual evidence'
        
        # Statistics
        original_phishing = len(df_predictions[df_predictions['final_label'] == 'Phishing'])
        final_phishing = len(df_validated[df_validated['final_label'] == 'Phishing'])
        
        print(f"📊 Post-validation results:")
        print(f"   - Original Phishing: {original_phishing}")
        print(f"   - Final Phishing: {final_phishing}")
        print(f"   - Downgraded: {original_phishing - final_phishing}")
        
        return df_validated