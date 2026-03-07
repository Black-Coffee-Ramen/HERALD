# src/core/visual_analyzer.py
import cv2
import imagehash
from PIL import Image
import numpy as np
import os

class VisualAnalyzer:
    def __init__(self):
        self.phash_threshold = 15
        self.template_threshold = 0.8
    
    def load_cse_templates(self, cse_name):
        """Load CSE reference templates - placeholder implementation"""
        # In production, this would load actual reference images
        # For now, return empty list
        return []
    
    def calculate_phash_distance(self, img1_path, img2_path):
        """Calculate perceptual hash distance between images"""
        try:
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path)
            
            # Resize to common size for consistent hashing
            img1 = img1.resize((64, 64))
            img2 = img2.resize((64, 64))
            
            hash1 = imagehash.phash(img1)
            hash2 = imagehash.phash(img2)
            
            return hash1 - hash2  # Hamming distance
        except Exception as e:
            print(f"Error calculating phash: {e}")
            return float('inf')
    
    def compare_with_cse_templates(self, screenshot_path, cse_name):
        """Compare screenshot with CSE reference templates"""
        cse_templates = self.load_cse_templates(cse_name)
        min_distance = float('inf')
        
        for template_path in cse_templates:
            if os.path.exists(template_path):
                distance = self.calculate_phash_distance(screenshot_path, template_path)
                min_distance = min(min_distance, distance)
        
        return {
            'min_distance': min_distance if min_distance != float('inf') else 100,
            'is_lookalike': min_distance <= self.phash_threshold,
            'templates_compared': len(cse_templates)
        }
    
    def detect_ui_elements(self, screenshot_path):
        """Detect UI elements like logos, forms, buttons"""
        try:
            img = cv2.imread(screenshot_path)
            if img is None:
                return {'forms_detected': False, 'logos_detected': False}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Simple form detection (looking for rectangles)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            form_rectangles = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                # Forms typically have specific aspect ratios
                if 1.5 < aspect_ratio < 5 and w > 100 and h > 30:
                    form_rectangles += 1
            
            return {
                'forms_detected': form_rectangles > 0,
                'form_count': form_rectangles,
                'image_size': img.shape
            }
        except Exception as e:
            print(f"UI element detection failed: {e}")
