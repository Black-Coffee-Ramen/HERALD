# src/features/enhanced_visual_similarity.py
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
import json

class EnhancedVisualSimilarity:
    """
    Enhanced visual similarity detection using multiple methods
    """
    
    def __init__(self, template_dir="data/templates", reference_dir="data/reference"):
        self.template_dir = template_dir
        self.reference_dir = reference_dir
        self.cse_templates = self.load_cse_templates()
        
    def load_cse_templates(self):
        """Load CSE reference templates with metadata"""
        templates = {}
        
        if not os.path.exists(self.reference_dir):
            print(f"⚠️  Reference directory not found: {self.reference_dir}")
            return templates
        
        # Load template metadata if exists
        metadata_file = os.path.join(self.reference_dir, "templates_metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                templates = json.load(f)
        else:
            # Auto-discover templates
            for file in os.listdir(self.reference_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    template_name = os.path.splitext(file)[0]
                    template_path = os.path.join(self.reference_dir, file)
                    templates[template_name] = {
                        'path': template_path,
                        'cse_name': self.extract_cse_from_filename(template_name),
                        'hash': self.calculate_perceptual_hash(template_path),
                        'features': self.extract_template_features(template_path)
                    }
            
            # Save metadata for future use
            with open(metadata_file, 'w') as f:
                json.dump(templates, f, indent=2)
        
        return templates
    
    def extract_cse_from_filename(self, filename):
        """Extract CSE name from template filename"""
        filename_lower = filename.lower()
        
        cse_mapping = {
            'sbi': 'State Bank of India (SBI)',
            'irctc': 'Indian Railway Catering and Tourism Corporation (IRCTC)',
            'icici': 'ICICI Bank',
            'hdfc': 'HDFC Bank',
            'pnb': 'Punjab National Bank (PNB)',
            'bob': 'Bank of Baroda (BOB)',
            'airtel': 'Airtel',
            'iocl': 'Indian Oil Corporation Limited (IOCL)',
            'nic': 'National Informatics Centre (NIC)'
        }
        
        for key, value in cse_mapping.items():
            if key in filename_lower:
                return value
        
        return 'Unknown'
    
    def calculate_perceptual_hash(self, image_path, hash_size=16):
        """Calculate perceptual hash using imagehash"""
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale and resize for consistent hashing
                img_gray = img.convert('L')
                return imagehash.phash(img_gray, hash_size=hash_size)
        except Exception as e:
            print(f"⚠️  Perceptual hash error for {image_path}: {e}")
            return None
    
    def extract_template_features(self, template_path):
        """Extract structural features from template"""
        try:
            img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {}
            
            # Resize for consistent feature extraction
            img = cv2.resize(img, (300, 200))
            
            # Calculate structural features
            edges = cv2.Canny(img, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            features = {
                'contour_count': len(contours),
                'edge_density': np.sum(edges > 0) / edges.size,
                'brightness_mean': np.mean(img),
                'brightness_std': np.std(img)
            }
            
            return features
        except Exception as e:
            print(f"⚠️  Feature extraction error for {template_path}: {e}")
            return {}
    
    def calculate_structural_similarity(self, img1_path, img2_path):
        """Calculate SSIM between two images"""
        try:
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Resize to same dimensions
            img1 = cv2.resize(img1, (300, 200))
            img2 = cv2.resize(img2, (300, 200))
            
            # Calculate SSIM
            similarity, _ = ssim(img1, img2, full=True)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"⚠️  SSIM calculation error: {e}")
            return 0.0
    
    def calculate_hash_similarity(self, hash1, hash2):
        """Calculate similarity between two perceptual hashes"""
        if hash1 is None or hash2 is None:
            return 0.0
        
        # Calculate Hamming distance and convert to similarity score
        hamming_distance = hash1 - hash2
        max_distance = len(hash1.hash) ** 2  # For hash_size x hash_size matrix
        similarity = 1 - (hamming_distance / max_distance)
        
        return max(0.0, min(1.0, similarity))
    
    def detect_cse_specific_elements(self, screenshot_path, cse_name):
        """Detect CSE-specific UI elements"""
        elements = {
            'login_form': False,
            'brand_logo': False,
            'color_scheme_match': False,
            'layout_structure_match': False
        }
        
        try:
            img = cv2.imread(screenshot_path)
            if img is None:
                return elements
            
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # CSE-specific color schemes
            cse_colors = {
                'State Bank of India (SBI)': {
                    'blue': [(100, 150, 50), (140, 255, 255)]  # SBI blue
                },
                'ICICI Bank': {
                    'blue': [(100, 150, 50), (140, 255, 255)],  # ICICI blue
                    'orange': [(10, 150, 150), (25, 255, 255)]  # ICICI orange
                },
                'HDFC Bank': {
                    'blue': [(100, 150, 50), (140, 255, 255)]  # HDFC blue
                }
            }
            
            # Check color scheme
            if cse_name in cse_colors:
                color_match_score = 0
                for color_name, color_range in cse_colors[cse_name].items():
                    lower, upper = color_range
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    color_ratio = np.sum(mask > 0) / mask.size
                    if color_ratio > 0.05:  # At least 5% of image has brand color
                        color_match_score += 1
                
                elements['color_scheme_match'] = color_match_score > 0
            
            # Simple logo detection (template matching)
            elements['brand_logo'] = self.detect_brand_logo(img, cse_name)
            
            # Layout structure analysis
            elements['layout_structure_match'] = self.analyze_layout_structure(img, cse_name)
            
            return elements
            
        except Exception as e:
            print(f"⚠️  CSE element detection error: {e}")
            return elements
    
    def detect_brand_logo(self, image, cse_name):
        """Simple brand logo detection using template matching"""
        # This is a simplified version - in practice, you'd use trained logo detectors
        # or more sophisticated template matching
        
        # For now, return based on color analysis and structural features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # High edge density often indicates complex elements like logos
        return edge_density > 0.1
    
    def analyze_layout_structure(self, image, cse_name):
        """Analyze layout structure similarity"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines (indicative of form structures)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=50, maxLineGap=10)
            
            if lines is not None:
                horizontal_lines = sum(1 for line in lines if abs(line[0][1] - line[0][3]) < 5)
                vertical_lines = sum(1 for line in lines if abs(line[0][0] - line[0][2]) < 5)
                
                # Banking sites often have structured layouts with forms
                has_structured_layout = (horizontal_lines >= 2 and vertical_lines >= 2)
                return has_structured_layout
            
            return False
            
        except Exception as e:
            print(f"⚠️  Layout analysis error: {e}")
            return False
    
    def calculate_enhanced_similarity(self, screenshot_path, cse_name):
        """
        Calculate enhanced visual similarity using multiple methods
        Returns: overall_score, has_high_similarity, similarity_details
        """
        # Find best matching template
        best_template = None
        best_template_score = -1
        
        for template_name, template_info in self.cse_templates.items():
            if cse_name.lower() in template_info['cse_name'].lower():
                template_score = self.calculate_template_match_score(template_name, cse_name)
                if template_score > best_template_score:
                    best_template_score = template_score
                    best_template = template_info
        
        if not best_template:
            print(f"🔍 No template found for CSE: {cse_name}")
            return 0.0, False, {}
        
        template_path = best_template['path']
        
        # Calculate multiple similarity metrics
        ssim_score = self.calculate_structural_similarity(screenshot_path, template_path)
        
        screenshot_hash = self.calculate_perceptual_hash(screenshot_path)
        template_hash = best_template['hash']
        hash_similarity = self.calculate_hash_similarity(screenshot_hash, template_hash)
        
        # Detect CSE-specific elements
        cse_elements = self.detect_cse_specific_elements(screenshot_path, cse_name)
        
        # Calculate overall score with weights
        weights = {
            'structural_similarity': 0.4,
            'hash_similarity': 0.3,
            'cse_elements': 0.3
        }
        
        cse_elements_score = sum(cse_elements.values()) / len(cse_elements)
        
        overall_score = (
            weights['structural_similarity'] * ssim_score +
            weights['hash_similarity'] * hash_similarity +
            weights['cse_elements'] * cse_elements_score
        )
        
        # Determine if high similarity
        has_high_similarity = (
            overall_score > 0.85 and  # High overall score
            ssim_score > 0.8 and      # Good structural similarity
            hash_similarity > 0.7 and # Good hash similarity
            cse_elements_score > 0.5  # Some CSE elements detected
        )
        
        similarity_details = {
            'overall_score': overall_score,
            'structural_similarity': ssim_score,
            'hash_similarity': hash_similarity,
            'cse_elements': cse_elements,
            'cse_elements_score': cse_elements_score,
            'has_high_similarity': has_high_similarity,
            'template_used': os.path.basename(template_path)
        }
        
        print(f"🔍 Enhanced similarity for {cse_name}:")
        print(f"   - Overall: {overall_score:.3f}")
        print(f"   - Structural: {ssim_score:.3f}")
        print(f"   - Hash: {hash_similarity:.3f}")
        print(f"   - CSE Elements: {cse_elements_score:.3f}")
        print(f"   - High Similarity: {has_high_similarity}")
        
        return overall_score, has_high_similarity, similarity_details
    
    def calculate_template_match_score(self, template_name, cse_name):
        """Calculate how well template matches CSE"""
        template_lower = template_name.lower()
        cse_lower = cse_name.lower()
        
        score = 0
        
        # Exact name match
        cse_keywords = ['sbi', 'irctc', 'icici', 'hdfc', 'pnb', 'bob', 'airtel', 'iocl', 'nic']
        for keyword in cse_keywords:
            if keyword in cse_lower and keyword in template_lower:
                score += 3
        
        # Partial match
        cse_words = cse_lower.split()
        for word in cse_words:
            if len(word) > 3 and word in template_lower:
                score += 1
        
        return score