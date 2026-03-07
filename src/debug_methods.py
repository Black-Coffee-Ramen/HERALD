# src/debug_methods.py
from src.core.content_classifier import ContentClassifier

def debug_methods():
    """Debug to see what methods are actually available"""
    print("🔍 DEBUGGING CONTENT CLASSIFIER METHODS")
    print("=" * 50)
    
    # Create instance
    classifier = ContentClassifier()
    print("✅ ContentClassifier instance created")
    
    # Get ALL methods (including private ones)
    all_methods = [method for method in dir(classifier) if callable(getattr(classifier, method))]
    print(f"📋 ALL METHODS: {all_methods}")
    
    # Check if analyze_content exists
    if 'analyze_content' in all_methods:
        print("✅ analyze_content method exists!")
    else:
        print("❌ analyze_content method NOT found!")
        
    # Try to call whatever method exists
    test_html = "<html><body><p>IFSC code lookup service</p></body></html>"
    
    # Try different possible method names
    possible_methods = ['analyze_content', 'classify_content', 'analyze', 'classify']
    
    for method_name in possible_methods:
        if hasattr(classifier, method_name):
            print(f"🎯 Trying method: {method_name}")
            try:
                method = getattr(classifier, method_name)
                result = method(test_html, "test.com", "Financial Institution (Generic)")
                print(f"✅ SUCCESS with {method_name}: {result}")
                break
            except Exception as e:
                print(f"❌ Failed with {method_name}: {e}")

if __name__ == "__main__":
    debug_methods()