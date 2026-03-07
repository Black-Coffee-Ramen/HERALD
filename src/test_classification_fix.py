# test_classification_fix.py
from src.core.content_classifier import ContentClassifier

def test_fix():
    """Test the classification fix"""
    print("🧪 TESTING CLASSIFICATION FIX")
    print("=" * 50)
    
    classifier = ContentClassifier()
    
    test_cases = [
        {
            "domain": "ifsc.bankifsccode.com",
            "html": """
            <html>
            <head><title>IFSC Code Finder - All Bank Codes</title></head>
            <body>
                <h1>IFSC Code Lookup</h1>
                <p>Find IFSC codes for all banks in India</p>
                <p>Search by bank name, branch, or location</p>
                <input type="text" placeholder="Enter bank name">
                <button>Search</button>
            </body>
            </html>
            """,
            "target_cse": "Financial Institution (Generic)",
            "expected": "Legitimate Service"
        },
        {
            "domain": "sbi-online-login.xyz", 
            "html": """
            <html>
            <head><title>State Bank of India - Secure Login</title></head>
            <body>
                <h1>SBI Online Banking</h1>
                <form>
                    <input type="text" placeholder="Username">
                    <input type="password" placeholder="Password">
                    <input type="submit" value="Login">
                </form>
                <p>Secure login to your SBI account</p>
            </body>
            </html>
            """,
            "target_cse": "State Bank of India (SBI)",
            "expected": "Phishing"
        },
        {
            "domain": "parked-domain-123.xyz",
            "html": """
            <html>
            <body>
                <h1>This domain is parked</h1>
                <p>This domain may be for sale</p>
                <p>Contact us to buy this domain</p>
            </body>
            </html>
            """,
            "target_cse": "Unknown", 
            "expected": "Suspected"
        }
    ]
    
    for test in test_cases:
        result = classifier.analyze_content(test["html"], test["domain"], test["target_cse"])
        status = "✅ PASS" if result == test["expected"] else "❌ FAIL"
        print(f"{status} {test['domain']}")
        print(f"   Expected: {test['expected']}")
        print(f"   Got: {result}")
        print()

if __name__ == "__main__":
    test_fix()