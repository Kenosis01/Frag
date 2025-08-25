#!/usr/bin/env python3
"""
HAWK Setup Script
Installs required models and data for advanced NLP features
"""

import subprocess
import sys

def install_spacy_model():
    """Install spaCy English model."""
    try:
        import spacy
        # Try to load the model
        try:
            nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy English model already installed")
        except OSError:
            print("📦 Installing spaCy English model...")
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            print("✅ spaCy English model installed successfully")
    except ImportError:
        print("❌ spaCy not installed. Run: pip install spacy")
        return False
    return True

def setup_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk
        print("📦 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded successfully")
    except ImportError:
        print("❌ NLTK not installed. Run: pip install nltk")
        return False
    return True

if __name__ == "__main__":
    print("🦅 HAWK Setup - Installing NLP Models")
    print("=" * 40)
    
    success = True
    success &= install_spacy_model()
    success &= setup_nltk_data()
    
    if success:
        print("\n🎉 Setup complete! HAWK is ready for advanced feature extraction.")
    else:
        print("\n⚠️ Setup incomplete. Please install missing dependencies.")