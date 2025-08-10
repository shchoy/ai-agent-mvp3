#!/usr/bin/env python3
"""
Quick test script for token management
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.llm_service import LLMService
from src.utils.config import Config

def test_token_management():
    """Test the token management functionality"""
    
    # Create a large test document that would exceed token limits
    large_text = "This is a test document. " * 2000  # Create ~50KB of text
    
    test_docs = {
        "test_large_document.txt": large_text,
        "test_small_document.txt": "This is a small test document with tender information."
    }
    
    try:
        config = Config()
        print(f"Using model: {config.OPENAI_MODEL}")
        print(f"Chunk size: {config.CHUNK_SIZE}")
        print(f"Max tokens output: {config.MAX_TOKENS_OUTPUT}")
        
        llm_service = LLMService()
        
        print(f"\nTesting with {len(str(test_docs))} characters of text...")
        
        # This should trigger chunking if the text is too large
        result = llm_service.extract_tender_information(test_docs)
        
        print("✅ Token management test successful!")
        print(f"Result length: {len(result)} characters")
        print(f"Result preview: {result[:500]}...")
        
    except Exception as e:
        print(f"❌ Token management test failed: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    test_token_management()
