"""
Core package initialization
"""

from .document_processor import DocumentProcessor
from .information_extractor import InformationExtractor
from .similarity_matcher import SimilarityMatcher

__all__ = [
    "DocumentProcessor",
    "InformationExtractor",
    "SimilarityMatcher"
]
