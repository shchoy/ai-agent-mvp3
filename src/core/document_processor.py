"""
Document Processor for handling ZIP files and extracting text
"""

import logging
from typing import Dict, List, Tuple
import time

from utils.file_handler import FileHandler
from models.tender_models import TenderInformation

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Processes tender documents from ZIP files"""
    
    def __init__(self):
        self.file_handler = FileHandler()
    
    def process_tender_zip(self, zip_path: str) -> Tuple[Dict[str, str], List[str], List[str]]:
        """
        Process a ZIP file containing tender documents
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Tuple of (extracted_texts, processed_files, errors)
        """
        start_time = time.time()
        processed_files = []
        errors = []
        
        try:
            logger.info(f"Processing tender ZIP file: {zip_path}")
            
            # Extract text from all documents in ZIP
            extracted_texts = self.file_handler.process_zip_file(zip_path)
            
            # Track processed files
            processed_files = list(extracted_texts.keys())
            
            # Log processing summary
            processing_time = time.time() - start_time
            logger.info(f"Processed {len(processed_files)} documents in {processing_time:.2f} seconds")
            
            return extracted_texts, processed_files, errors
            
        except Exception as e:
            error_msg = f"Error processing ZIP file: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return {}, processed_files, errors
        
        finally:
            # Clean up temporary files
            self.file_handler.cleanup()
    
    def validate_zip_file(self, zip_path: str) -> Tuple[bool, List[str]]:
        """
        Validate ZIP file and contents
        
        Args:
            zip_path: Path to the ZIP file
            
        Returns:
            Tuple of (is_valid, validation_errors)
        """
        validation_errors = []
        
        try:
            # Check if file exists
            import os
            if not os.path.exists(zip_path):
                validation_errors.append("ZIP file does not exist")
                return False, validation_errors
            
            # Check file size
            file_size = os.path.getsize(zip_path)
            if file_size == 0:
                validation_errors.append("ZIP file is empty")
                return False, validation_errors
            
            if file_size > self.file_handler.config.MAX_FILE_SIZE:
                validation_errors.append(f"ZIP file too large (max {self.file_handler.config.MAX_FILE_SIZE} bytes)")
                return False, validation_errors
            
            # Try to extract and validate contents
            try:
                extraction_dir = self.file_handler.extract_zip_file(zip_path)
                document_files = self.file_handler.get_document_files(extraction_dir)
                
                if not document_files:
                    validation_errors.append("No supported document files found in ZIP")
                    return False, validation_errors
                
                logger.info(f"ZIP validation passed - found {len(document_files)} supported documents")
                return True, validation_errors
                
            except Exception as e:
                validation_errors.append(f"Error extracting ZIP contents: {str(e)}")
                return False, validation_errors
            
            finally:
                # Clean up temporary files
                self.file_handler.cleanup()
        
        except Exception as e:
            validation_errors.append(f"Validation error: {str(e)}")
            return False, validation_errors
    
    def get_document_summary(self, extracted_texts: Dict[str, str]) -> Dict[str, Dict[str, any]]:
        """
        Get summary information about extracted documents
        
        Args:
            extracted_texts: Dictionary of filename -> extracted text
            
        Returns:
            Summary information for each document
        """
        summary = {}
        
        for filename, text in extracted_texts.items():
            word_count = len(text.split())
            char_count = len(text)
            
            # Estimate document type based on content
            doc_type = self._estimate_document_type(filename, text)
            
            summary[filename] = {
                "word_count": word_count,
                "char_count": char_count,
                "estimated_type": doc_type,
                "has_content": bool(text.strip())
            }
        
        return summary
    
    def _estimate_document_type(self, filename: str, text: str) -> str:
        """
        Estimate document type based on filename and content
        
        Args:
            filename: Name of the file
            text: Extracted text content
            
        Returns:
            Estimated document type
        """
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        # Check filename patterns
        if any(word in filename_lower for word in ['rfp', 'request', 'proposal']):
            return "RFP Document"
        elif any(word in filename_lower for word in ['spec', 'requirement', 'technical']):
            return "Technical Specification"
        elif any(word in filename_lower for word in ['contract', 'terms', 'condition']):
            return "Contract Terms"
        elif any(word in filename_lower for word in ['price', 'cost', 'budget']):
            return "Pricing Document"
        elif any(word in filename_lower for word in ['scope', 'sow', 'work']):
            return "Scope of Work"
        
        # Check content patterns
        elif any(phrase in text_lower for phrase in ['terms and conditions', 'contract', 'agreement']):
            return "Contract Document"
        elif any(phrase in text_lower for phrase in ['technical requirements', 'specifications', 'architecture']):
            return "Technical Document"
        elif any(phrase in text_lower for phrase in ['pricing', 'cost', 'budget', 'financial']):
            return "Financial Document"
        elif any(phrase in text_lower for phrase in ['scope of work', 'deliverables', 'timeline']):
            return "Project Document"
        
        return "General Document"
    
    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """
        Extract key sections from document text
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of section_name -> section_content
        """
        sections = {}
        lines = text.split('\n')
        current_section = "general"
        current_content = []
        
        section_keywords = {
            'requirements': ['requirement', 'specification', 'technical', 'functional'],
            'scope': ['scope', 'deliverable', 'work', 'service'],
            'timeline': ['timeline', 'schedule', 'deadline', 'milestone'],
            'pricing': ['price', 'cost', 'budget', 'financial', 'payment'],
            'terms': ['terms', 'condition', 'contract', 'legal'],
            'contact': ['contact', 'person', 'email', 'phone', 'address']
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            section_found = False
            for section_name, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords) and len(line.strip()) < 100:
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content).strip()
                    
                    # Start new section
                    current_section = section_name
                    current_content = []
                    section_found = True
                    break
            
            if not section_found:
                current_content.append(line)
        
        # Save final section
        if current_content:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
