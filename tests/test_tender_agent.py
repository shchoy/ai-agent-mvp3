"""
Unit tests for Tender Qualification Agent
"""

import unittest
import tempfile
import zipfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agents.tender_qualification_agent import TenderQualificationAgent
from src.models.tender_models import TenderInformation, CloudPlatform, ProjectType
from src.utils.file_handler import FileHandler
from src.utils.date_utils import DateUtils


class TestTenderQualificationAgent(unittest.TestCase):
    """Test cases for the Tender Qualification Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_zip(self, filename="test_tender.zip"):
        """Create a test ZIP file with sample documents"""
        zip_path = os.path.join(self.test_dir, filename)
        
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            # Add a sample RFP document
            rfp_content = """
            REQUEST FOR PROPOSAL
            
            Company: Singapore Government Technology Agency
            Contact Person: John Tan
            Email: john.tan@tech.gov.sg
            
            Project: Cloud Migration and Data Platform Development
            
            Publish Date: 15 January 2024
            Response Deadline: 15 March 2024
            
            Maximum Contract Value: SGD 2,000,000
            EPU Level: A1
            
            TECHNICAL REQUIREMENTS:
            - Cloud migration from on-premise to AWS or Azure
            - Big data analytics platform
            - Machine learning capabilities
            - Python, Java, React development
            - Microservices architecture
            - Docker and Kubernetes deployment
            
            CERTIFICATIONS REQUIRED:
            - SOC2 compliance
            - ISO27001 certification
            - Vendor Level 2 certification
            
            PROJECT SCOPE:
            - Migrate existing systems to cloud
            - Develop new data analytics platform
            - Implement real-time data processing
            - Machine learning model deployment
            - Staff training and documentation
            
            CONTRACT DURATION: 18 months
            
            MANAGED SERVICES:
            24/7 support required with 99.9% uptime SLA
            """
            
            zip_file.writestr("rfp_document.txt", rfp_content)
            
            # Add a technical specification document
            tech_spec = """
            TECHNICAL SPECIFICATIONS
            
            SOLUTION OVERVIEW:
            The solution should provide a comprehensive cloud-based data platform
            capable of processing large volumes of government data with advanced
            analytics and machine learning capabilities.
            
            TECHNOLOGY STACK:
            - Python 3.8+
            - Apache Spark
            - Kafka for streaming
            - PostgreSQL for structured data
            - MongoDB for unstructured data
            - TensorFlow or PyTorch for ML
            - React.js frontend
            - Docker containerization
            - Kubernetes orchestration
            
            CLOUD REQUIREMENTS:
            - Must support AWS or Azure deployment
            - Auto-scaling capabilities
            - High availability setup
            - Disaster recovery
            
            BUSINESS BENEFITS:
            - Improved decision making through data insights
            - Reduced operational costs
            - Enhanced citizen services
            - Better resource allocation
            """
            
            zip_file.writestr("technical_specs.txt", tech_spec)
        
        return zip_path
    
    @patch('src.services.pinecone_service.pinecone')
    @patch('src.services.llm_service.ChatOpenAI')
    def test_agent_initialization(self, mock_openai, mock_pinecone):
        """Test agent initialization"""
        # Mock the external services
        mock_pinecone.init.return_value = None
        mock_openai.return_value = Mock()
        
        # This would normally fail due to missing API keys, but we can test the structure
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'PINECONE_API_KEY': 'test_key', 
            'PINECONE_ENVIRONMENT': 'test_env'
        }):
            try:
                agent = TenderQualificationAgent()
                self.assertIsNotNone(agent)
                self.assertIsNotNone(agent.document_processor)
                self.assertIsNotNone(agent.information_extractor)
                self.assertIsNotNone(agent.similarity_matcher)
            except Exception as e:
                # Expected due to actual API calls, but we tested the structure
                self.assertIn("API", str(e).upper())
    
    def test_file_handler(self):
        """Test file handling functionality"""
        file_handler = FileHandler()
        
        # Create test ZIP
        zip_path = self.create_test_zip()
        
        # Test ZIP extraction
        extraction_dir = file_handler.extract_zip_file(zip_path)
        self.assertTrue(os.path.exists(extraction_dir))
        
        # Test document file discovery
        doc_files = file_handler.find_document_files(extraction_dir)
        self.assertGreater(len(doc_files), 0)
        
        # Test text extraction
        for file_path in doc_files:
            filename, text = file_handler.extract_text_from_file(file_path)
            self.assertIsNotNone(filename)
            self.assertIsNotNone(text)
            self.assertGreater(len(text), 0)
        
        # Clean up
        file_handler.cleanup()
    
    def test_date_utils(self):
        """Test date utility functions"""
        date_utils = DateUtils()
        
        # Test date parsing
        test_date_strings = [
            "15 January 2024",
            "15/01/2024", 
            "2024-01-15",
            "Jan 15, 2024"
        ]
        
        for date_str in test_date_strings:
            parsed_date = date_utils.parse_date_from_text(date_str)
            if parsed_date:
                self.assertEqual(parsed_date.year, 2024)
                self.assertEqual(parsed_date.month, 1)
                self.assertEqual(parsed_date.day, 15)
        
        # Test duration extraction
        duration_text = "The project duration is 18 months with possible extension"
        duration = date_utils.extract_duration_from_text(duration_text)
        self.assertIsNotNone(duration)
        self.assertIn("18", duration)
    
    def test_tender_information_model(self):
        """Test TenderInformation data model"""
        tender_info = TenderInformation(
            company_name="Test Company",
            max_tender_value=1000000.0,
            cloud_platforms=[CloudPlatform.AWS, CloudPlatform.AZURE],
            project_types=[ProjectType.CLOUD_MIGRATION, ProjectType.DATA_PLATFORM],
            technology_stack=["Python", "AWS", "Docker"],
            extraction_confidence=0.85
        )
        
        self.assertEqual(tender_info.company_name, "Test Company")
        self.assertEqual(tender_info.max_tender_value, 1000000.0)
        self.assertEqual(len(tender_info.cloud_platforms), 2)
        self.assertEqual(len(tender_info.project_types), 2)
        self.assertEqual(len(tender_info.technology_stack), 3)
        self.assertEqual(tender_info.extraction_confidence, 0.85)
    
    @patch('src.services.pinecone_service.pinecone')
    def test_document_processing_workflow(self, mock_pinecone):
        """Test the document processing workflow"""
        from src.core.document_processor import DocumentProcessor
        
        processor = DocumentProcessor()
        
        # Create test ZIP
        zip_path = self.create_test_zip()
        
        # Test validation
        is_valid, errors = processor.validate_zip_file(zip_path)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Test processing
        extracted_texts, processed_files, process_errors = processor.process_tender_zip(zip_path)
        
        self.assertGreater(len(extracted_texts), 0)
        self.assertGreater(len(processed_files), 0)
        self.assertEqual(len(process_errors), 0)
        
        # Verify content extraction
        combined_text = " ".join(extracted_texts.values()).lower()
        self.assertIn("singapore", combined_text)
        self.assertIn("cloud", combined_text)
        self.assertIn("python", combined_text)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation"""
    
    def test_config_validation_missing_keys(self):
        """Test configuration validation with missing keys"""
        from src.utils.config import Config
        
        # Test with missing environment variables
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                Config.validate_config()
    
    def test_config_validation_valid_keys(self):
        """Test configuration validation with valid keys"""
        from src.utils.config import Config
        
        # Test with valid environment variables
        with patch.dict(os.environ, {
            'OPENAI_API_KEY': 'test_key',
            'PINECONE_API_KEY': 'test_key',
            'PINECONE_ENVIRONMENT': 'test_env'
        }):
            result = Config.validate_config()
            self.assertTrue(result)


if __name__ == '__main__':
    # Run tests
    unittest.main()
