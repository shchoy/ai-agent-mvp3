"""
Configuration management utilities
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")  # Use mini for better token limits
    OPENAI_TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE", "0.1"))
    
    # Token Management
    MAX_TOKENS_PER_REQUEST: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "25000"))
    MAX_TOKENS_OUTPUT: int = int(os.getenv("MAX_TOKENS_OUTPUT", "4000"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "15000"))  # Characters per chunk
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "1000"))  # Overlap between chunks
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "ai-agent-tender-response")
    
    # Business Rules
    MAX_TENDER_VALUE: float = float(os.getenv("MAX_TENDER_VALUE", "500000"))
    MIN_DAYS_TO_SUBMISSION: int = int(os.getenv("MIN_DAYS_TO_SUBMISSION", "10"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # LangSmith (Optional)
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY")
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "Tender-Qualification-Agent")
    
    # File Processing
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "50000000"))  # 50MB
    SUPPORTED_EXTENSIONS: list = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls"]
    
    # Directories
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "data/uploads")
    SAMPLE_DIR: str = os.getenv("SAMPLE_DIR", "data/sample_tenders")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        required_vars = [
            cls.OPENAI_API_KEY,
            cls.PINECONE_API_KEY,
            cls.PINECONE_ENVIRONMENT,
        ]
        
        missing_vars = []
        if not cls.OPENAI_API_KEY:
            missing_vars.append("OPENAI_API_KEY")
        if not cls.PINECONE_API_KEY:
            missing_vars.append("PINECONE_API_KEY")
        if not cls.PINECONE_ENVIRONMENT:
            missing_vars.append("PINECONE_ENVIRONMENT")
            
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        return True
    
    @classmethod
    def get_supported_cloud_platforms(cls) -> list:
        """Get list of supported cloud platforms"""
        return ["aws", "azure"]
    
    @classmethod
    def get_target_project_types(cls) -> list:
        """Get list of target project types we want to bid on"""
        return [
            "cloud_migration",
            "data_platform", 
            "big_data",
            "serverless",
            "ai_agents"
        ]
