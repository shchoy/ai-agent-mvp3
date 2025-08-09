"""
Pinecone Vector Database Service for similarity search
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from utils.config import Config
from models.tender_models import SimilarityMatch

logger = logging.getLogger(__name__)


class PineconeService:
    """Service for Pinecone vector database operations"""
    
    def __init__(self):
        self.config = Config()
        
        # Initialize Pinecone with new API
        self.pc = Pinecone(
            api_key=self.config.PINECONE_API_KEY
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.config.OPENAI_API_KEY
        )
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.config.PINECONE_INDEX_NAME,
            embedding=self.embeddings
        )
        
        logger.info(f"Initialized Pinecone service with index: {self.config.PINECONE_INDEX_NAME}")
    
    def search_similar_tenders(
        self, 
        query_text: str, 
        k: int = 10,
        score_threshold: float = 0.7
    ) -> List[SimilarityMatch]:
        """
        Search for similar historical tender responses
        
        Args:
            query_text: Text describing the current tender requirements
            k: Number of similar documents to retrieve
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of similar tender matches
        """
        try:
            # Perform similarity search
            results = self.vector_store.similarity_search_with_score(
                query_text, 
                k=k
            )
            
            similar_matches = []
            
            for document, score in results:
                # Only include results above threshold
                if score >= score_threshold:
                    # Extract metadata
                    metadata = document.metadata
                    
                    similarity_match = SimilarityMatch(
                        document_id=metadata.get("document_id", "unknown"),
                        similarity_score=float(score),
                        title=metadata.get("title", "Untitled"),
                        solution_type=metadata.get("solution_type", "Unknown"),
                        technology_stack=metadata.get("technology_stack", []),
                        project_value=metadata.get("project_value"),
                        success_outcome=metadata.get("success_outcome"),
                        lessons_learned=metadata.get("lessons_learned", "")
                    )
                    
                    similar_matches.append(similarity_match)
            
            logger.info(f"Found {len(similar_matches)} similar tenders above threshold {score_threshold}")
            return similar_matches
            
        except Exception as e:
            logger.error(f"Error searching similar tenders: {str(e)}")
            raise
    
    def add_tender_response(
        self,
        title: str,
        solution_type: str,
        content: str,
        technology_stack: List[str],
        project_value: Optional[float] = None,
        success_outcome: Optional[bool] = None,
        lessons_learned: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a new tender response to the vector database
        
        Args:
            title: Title of the tender response
            solution_type: Type of solution provided
            content: Full content of the tender response
            technology_stack: List of technologies used
            project_value: Value of the project in SGD
            success_outcome: Whether the bid was successful
            lessons_learned: Key lessons from the project
            additional_metadata: Any additional metadata
            
        Returns:
            Document ID of the added document
        """
        try:
            # Generate unique document ID
            doc_id = self._generate_document_id(title, content)
            
            # Prepare metadata
            metadata = {
                "document_id": doc_id,
                "title": title,
                "solution_type": solution_type,
                "technology_stack": technology_stack,
                "project_value": project_value,
                "success_outcome": success_outcome,
                "lessons_learned": lessons_learned or "",
                "added_date": datetime.now().isoformat()
            }
            
            # Add additional metadata if provided
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Create document
            document = Document(
                page_content=content,
                metadata=metadata
            )
            
            # Add to vector store
            self.vector_store.add_documents([document])
            
            logger.info(f"Added tender response to vector store: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding tender response: {str(e)}")
            raise
    
    def update_tender_response(
        self,
        document_id: str,
        success_outcome: bool,
        lessons_learned: str,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing tender response with outcome information
        
        Args:
            document_id: ID of the document to update
            success_outcome: Whether the bid was successful
            lessons_learned: Lessons learned from the project
            additional_metadata: Additional metadata to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Note: Pinecone doesn't support direct updates
            # In a production system, you might want to implement a pattern
            # where you delete and re-add with updated metadata
            
            logger.warning("Update functionality not implemented for Pinecone")
            return False
            
        except Exception as e:
            logger.error(f"Error updating tender response: {str(e)}")
            return False
    
    def get_tender_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the tender database
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get index stats
            index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
            
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", 0),
                "index_fullness": stats.get("index_fullness", 0),
                "namespaces": stats.get("namespaces", {})
            }
            
        except Exception as e:
            logger.error(f"Error getting tender statistics: {str(e)}")
            return {}
    
    def _generate_document_id(self, title: str, content: str) -> str:
        """
        Generate a unique document ID based on title and content hash
        
        Args:
            title: Document title
            content: Document content
            
        Returns:
            Unique document ID
        """
        # Create hash from title and content
        combined_text = f"{title}|{content}"
        hash_object = hashlib.sha256(combined_text.encode())
        hash_hex = hash_object.hexdigest()[:16]  # Use first 16 characters
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d")
        
        return f"tender_{timestamp}_{hash_hex}"
    
    def delete_tender_response(self, document_id: str) -> bool:
        """
        Delete a tender response from the vector database
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index = self.pc.Index(self.config.PINECONE_INDEX_NAME)
            index.delete(ids=[document_id])
            
            logger.info(f"Deleted tender response: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting tender response: {str(e)}")
            return False
    
    def check_index_exists(self) -> bool:
        """
        Check if the Pinecone index exists
        
        Returns:
            True if index exists, False otherwise
        """
        try:
            index_list = self.pc.list_indexes()
            return self.config.PINECONE_INDEX_NAME in [idx.name for idx in index_list.indexes]
            
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False
    
    def create_index_if_not_exists(self, dimension: int = 1536) -> bool:
        """
        Create Pinecone index if it doesn't exist
        
        Args:
            dimension: Vector dimension (default for OpenAI embeddings)
            
        Returns:
            True if created or exists, False if error
        """
        try:
            if not self.check_index_exists():
                self.pc.create_index(
                    name=self.config.PINECONE_INDEX_NAME,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created Pinecone index: {self.config.PINECONE_INDEX_NAME}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False
