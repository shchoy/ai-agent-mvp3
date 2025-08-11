"""
Similarity Matcher for comparing tenders with historical data
"""

import logging
from typing import List, Dict, Any, Tuple
import json

from models.tender_models import TenderInformation, SimilarityMatch, ProjectType, CloudPlatform
from services.pinecone_service import PineconeService
from services.llm_service import LLMService
from utils.config import Config

logger = logging.getLogger(__name__)


class SimilarityMatcher:
    """Matches current tender with historical tender responses"""
    
    def __init__(self):
        self.pinecone_service = PineconeService()
        self.llm_service = LLMService()
        self.config = Config()
    
    def find_similar_tenders(self, tender_info: TenderInformation) -> Tuple[List[SimilarityMatch], float, str]:
        """
        Find similar historical tenders and analyze similarities
        
        Args:
            tender_info: Current tender information
            
        Returns:
            Tuple of (similar_matches, highest_score, similarity_analysis)
        """
        try:
            logger.info("Searching for similar historical tenders")
            
            # Create search query from tender information
            search_query = self._create_search_query(tender_info)
            
            # Search for similar tenders in vector database
            similar_matches = self.pinecone_service.search_similar_tenders(
                query_text=search_query,
                k=10,
                score_threshold=0.5  # Lower threshold for initial search
            )
            
            # Filter matches based on business criteria
            filtered_matches = self._filter_matches_by_criteria(similar_matches, tender_info)
            
            # Get the highest similarity score
            highest_score = max([match.similarity_score for match in filtered_matches]) if filtered_matches else 0.0
            
            # Generate detailed similarity analysis using LLM
            similarity_analysis = ""
            if filtered_matches:
                similarity_analysis = self._generate_similarity_analysis(tender_info, filtered_matches)
            
            logger.info(f"Found {len(filtered_matches)} similar tenders with highest score: {highest_score:.3f}")
            
            return filtered_matches, highest_score, similarity_analysis
            
        except Exception as e:
            logger.error(f"Error finding similar tenders: {str(e)}")
            return [], 0.0, "Error occurred during similarity search"
    
    def _create_search_query(self, tender_info: TenderInformation) -> str:
        """
        Create search query text from tender information using LLM extraction summary
        
        Args:
            tender_info: Tender information object
            
        Returns:
            Search query string
        """
        # Prioritize the LLM extraction text as it contains the most comprehensive summary
        if hasattr(tender_info, 'llm_extraction_text') and tender_info.llm_extraction_text:
            logger.info(f"Using LLM extraction text for search query ({len(tender_info.llm_extraction_text)} chars)")
            return tender_info.llm_extraction_text
        
        # Secondary fallback to raw_extraction if available
        if hasattr(tender_info, 'raw_extraction') and tender_info.raw_extraction:
            logger.info(f"Using raw extraction for search query ({len(tender_info.raw_extraction)} chars)")
            return tender_info.raw_extraction
        
        # Final fallback to building query from individual fields
        logger.warning("No LLM extraction available, building query from structured fields")
        query_parts = []
        
        # Add solution summary
        if tender_info.solution_summary:
            query_parts.append(f"Solution: {tender_info.solution_summary}")
        
        # Add business purpose
        if tender_info.business_purpose:
            query_parts.append(f"Purpose: {tender_info.business_purpose}")
        
        # Add scope of work
        if tender_info.scope_of_work:
            query_parts.append(f"Scope: {tender_info.scope_of_work}")
        
        # Add technology stack
        if tender_info.technology_stack:
            tech_stack = ", ".join(tender_info.technology_stack)
            query_parts.append(f"Technologies: {tech_stack}")
        
        # Add cloud platforms
        if tender_info.cloud_platforms:
            platforms = ", ".join([platform.value for platform in tender_info.cloud_platforms])
            query_parts.append(f"Cloud platforms: {platforms}")
        
        # Add project types
        if tender_info.project_types:
            project_types = ", ".join([pt.value for pt in tender_info.project_types])
            query_parts.append(f"Project types: {project_types}")
        
        # Add managed services info
        if tender_info.managed_services:
            query_parts.append("Managed services required")
        
        # Add project duration
        if tender_info.project_duration:
            query_parts.append(f"Duration: {tender_info.project_duration}")
        
        return " | ".join(query_parts)
    
    def _filter_matches_by_criteria(
        self, 
        matches: List[SimilarityMatch], 
        tender_info: TenderInformation
    ) -> List[SimilarityMatch]:
        """
        Filter matches based on business criteria
        
        Args:
            matches: List of similarity matches
            tender_info: Current tender information
            
        Returns:
            Filtered list of matches
        """
        filtered = []
        
        for match in matches:
            # Basic similarity threshold
            if match.similarity_score < 0.5:
                continue
            
            # Check technology stack overlap if available
            if (match.technology_stack and tender_info.technology_stack and 
                not self._has_technology_overlap(match.technology_stack, tender_info.technology_stack)):
                continue
            
            # Prefer successful outcomes
            if match.success_outcome is False:
                # Lower the score for unsuccessful projects
                match.similarity_score *= 0.8
            
            filtered.append(match)
        
        # Sort by similarity score (highest first)
        filtered.sort(key=lambda x: x.similarity_score, reverse=True)
        
        # Return top 5 matches
        return filtered[:5]
    
    def _has_technology_overlap(self, tech_stack1: List[str], tech_stack2: List[str]) -> bool:
        """
        Check if two technology stacks have significant overlap
        
        Args:
            tech_stack1: First technology stack
            tech_stack2: Second technology stack
            
        Returns:
            True if there's significant overlap
        """
        if not tech_stack1 or not tech_stack2:
            return True  # Assume compatibility if one is unknown
        
        # Convert to lowercase for comparison
        stack1_lower = [tech.lower() for tech in tech_stack1]
        stack2_lower = [tech.lower() for tech in tech_stack2]
        
        # Calculate overlap percentage
        overlap = len(set(stack1_lower) & set(stack2_lower))
        total_unique = len(set(stack1_lower) | set(stack2_lower))
        
        overlap_percentage = overlap / total_unique if total_unique > 0 else 0
        
        # Consider 20% overlap as significant
        return overlap_percentage >= 0.2
    
    def _generate_similarity_analysis(
        self, 
        tender_info: TenderInformation, 
        similar_matches: List[SimilarityMatch]
    ) -> str:
        """
        Generate detailed similarity analysis using LLM
        
        Args:
            tender_info: Current tender information
            similar_matches: List of similar matches
            
        Returns:
            Detailed similarity analysis
        """
        try:
            # Use the raw LLM extraction text for similarity analysis
            # This preserves all the structured information from the LLM
            raw_extraction = getattr(tender_info, 'raw_extraction', None) or getattr(tender_info, 'llm_extraction_text', None)
            
            if not raw_extraction:
                # Fallback to summary if raw extraction not available
                logger.warning("Raw extraction not available, falling back to summary")
                raw_extraction = self._summarize_tender_for_llm(tender_info)
            
            # Prepare historical projects data
            historical_data = []
            for match in similar_matches:
                project_data = {
                    "title": match.title,
                    "solution_type": match.solution_type,
                    "technology_stack": match.technology_stack,
                    "project_value": match.project_value,
                    "success_outcome": match.success_outcome,
                    "lessons_learned": match.lessons_learned,
                    "similarity_score": match.similarity_score
                }
                historical_data.append(project_data)
            
            # Generate analysis using LLM with raw extraction text
            # This focuses on technical similarity excluding administrative details
            analysis = self.llm_service.analyze_similarities(raw_extraction, historical_data)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating similarity analysis: {str(e)}")
            return "Unable to generate detailed similarity analysis"
    
    def _summarize_tender_for_llm(self, tender_info: TenderInformation) -> str:
        """
        Summarize tender information for LLM analysis
        
        Args:
            tender_info: Tender information object
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        if tender_info.company_name:
            summary_parts.append(f"Client: {tender_info.company_name}")
        
        if tender_info.solution_summary:
            summary_parts.append(f"Solution Required: {tender_info.solution_summary}")
        
        if tender_info.business_purpose:
            summary_parts.append(f"Business Purpose: {tender_info.business_purpose}")
        
        if tender_info.scope_of_work:
            summary_parts.append(f"Scope of Work: {tender_info.scope_of_work}")
        
        if tender_info.technology_stack:
            tech_stack = ", ".join(tender_info.technology_stack)
            summary_parts.append(f"Technology Requirements: {tech_stack}")
        
        if tender_info.cloud_platforms:
            platforms = ", ".join([p.value for p in tender_info.cloud_platforms])
            summary_parts.append(f"Cloud Platforms: {platforms}")
        
        if tender_info.project_types:
            types = ", ".join([pt.value for pt in tender_info.project_types])
            summary_parts.append(f"Project Types: {types}")
        
        if tender_info.max_tender_value:
            summary_parts.append(f"Maximum Value: SGD {tender_info.max_tender_value:,.0f}")
        
        if tender_info.project_duration:
            summary_parts.append(f"Duration: {tender_info.project_duration}")
        
        if tender_info.managed_services:
            summary_parts.append("Managed Services: Required")
        
        if tender_info.sla_requirements:
            summary_parts.append(f"SLA Requirements: {tender_info.sla_requirements}")
        
        return "\n".join(summary_parts)
    
    def calculate_overall_similarity(self, similar_matches: List[SimilarityMatch]) -> float:
        """
        Calculate overall similarity score based on best matches
        
        Args:
            similar_matches: List of similarity matches
            
        Returns:
            Overall similarity score (0-1)
        """
        if not similar_matches:
            return 0.0
        
        # Weight the top matches more heavily
        weights = [0.5, 0.3, 0.15, 0.05]  # First 4 matches
        total_score = 0.0
        total_weight = 0.0
        
        for i, match in enumerate(similar_matches[:4]):
            weight = weights[i] if i < len(weights) else 0.0
            total_score += match.similarity_score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def get_compatibility_score(self, tender_info: TenderInformation) -> Dict[str, bool]:
        """
        Check compatibility with business criteria using LLM extraction summary
        
        Args:
            tender_info: Tender information
            
        Returns:
            Dictionary of compatibility checks
        """
        compatibility = {}
        
        # Use LLM extraction text for enhanced criteria analysis
        llm_extraction = getattr(tender_info, 'llm_extraction_text', '') or ''
        
        # Check cloud platform compatibility using both structured data and LLM extraction
        supported_platforms = {CloudPlatform.AWS, CloudPlatform.AZURE}
        tender_platforms = set(tender_info.cloud_platforms) if tender_info.cloud_platforms else set()
        
        # Enhanced cloud compatibility check using LLM extraction
        cloud_mentioned_in_llm = self._check_cloud_platforms_in_text(llm_extraction)
        
        compatibility['cloud_compatibility'] = (
            not tender_platforms or  # No specific requirement in structured data
            bool(tender_platforms & supported_platforms) or  # Has compatible platform in structured data
            not cloud_mentioned_in_llm or  # No cloud requirement mentioned in LLM extraction
            self._has_supported_cloud_in_text(llm_extraction)  # LLM extraction mentions supported platforms
        )
        
        # Check project type alignment using both structured data and LLM extraction
        target_types = {
            ProjectType.CLOUD_MIGRATION,
            ProjectType.DATA_PLATFORM,
            ProjectType.BIG_DATA,
            ProjectType.SERVERLESS,
            ProjectType.AI_AGENTS
        }
        tender_types = set(tender_info.project_types) if tender_info.project_types else set()
        
        # Enhanced project type matching using LLM extraction
        compatibility['project_type_match'] = (
            not tender_types or  # No specific requirement in structured data
            bool(tender_types & target_types) or  # Has target project type in structured data
            self._has_compatible_project_type_in_text(llm_extraction)  # LLM extraction indicates compatible project
        )
        
        # Check tender value threshold
        compatibility['meets_value_threshold'] = (
            tender_info.max_tender_value is None or
            tender_info.max_tender_value >= self.config.MAX_TENDER_VALUE
        )
        
        # Check timeline adequacy
        from utils.date_utils import DateUtils
        date_utils = DateUtils()
        compatibility['meets_timeline_threshold'] = (
            tender_info.response_date is None or
            date_utils.is_sufficient_time(tender_info.response_date, self.config.MIN_DAYS_TO_SUBMISSION)
        )
        
        logger.info(f"Compatibility analysis using LLM extraction ({len(llm_extraction)} chars): "
                   f"cloud={compatibility['cloud_compatibility']}, "
                   f"project_type={compatibility['project_type_match']}, "
                   f"value={compatibility['meets_value_threshold']}, "
                   f"timeline={compatibility['meets_timeline_threshold']}")
        
        return compatibility
    
    def _check_cloud_platforms_in_text(self, text: str) -> bool:
        """Check if cloud platforms are mentioned in the LLM extraction text"""
        if not text:
            return False
        
        text_lower = text.lower()
        cloud_keywords = ['aws', 'azure', 'gcp', 'google cloud', 'amazon web services', 
                         'microsoft azure', 'cloud platform', 'cloud infrastructure']
        
        return any(keyword in text_lower for keyword in cloud_keywords)
    
    def _has_supported_cloud_in_text(self, text: str) -> bool:
        """Check if supported cloud platforms (AWS/Azure) are mentioned in LLM extraction"""
        if not text:
            return True  # Assume compatible if no specific mention
        
        text_lower = text.lower()
        supported_keywords = ['aws', 'azure', 'amazon web services', 'microsoft azure']
        
        return any(keyword in text_lower for keyword in supported_keywords)
    
    def _has_compatible_project_type_in_text(self, text: str) -> bool:
        """Check if compatible project types are mentioned in LLM extraction"""
        if not text:
            return True  # Assume compatible if no specific mention
        
        text_lower = text.lower()
        compatible_keywords = [
            'cloud migration', 'data platform', 'big data', 'serverless',
            'ai agent', 'artificial intelligence', 'machine learning',
            'data analytics', 'data engineering', 'microservices',
            'api development', 'web application', 'software development'
        ]
        
        return any(keyword in text_lower for keyword in compatible_keywords)
