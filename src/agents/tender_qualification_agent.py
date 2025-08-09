"""
Main Tender Qualification AI Agent
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.document_processor import DocumentProcessor
from core.information_extractor import InformationExtractor
from core.similarity_matcher import SimilarityMatcher
from services.llm_service import LLMService
from services.pinecone_service import PineconeService
from models.tender_models import (
    TenderAnalysisResult, 
    TenderInformation, 
    BiddingRecommendation,
    SimilarityMatch
)
from utils.config import Config
from utils.date_utils import DateUtils

logger = logging.getLogger(__name__)


class TenderQualificationAgent:
    """
    Main AI Agent for qualifying tender documents and providing bidding recommendations
    """
    
    def __init__(self):
        """Initialize the tender qualification agent"""
        self.config = Config()
        
        # Validate configuration
        self.config.validate_config()
        
        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.information_extractor = InformationExtractor()
        self.similarity_matcher = SimilarityMatcher()
        self.llm_service = LLMService()
        self.pinecone_service = PineconeService()
        self.date_utils = DateUtils()
        
        logger.info("Tender Qualification Agent initialized successfully")
    
    def analyze_tender(self, zip_file_path: str) -> TenderAnalysisResult:
        """
        Analyze a tender ZIP file and provide bidding recommendation
        
        Args:
            zip_file_path: Path to the ZIP file containing tender documents
            
        Returns:
            Complete analysis result with recommendation
        """
        start_time = time.time()
        errors = []
        processed_files = []
        
        try:
            logger.info(f"Starting analysis of tender: {zip_file_path}")
            
            # Step 1: Validate and process ZIP file
            logger.info("Step 1: Processing tender documents")
            is_valid, validation_errors = self.document_processor.validate_zip_file(zip_file_path)
            
            if not is_valid:
                errors.extend(validation_errors)
                return self._create_error_result(errors, processed_files, start_time)
            
            # Extract text from all documents
            extracted_texts, processed_files, processing_errors = self.document_processor.process_tender_zip(zip_file_path)
            
            if processing_errors:
                errors.extend(processing_errors)
            
            if not extracted_texts:
                errors.append("No text could be extracted from the provided documents")
                return self._create_error_result(errors, processed_files, start_time)
            
            # Step 2: Extract structured information
            logger.info("Step 2: Extracting tender information")
            tender_info = self.information_extractor.extract_tender_information(extracted_texts)
            
            # Step 3: Find similar historical tenders
            logger.info("Step 3: Finding similar historical tenders")
            similar_matches, highest_similarity, similarity_analysis = self.similarity_matcher.find_similar_tenders(tender_info)
            
            # Step 4: Analyze compatibility with business criteria
            logger.info("Step 4: Analyzing business criteria compatibility")
            compatibility = self.similarity_matcher.get_compatibility_score(tender_info)
            
            # Step 5: Generate bidding recommendation
            logger.info("Step 5: Generating bidding recommendation")
            recommendation = self._generate_bidding_recommendation(
                tender_info,
                similar_matches,
                highest_similarity,
                similarity_analysis,
                compatibility
            )
            
            # Step 6: Create final result
            processing_time = time.time() - start_time
            
            result = TenderAnalysisResult(
                tender_info=tender_info,
                recommendation=recommendation,
                processing_time=processing_time,
                processed_files=processed_files,
                errors=errors
            )
            
            logger.info(f"Tender analysis completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Recommendation: {'BID' if recommendation.should_bid else 'DO NOT BID'}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during tender analysis: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return self._create_error_result(errors, processed_files, start_time)
    
    def _generate_bidding_recommendation(
        self,
        tender_info: TenderInformation,
        similar_matches: List[SimilarityMatch],
        highest_similarity: float,
        similarity_analysis: str,
        compatibility: Dict[str, bool]
    ) -> BiddingRecommendation:
        """
        Generate comprehensive bidding recommendation
        
        Args:
            tender_info: Extracted tender information
            similar_matches: List of similar historical projects
            highest_similarity: Highest similarity score found
            similarity_analysis: Detailed similarity analysis
            compatibility: Compatibility with business criteria
            
        Returns:
            Bidding recommendation with detailed reasoning
        """
        try:
            # Determine if we should bid based on ALL criteria
            should_bid = (
                highest_similarity >= self.config.SIMILARITY_THRESHOLD and
                compatibility.get('meets_value_threshold', False) and
                compatibility.get('meets_timeline_threshold', False) and
                compatibility.get('cloud_compatibility', False) and
                compatibility.get('project_type_match', False)
            )
            
            # Calculate confidence score
            confidence_factors = [
                highest_similarity,
                tender_info.extraction_confidence,
                1.0 if compatibility.get('meets_value_threshold', False) else 0.0,
                1.0 if compatibility.get('meets_timeline_threshold', False) else 0.0,
                1.0 if compatibility.get('cloud_compatibility', False) else 0.0,
                1.0 if compatibility.get('project_type_match', False) else 0.0
            ]
            confidence_score = sum(confidence_factors) / len(confidence_factors)
            
            # Generate detailed reasoning using LLM
            reasoning = self._generate_detailed_reasoning(
                tender_info, similar_matches, highest_similarity, compatibility, should_bid
            )
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(tender_info, compatibility, similar_matches)
            
            # Identify success factors
            success_factors = self._identify_success_factors(tender_info, compatibility, similar_matches)
            
            # Identify missing information
            missing_information = self._identify_missing_information(tender_info)
            
            # Generate sales recommendations
            sales_recommendations = self._generate_sales_recommendations(
                tender_info, should_bid, missing_information, risk_factors
            )
            
            return BiddingRecommendation(
                should_bid=should_bid,
                confidence_score=confidence_score,
                similarity_score=highest_similarity,
                meets_value_threshold=compatibility.get('meets_value_threshold', False),
                meets_timeline_threshold=compatibility.get('meets_timeline_threshold', False),
                cloud_compatibility=compatibility.get('cloud_compatibility', False),
                project_type_match=compatibility.get('project_type_match', False),
                similar_projects=similar_matches,
                reasoning=reasoning,
                risk_factors=risk_factors,
                success_factors=success_factors,
                missing_information=missing_information,
                sales_recommendations=sales_recommendations
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return BiddingRecommendation(
                should_bid=False,
                confidence_score=0.0,
                similarity_score=0.0,
                meets_value_threshold=False,
                meets_timeline_threshold=False,
                cloud_compatibility=False,
                project_type_match=False,
                similar_projects=[],
                reasoning=f"Error generating recommendation: {str(e)}",
                risk_factors=["Error in analysis"],
                success_factors=[],
                missing_information=["Complete analysis could not be performed"],
                sales_recommendations=["Re-submit documents for analysis"]
            )
    
    def _generate_detailed_reasoning(
        self,
        tender_info: TenderInformation,
        similar_matches: List[SimilarityMatch],
        similarity_score: float,
        compatibility: Dict[str, bool],
        should_bid: bool
    ) -> str:
        """Generate detailed reasoning using LLM"""
        try:
            # Prepare summary for LLM
            tender_summary = f"""
Company: {tender_info.company_name or 'Not specified'}
Solution: {tender_info.solution_summary or 'Not specified'}
Value: SGD {tender_info.max_tender_value:,.0f if tender_info.max_tender_value else 'Not specified'}
Response Date: {tender_info.response_date or 'Not specified'}
Technology Stack: {', '.join(tender_info.technology_stack) if tender_info.technology_stack else 'Not specified'}
Cloud Platforms: {', '.join([p.value for p in tender_info.cloud_platforms]) if tender_info.cloud_platforms else 'Not specified'}
"""
            
            similarity_summary = f"Highest similarity score: {similarity_score:.1%}"
            if similar_matches:
                similarity_summary += f"\nTop similar project: {similar_matches[0].title} (Score: {similar_matches[0].similarity_score:.1%})"
            
            reasoning = self.llm_service.generate_recommendation(
                tender_summary,
                similarity_summary,
                similarity_score,
                compatibility
            )
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Error generating detailed reasoning: {str(e)}")
            return f"Basic analysis: {'Recommend bidding' if should_bid else 'Do not recommend bidding'} based on criteria evaluation."
    
    def _identify_risk_factors(
        self,
        tender_info: TenderInformation,
        compatibility: Dict[str, bool],
        similar_matches: List[SimilarityMatch]
    ) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        # Timeline risks
        if tender_info.response_date:
            days_remaining = self.date_utils.calculate_working_days(
                self.date_utils.get_current_date_string(), tender_info.response_date
            )
            if days_remaining < self.config.MIN_DAYS_TO_SUBMISSION:
                risks.append(f"Tight timeline - only {days_remaining} working days to prepare response")
        
        # Value risks
        if tender_info.max_tender_value and tender_info.max_tender_value < self.config.MAX_TENDER_VALUE:
            risks.append(f"Tender value below minimum threshold (SGD {self.config.MAX_TENDER_VALUE:,})")
        
        # Technical risks
        if not compatibility.get('cloud_compatibility', True):
            risks.append("Required cloud platform not in our supported list (AWS, Azure)")
        
        if not compatibility.get('project_type_match', True):
            risks.append("Project type not aligned with our target capabilities")
        
        # Historical performance risks
        unsuccessful_projects = [m for m in similar_matches if m.success_outcome is False]
        if len(unsuccessful_projects) > len(similar_matches) / 2:
            risks.append("High number of similar projects were unsuccessful")
        
        # Information completeness risks
        if tender_info.extraction_confidence < 0.5:
            risks.append("Low confidence in extracted information - many details unclear")
        
        return risks
    
    def _identify_success_factors(
        self,
        tender_info: TenderInformation,
        compatibility: Dict[str, bool],
        similar_matches: List[SimilarityMatch]
    ) -> List[str]:
        """Identify factors supporting success"""
        success_factors = []
        
        # Strong similarity
        if similar_matches and similar_matches[0].similarity_score > 0.8:
            success_factors.append(f"Very high similarity with successful project: {similar_matches[0].title}")
        
        # Good timeline
        if compatibility.get('meets_timeline_threshold', False):
            success_factors.append("Adequate time available for thorough response preparation")
        
        # Good value
        if compatibility.get('meets_value_threshold', False):
            success_factors.append("Tender value meets our minimum threshold")
        
        # Technology alignment
        if compatibility.get('cloud_compatibility', False):
            success_factors.append("Technology requirements align with our cloud capabilities")
        
        # Project type match
        if compatibility.get('project_type_match', False):
            success_factors.append("Project type matches our core expertise areas")
        
        # Historical success
        successful_projects = [m for m in similar_matches if m.success_outcome is True]
        if successful_projects:
            success_factors.append(f"Strong track record in similar projects ({len(successful_projects)} successful)")
        
        return success_factors
    
    def _identify_missing_information(self, tender_info: TenderInformation) -> List[str]:
        """Identify information that needs clarification"""
        missing = []
        
        critical_fields = {
            'company_name': 'Client company/agency name',
            'contact_person': 'Primary contact person details',
            'response_date': 'Tender response deadline',
            'max_tender_value': 'Maximum tender award value',
            'solution_summary': 'Clear description of required solution',
            'scope_of_work': 'Detailed scope of work and deliverables'
        }
        
        for field, description in critical_fields.items():
            value = getattr(tender_info, field, None)
            if not value or (isinstance(value, str) and not value.strip()):
                missing.append(description)
        
        # Check for specific requirements
        if not tender_info.technology_stack:
            missing.append("Technology stack preferences and requirements")
        
        if not tender_info.cloud_platforms:
            missing.append("Cloud platform requirements (AWS, Azure, etc.)")
        
        if tender_info.managed_services is None:
            missing.append("Managed services requirements")
        
        if not tender_info.sla_requirements:
            missing.append("Service Level Agreement (SLA) requirements")
        
        return missing
    
    def _generate_sales_recommendations(
        self,
        tender_info: TenderInformation,
        should_bid: bool,
        missing_information: List[str],
        risk_factors: List[str]
    ) -> List[str]:
        """Generate actionable recommendations for the sales team"""
        recommendations = []
        
        if should_bid:
            recommendations.append("✅ RECOMMEND PROCEEDING with this tender opportunity")
            
            # Specific action items
            if missing_information:
                recommendations.append("📋 Clarify the following information with the client:")
                recommendations.extend([f"   • {info}" for info in missing_information[:5]])
            
            if risk_factors:
                recommendations.append("⚠️ Address these risk factors in your response:")
                recommendations.extend([f"   • {risk}" for risk in risk_factors[:3]])
            
            # Strategic recommendations
            if tender_info.similar_projects:
                recommendations.append("📚 Leverage experience from similar successful projects in your proposal")
            
            recommendations.append("💡 Highlight our cloud expertise and relevant certifications")
            recommendations.append("🎯 Emphasize our track record in similar technology implementations")
            
        else:
            recommendations.append("❌ DO NOT RECOMMEND proceeding with this tender")
            recommendations.append("🔍 Consider these alternatives:")
            
            if not should_bid:
                if missing_information:
                    recommendations.append("   • Request additional information to better assess the opportunity")
                
                recommendations.append("   • Focus on higher-value opportunities that better match our capabilities")
                recommendations.append("   • Use this as a learning opportunity for future similar tenders")
        
        return recommendations
    
    def _create_error_result(
        self, 
        errors: List[str], 
        processed_files: List[str], 
        start_time: float
    ) -> TenderAnalysisResult:
        """Create a result object for error cases"""
        processing_time = time.time() - start_time
        
        return TenderAnalysisResult(
            tender_info=TenderInformation(extraction_confidence=0.0),
            recommendation=BiddingRecommendation(
                should_bid=False,
                confidence_score=0.0,
                similarity_score=0.0,
                meets_value_threshold=False,
                meets_timeline_threshold=False,
                cloud_compatibility=False,
                project_type_match=False,
                similar_projects=[],
                reasoning="Analysis could not be completed due to errors",
                risk_factors=["Analysis incomplete"],
                success_factors=[],
                missing_information=["Complete tender analysis"],
                sales_recommendations=["Resolve document processing issues and resubmit"]
            ),
            processing_time=processing_time,
            processed_files=processed_files,
            errors=errors
        )
    
    def add_historical_tender(
        self,
        title: str,
        solution_type: str,
        content: str,
        technology_stack: List[str],
        project_value: Optional[float] = None,
        success_outcome: Optional[bool] = None,
        lessons_learned: Optional[str] = None
    ) -> str:
        """
        Add a historical tender response to the knowledge base
        
        Args:
            title: Title of the tender response
            solution_type: Type of solution provided
            content: Full content of the tender response
            technology_stack: List of technologies used
            project_value: Value of the project in SGD
            success_outcome: Whether the bid was successful
            lessons_learned: Key lessons from the project
            
        Returns:
            Document ID of the added document
        """
        return self.pinecone_service.add_tender_response(
            title=title,
            solution_type=solution_type,
            content=content,
            technology_stack=technology_stack,
            project_value=project_value,
            success_outcome=success_outcome,
            lessons_learned=lessons_learned
        )
    
    def get_system_status(self) -> Dict[str, any]:
        """Get system status and statistics"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "config_valid": True,
                "pinecone_connected": self.pinecone_service.check_index_exists(),
                "database_stats": self.pinecone_service.get_tender_statistics()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "config_valid": False,
                "error": str(e)
            }
