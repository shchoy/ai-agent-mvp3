"""
Tender data models using Pydantic
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum


class CloudPlatform(str, Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"


class ITCertification(str, Enum):
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    VENDOR_LEVEL_1 = "vendor_level_1"
    VENDOR_LEVEL_2 = "vendor_level_2"
    VENDOR_LEVEL_3 = "vendor_level_3"
    CMMI = "cmmi"
    ITIL = "itil"


class ProjectType(str, Enum):
    CLOUD_MIGRATION = "cloud_migration"
    DATA_PLATFORM = "data_platform"
    BIG_DATA = "big_data"
    SERVERLESS = "serverless"
    AI_AGENTS = "ai_agents"
    WEB_APPLICATION = "web_application"
    MOBILE_APPLICATION = "mobile_application"
    INFRASTRUCTURE = "infrastructure"
    CYBERSECURITY = "cybersecurity"


class TenderInformation(BaseModel):
    """Main tender information extracted from documents"""
    
    # Basic Information
    company_name: Optional[str] = Field(None, description="Company or agency releasing the tender")
    contact_person: Optional[str] = Field(None, description="Contact person for the tender")
    contact_email: Optional[str] = Field(None, description="Contact email")
    contact_phone: Optional[str] = Field(None, description="Contact phone number")
    
    # Important Dates
    publish_date: Optional[date] = Field(None, description="Tender document publish date")
    response_date: Optional[date] = Field(None, description="Tender response deadline")
    
    # Financial Information
    epu_level: Optional[str] = Field(None, description="EPU level - maximum tender award value")
    max_tender_value: Optional[float] = Field(None, description="Maximum tender award value in SGD")
    pricing_schedule: Optional[str] = Field(None, description="Pricing or cost schedule requirements")
    
    # Technical Requirements
    it_certifications: List[ITCertification] = Field(default_factory=list, description="Required IT certifications")
    solution_summary: Optional[str] = Field(None, description="Summary of IT solution requested")
    business_purpose: Optional[str] = Field(None, description="Purpose and business benefit")
    scope_of_work: Optional[str] = Field(None, description="Professional services scope of work")
    technology_stack: List[str] = Field(default_factory=list, description="Technology stack preferences")
    cloud_platforms: List[CloudPlatform] = Field(default_factory=list, description="Preferred cloud platforms")
    project_types: List[ProjectType] = Field(default_factory=list, description="Type of project")
    
    # Service Requirements
    managed_services: Optional[bool] = Field(None, description="Whether managed services are required")
    sla_requirements: Optional[str] = Field(None, description="Service Level Agreement requirements")
    project_duration: Optional[str] = Field(None, description="Expected project or contract duration")
    
    # Raw extracted text for further analysis
    raw_documents: Dict[str, str] = Field(default_factory=dict, description="Raw text from documents")
    
    # Confidence scores
    extraction_confidence: float = Field(0.0, description="Confidence score for information extraction")


class SimilarityMatch(BaseModel):
    """Represents a similar historical tender response"""
    
    document_id: str = Field(..., description="Unique identifier for the historical document")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    title: str = Field(..., description="Title of the historical tender")
    solution_type: str = Field(..., description="Type of solution provided")
    technology_stack: List[str] = Field(default_factory=list, description="Technology stack used")
    project_value: Optional[float] = Field(None, description="Value of the historical project")
    success_outcome: Optional[bool] = Field(None, description="Whether the bid was successful")
    lessons_learned: Optional[str] = Field(None, description="Key lessons from the project")


class BiddingRecommendation(BaseModel):
    """AI Agent's recommendation for whether to bid on the tender"""
    
    should_bid: bool = Field(..., description="Whether to recommend bidding")
    confidence_score: float = Field(..., description="Confidence in the recommendation (0-1)")
    
    # Criteria Analysis
    similarity_score: float = Field(..., description="Highest similarity score found")
    meets_value_threshold: bool = Field(..., description="Meets minimum tender value requirement")
    meets_timeline_threshold: bool = Field(..., description="Meets minimum timeline requirement")
    cloud_compatibility: bool = Field(..., description="Solution can be deployed on AWS/Azure")
    project_type_match: bool = Field(..., description="Project type matches our capabilities")
    
    # Similar Projects
    similar_projects: List[SimilarityMatch] = Field(default_factory=list, description="Similar historical projects")
    
    # Detailed Reasoning
    reasoning: str = Field(..., description="Detailed reasoning for the recommendation")
    risk_factors: List[str] = Field(default_factory=list, description="Identified risk factors")
    success_factors: List[str] = Field(default_factory=list, description="Factors supporting success")
    
    # Missing Information
    missing_information: List[str] = Field(default_factory=list, description="Information that needs clarification")
    
    # Recommendations for Sales Team
    sales_recommendations: List[str] = Field(default_factory=list, description="Actionable recommendations for sales team")


class TenderAnalysisResult(BaseModel):
    """Complete analysis result from the AI Agent"""
    
    tender_info: TenderInformation = Field(..., description="Extracted tender information")
    recommendation: BiddingRecommendation = Field(..., description="Bidding recommendation")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    processed_files: List[str] = Field(default_factory=list, description="List of processed files")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered during processing")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat()
        }
