"""
LLM Service for OpenAI GPT-4 interactions
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

from utils.config import Config
from models.tender_models import TenderInformation

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations using OpenAI GPT-4"""
    
    def __init__(self):
        self.config = Config()
        self.llm = ChatOpenAI(
            model=self.config.OPENAI_MODEL,
            temperature=self.config.OPENAI_TEMPERATURE,
            api_key=self.config.OPENAI_API_KEY
        )
        
        # Initialize prompt templates
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for different tasks"""
        
        # Information extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_template("""
You are an expert AI agent specializing in analyzing tender documents. Your task is to extract key information from the provided tender documents.

Extract the following information from the tender documents:

1. Company or agency releasing the tender
2. Contact person and their details (email, phone)
3. Tender document publish date and response deadline
4. EPU level and maximum tender award value
5. IT certification requirements (SOC2, ISO27001, vendor levels, etc.)
6. Summary of the IT solution requested
7. Purpose and business benefits of the IT solution
8. Professional services scope of work
9. Technology stack preferences
10. Cloud platform preferences (AWS, Azure, GCP, etc.)
11. Managed services support requirements and SLA
12. Expected project or contract duration
13. Pricing or cost schedule requirements

IMPORTANT INSTRUCTIONS:
- If information is not found in the documents, clearly state "Information not available" for that field
- Extract dates in YYYY-MM-DD format when possible
- For monetary values, extract the number and specify the currency
- Be precise and factual - do not make assumptions
- Include confidence level for each extracted piece of information

TENDER DOCUMENTS:
{documents}

Please provide a structured analysis with clear sections for each type of information.
""")

        # Similarity analysis prompt
        self.similarity_prompt = ChatPromptTemplate.from_template("""
You are an expert AI agent analyzing similarities between tender requirements and historical project responses.

Current Tender Requirements:
{current_tender}

Historical Projects Found:
{historical_projects}

Analyze the similarity focusing on:
1. IT solution type and complexity
2. Technology stack alignment
3. Scope of work similarities
4. Project scale and duration
5. Client industry and requirements

Provide a detailed analysis of similarities and differences, highlighting why each historical project is or isn't a good match for the current tender.
""")

        # Recommendation prompt
        self.recommendation_prompt = ChatPromptTemplate.from_template("""
You are an expert AI agent providing strategic bidding recommendations for tender opportunities.

TENDER INFORMATION:
{tender_info}

HISTORICAL SIMILARITIES:
{similarity_analysis}

BUSINESS CRITERIA:
- Minimum tender value: SGD {min_value:,}
- Minimum days to submission: {min_days}
- Target cloud platforms: AWS, Azure
- Target project types: Cloud migration, Data platform, Big data, Serverless, AI Agents
- Required similarity threshold: {similarity_threshold}%

ANALYSIS REQUIREMENTS:
1. Evaluate if tender meets minimum value threshold
2. Check if submission timeline is adequate
3. Assess cloud platform compatibility
4. Determine project type alignment
5. Calculate highest similarity score from historical projects
6. Identify risk factors and success factors
7. List any missing information that needs clarification

RECOMMENDATION FORMAT:
Provide a clear YES/NO recommendation with:
- Confidence score (0-100%)
- Detailed reasoning
- Risk factors
- Success factors  
- Missing information requiring clarification
- Specific recommendations for the sales team

Base your recommendation on meeting ALL criteria:
- ≥70% similarity with historical projects
- Meets minimum tender value
- Adequate submission timeline
- Compatible with AWS/Azure deployment
- Aligns with target project types
""")
    
    def extract_tender_information(self, documents: Dict[str, str]) -> str:
        """
        Extract tender information from documents using LLM
        
        Args:
            documents: Dictionary of filename -> document content
            
        Returns:
            Extracted information as structured text
        """
        try:
            # Combine all documents
            combined_docs = ""
            for filename, content in documents.items():
                combined_docs += f"\n\n=== {filename} ===\n{content}"
            
            # Create chain
            chain = self.extraction_prompt | self.llm | StrOutputParser()
            
            # Execute extraction
            result = chain.invoke({"documents": combined_docs})
            
            logger.info("Successfully extracted tender information")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting tender information: {str(e)}")
            raise
    
    def analyze_similarities(self, tender_info: str, historical_projects: List[Dict[str, Any]]) -> str:
        """
        Analyze similarities between current tender and historical projects
        
        Args:
            tender_info: Current tender information
            historical_projects: List of historical project data
            
        Returns:
            Similarity analysis text
        """
        try:
            # Format historical projects
            historical_text = ""
            for i, project in enumerate(historical_projects, 1):
                historical_text += f"\nProject {i}:\n"
                historical_text += f"Title: {project.get('title', 'N/A')}\n"
                historical_text += f"Solution Type: {project.get('solution_type', 'N/A')}\n"
                historical_text += f"Technology Stack: {', '.join(project.get('technology_stack', []))}\n"
                historical_text += f"Project Value: {project.get('project_value', 'N/A')}\n"
                historical_text += f"Success: {project.get('success_outcome', 'N/A')}\n"
                historical_text += f"Lessons: {project.get('lessons_learned', 'N/A')}\n"
            
            # Create chain
            chain = self.similarity_prompt | self.llm | StrOutputParser()
            
            # Execute analysis
            result = chain.invoke({
                "current_tender": tender_info,
                "historical_projects": historical_text
            })
            
            logger.info("Successfully analyzed similarities")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing similarities: {str(e)}")
            raise
    
    def generate_recommendation(
        self, 
        tender_info: str, 
        similarity_analysis: str,
        similarity_score: float,
        meets_criteria: Dict[str, bool]
    ) -> str:
        """
        Generate bidding recommendation based on analysis
        
        Args:
            tender_info: Extracted tender information
            similarity_analysis: Similarity analysis results
            similarity_score: Highest similarity score found
            meets_criteria: Dict of criteria met/not met
            
        Returns:
            Recommendation text
        """
        try:
            # Create chain
            chain = self.recommendation_prompt | self.llm | StrOutputParser()
            
            # Execute recommendation
            result = chain.invoke({
                "tender_info": tender_info,
                "similarity_analysis": similarity_analysis,
                "min_value": self.config.MAX_TENDER_VALUE,
                "min_days": self.config.MIN_DAYS_TO_SUBMISSION,
                "similarity_threshold": int(self.config.SIMILARITY_THRESHOLD * 100)
            })
            
            logger.info("Successfully generated recommendation")
            return result
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            raise
    
    def parse_extraction_result(self, extraction_text: str) -> Dict[str, Any]:
        """
        Parse the extraction result into structured data
        
        Args:
            extraction_text: Raw extraction result from LLM
            
        Returns:
            Structured dictionary of extracted information
        """
        # This is a simplified parser - in production, you might want to use
        # a more sophisticated approach or train the LLM to output JSON
        
        parsed_data = {
            "company_name": None,
            "contact_person": None, 
            "contact_email": None,
            "publish_date": None,
            "response_date": None,
            "epu_level": None,
            "max_tender_value": None,
            "it_certifications": [],
            "solution_summary": None,
            "business_purpose": None,
            "scope_of_work": None,
            "technology_stack": [],
            "cloud_platforms": [],
            "managed_services": None,
            "sla_requirements": None,
            "project_duration": None,
            "pricing_schedule": None
        }
        
        # Basic text parsing - this would be enhanced in production
        lines = extraction_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for key information patterns
            if "company" in line.lower() and ":" in line:
                parsed_data["company_name"] = line.split(":", 1)[1].strip()
            elif "contact person" in line.lower() and ":" in line:
                parsed_data["contact_person"] = line.split(":", 1)[1].strip()
            elif "email" in line.lower() and ":" in line:
                parsed_data["contact_email"] = line.split(":", 1)[1].strip()
            # Add more parsing logic as needed
        
        return parsed_data
