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
            api_key=self.config.OPENAI_API_KEY,
            max_tokens=self.config.MAX_TOKENS_OUTPUT
        )
        
        # Initialize prompt templates
        self._setup_prompts()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks that fit within token limits"""
        try:
            # Input validation
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
            
            if not text or not text.strip():
                logger.warning("Empty or whitespace-only text provided")
                return []
            
            chunks = []
            chunk_size = self.config.CHUNK_SIZE
            overlap = self.config.CHUNK_OVERLAP
            
            # Validate configuration values
            if chunk_size <= 0:
                raise ValueError(f"Invalid chunk size: {chunk_size}. Must be positive.")
            
            if overlap < 0:
                raise ValueError(f"Invalid overlap: {overlap}. Must be non-negative.")
            
            if overlap >= chunk_size:
                logger.warning(f"Overlap ({overlap}) >= chunk_size ({chunk_size}). Setting overlap to chunk_size - 1.")
                overlap = max(0, chunk_size - 1)

            logger.info(f"Chunking text: {len(text)} chars, chunk size: {chunk_size}, overlap: {overlap}")

            start = 0
            chunk_count = 0
            max_chunks = 100  # Safety limit to prevent infinite loops
            min_chunk_size = 500  # Minimum meaningful chunk size in characters
            seen_chunks = set()  # Track duplicate chunks
            
            while start < len(text) and chunk_count < max_chunks:
                end = start + chunk_size
                if end > len(text):
                    end = len(text)
                
                chunk = text[start:end]
                chunk_stripped = chunk.strip()
                
                # Skip if chunk is too small or empty
                if len(chunk_stripped) < min_chunk_size:
                    logger.debug(f"Skipping small chunk of {len(chunk_stripped)} characters")
                    # Move forward by a larger step to avoid getting stuck on small sections
                    start = min(start + chunk_size, len(text))
                    continue
                
                # Create a hash of the chunk to detect duplicates/near-duplicates
                # Use first and last 100 chars to create a signature
                chunk_signature = (
                    chunk_stripped[:100] + "..." + chunk_stripped[-100:] 
                    if len(chunk_stripped) > 200 
                    else chunk_stripped
                )
                
                # Skip if we've seen a very similar chunk
                if chunk_signature in seen_chunks:
                    logger.debug(f"Skipping duplicate chunk starting with: {chunk_stripped[:50]}...")
                    # Move forward by a larger step to skip repetitive content
                    start = min(start + chunk_size, len(text))
                    continue
                
                # Add this chunk
                chunks.append(chunk)
                seen_chunks.add(chunk_signature)
                chunk_count += 1
                
                # Move start forward, accounting for overlap
                start = end - overlap
                if start >= len(text):
                    break
                
                # Additional safety: if we're not making progress, break
                if start <= end - chunk_size:
                    logger.warning("Chunking not making progress, breaking to prevent infinite loop")
                    break
            
            if chunk_count >= max_chunks:
                logger.warning(f"Reached maximum chunk limit ({max_chunks}). Text may be truncated.")
            
            logger.info(f"Successfully created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            # Return the original text as a single chunk if chunking fails
            if isinstance(text, str) and text.strip():
                logger.info("Returning original text as single chunk due to chunking error")
                return [text]
            else:
                return []
    
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
You are an expert AI agent analyzing technical similarities between tender requirements and historical project responses.

IMPORTANT: Focus ONLY on technical and solution requirements. Do NOT consider:
- Company or agency names
- Contact person details  
- Tender publish dates or response deadlines
- Administrative information

Current Tender Technical Requirements:
{current_tender}

Historical Projects Found:
{historical_projects}

Analyze the similarity focusing ONLY on:
1. IT solution type and technical complexity
2. Technology stack and technical architecture requirements
3. Scope of technical work and deliverables
4. Technical project scale and complexity
5. Cloud platform and infrastructure requirements
6. Integration and system requirements
7. Performance and scalability requirements

Provide a detailed technical similarity analysis explaining:
- Which technical aspects are similar/different
- Why each historical project is or isn't a good technical match
- Technical risks and opportunities based on past experience
- Technical complexity comparison

Focus purely on technical merit and solution alignment, ignoring all administrative details.
""")

        # Recommendation prompt
        self.recommendation_prompt = ChatPromptTemplate.from_template("""
You are an expert AI agent providing strategic bidding recommendations for tender opportunities.

TENDER TECHNICAL INFORMATION:
{tender_info}

TECHNICAL SIMILARITIES WITH HISTORICAL PROJECTS:
{similarity_analysis}

BUSINESS CRITERIA:
- Minimum tender value: SGD {min_value:,}
- Minimum days to response: {min_days}
- Target cloud platforms: AWS, Azure
- Target project types: Cloud migration, Data platform, Big data, Serverless, AI Agents
- Required similarity threshold: {similarity_threshold}%

ANALYSIS REQUIREMENTS:
1. Evaluate if tender meets minimum value threshold
2. Check if response timeline is adequate
3. Assess cloud platform compatibility
4. Determine project type alignment
5. Calculate highest similarity score from historical projects
6. Identify risk factors and success factors
7. List any missing information that needs clarification

RECOMMENDATION FORMAT:
Provide a clear YES/NO recommendation with:
- Confidence score (0-100%)
- Detailed reasoning based on technical merit
- Risk factors
- Success factors  
- Missing information requiring clarification
- Specific recommendations for the sales team

Base your recommendation on meeting ALL criteria:
- ≥70% similarity with historical projects (technical similarity only)
- Meets minimum tender value
- Adequate submission timeline
- Compatible with AWS/Azure deployment
- Aligns with target project types
""")
    
    def extract_tender_information(self, documents: Dict[str, str]) -> str:
        """
        Extract tender information from documents using LLM with chunking
        
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
            
            # Check if we need to chunk the text
            if len(combined_docs) > self.config.CHUNK_SIZE:
                logger.info(f"Document too large ({len(combined_docs)} chars), processing in chunks")
                return self._extract_with_chunking(combined_docs)
            else:
                # Process normally if within limits
                chain = self.extraction_prompt | self.llm | StrOutputParser()
                result = chain.invoke({"documents": combined_docs})
                logger.info("Successfully extracted tender information")
                return result
            
        except Exception as e:
            logger.error(f"Error extracting tender information: {str(e)}")
            raise
    
    def _extract_with_chunking(self, text: str) -> str:
        """Extract information from large documents using chunking"""
        chunks = self._chunk_text(text)
        extracted_parts = []
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                
                # Create modified prompt for chunk processing
                chunk_prompt = ChatPromptTemplate.from_template("""
You are analyzing part {chunk_num} of {total_chunks} of tender documents. Extract any relevant information you can find.

Extract the following information if present in this chunk:

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
11. Specific technical requirements
12. Any business criteria or constraints

If information is not present in this chunk, respond with "Not found in this chunk" for that item.

Document chunk:
{documents}
""")
                
                chain = chunk_prompt | self.llm | StrOutputParser()
                result = chain.invoke({
                    "documents": chunk,
                    "chunk_num": i+1,
                    "total_chunks": len(chunks)
                })
                
                extracted_parts.append(f"--- Chunk {i+1} Results ---\n{result}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                extracted_parts.append(f"--- Chunk {i+1} Error ---\nFailed to process this chunk: {str(e)}")
        
        # Combine all results
        combined_results = "\n\n".join(extracted_parts)
        
        # Create a final summary
        summary_prompt = ChatPromptTemplate.from_template("""
Based on the following extracted information from multiple document chunks, create a comprehensive summary of the tender information:

{chunk_results}

Please consolidate and deduplicate the information, providing a clear, structured summary covering all 12 required information points.
""")
        
        try:
            chain = summary_prompt | self.llm | StrOutputParser()
            final_result = chain.invoke({"chunk_results": combined_results})
            logger.info("Successfully processed all chunks and created summary")
            return final_result
        except Exception as e:
            logger.error(f"Error creating final summary: {str(e)}")
            # Return the chunk results if summary fails
            return combined_results
    
    def analyze_similarities(self, tender_info_text: str, historical_projects: List[Dict[str, Any]]) -> str:
        """
        Analyze technical similarities between current tender and historical projects
        
        Args:
            tender_info_text: Complete structured extraction text from LLM
            historical_projects: List of historical project data
            
        Returns:
            Similarity analysis text
        """
        try:
            # Use the complete LLM extraction but filter out administrative sections
            # Remove company, contact, and date information for similarity analysis
            filtered_tender_info = self._filter_for_technical_similarity(tender_info_text)
            
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
                "current_tender": filtered_tender_info,
                "historical_projects": historical_text
            })
            
            logger.info("Successfully analyzed technical similarities")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing similarities: {str(e)}")
            raise
    
    def _filter_for_technical_similarity(self, extraction_text: str) -> str:
        """
        Filter extraction text to focus only on technical requirements for similarity analysis
        
        Args:
            extraction_text: Complete LLM extraction
            
        Returns:
            Filtered text focusing on technical aspects
        """
        lines = extraction_text.split('\n')
        filtered_lines = []
        skip_section = False
        
        for line in lines:
            line_lower = line.lower()
            
            # Skip administrative sections
            if any(admin_term in line_lower for admin_term in [
                'company or agency releasing',
                'contact person and their details',
                'tender document publish date',
                'response deadline'
            ]):
                skip_section = True
                continue
            
            # Check if we're starting a new technical section
            if any(tech_term in line_lower for tech_term in [
                'it certification requirements',
                'summary of the it solution',
                'purpose and business benefits',
                'professional services scope',
                'technology stack preferences',
                'cloud platform preferences',
                'specific technical requirements',
                'managed services',
                'service level agreement',
                'project or contract duration',
                'pricing or cost schedule'
            ]):
                skip_section = False
            
            # Include line if we're in a technical section
            if not skip_section:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
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
            # Truncate inputs if they're too long to prevent token limit issues
            max_info_length = self.config.CHUNK_SIZE // 2  # Use half chunk size for each input
            
            if len(tender_info) > max_info_length:
                tender_info = tender_info[:max_info_length] + "... [truncated]"
                logger.warning("Tender info truncated due to length")
                
            if len(similarity_analysis) > max_info_length:
                similarity_analysis = similarity_analysis[:max_info_length] + "... [truncated]"
                logger.warning("Similarity analysis truncated due to length")
            
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
        # Store the complete LLM extraction as the primary data source
        
        parsed_data = {
            "company_name": None,
            "contact_person": None, 
            "contact_email": None,
            "contact_phone": None,
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
            "project_types": [],
            "managed_services": None,
            "sla_requirements": None,
            "project_duration": None,
            "pricing_schedule": None,
            "llm_extraction_text": extraction_text,  # Store for UI display
            "raw_extraction": extraction_text  # Store the complete LLM extraction as primary data
        }
        
        # Debug: Log the first few lines of extraction to see the format
        lines = extraction_text.split('\n')[:10]  # First 10 lines for debugging
        logger.info(f"First 10 lines of LLM extraction for parsing:")
        for i, line in enumerate(lines):
            logger.info(f"Line {i}: '{line.strip()}'")
        
        # Parse structured fields from LLM result with more flexible patterns
        all_lines = extraction_text.split('\n')
        
        for i, line in enumerate(all_lines):
            line = line.strip()
            if not line or line.startswith('---'):
                continue
            
            line_lower = line.lower()
            
            # Extract company information - more flexible patterns
            if any(pattern in line_lower for pattern in [
                "company", "agency", "organization", "client", "issuer"
            ]) and ":" in line and not parsed_data["company_name"]:
                value = line.split(":")[-1].strip()
                if value and len(value) > 3 and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["company_name"] = value
                    logger.info(f"Extracted company_name: {value}")
            
            # Extract contact information
            elif any(pattern in line_lower for pattern in ["contact person", "contact name"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and len(value) > 3 and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["contact_person"] = value
                    logger.info(f"Extracted contact_person: {value}")
            
            elif "email" in line_lower and "@" in line:
                # Extract email address from the line
                import re
                email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', line)
                if email_match:
                    parsed_data["contact_email"] = email_match.group()
                    logger.info(f"Extracted contact_email: {email_match.group()}")
            
            elif "phone" in line_lower and ":" in line:
                value = line.split(":")[-1].strip()
                if value and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["contact_phone"] = value
                    logger.info(f"Extracted contact_phone: {value}")
            
            # Extract dates - more flexible patterns
            elif any(pattern in line_lower for pattern in ["deadline", "response date", "closing date"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["response_date"] = value
                    logger.info(f"Extracted response_date: {value}")
            
            elif any(pattern in line_lower for pattern in ["publish date", "issue date", "tender date"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["publish_date"] = value
                    logger.info(f"Extracted publish_date: {value}")
            
            # Extract financial information - look for money values
            elif any(pattern in line_lower for pattern in [
                "tender value", "award value", "maximum value", "contract value", "budget"
            ]):
                # Look for monetary values in current and next few lines
                for j in range(i, min(i+3, len(all_lines))):
                    check_line = all_lines[j]
                    if any(currency in check_line.lower() for currency in ["$", "sgd", "usd"]):
                        import re
                        money_match = re.search(r'[\$]?[\d,]+(?:\.\d{2})?', check_line)
                        if money_match:
                            value_str = money_match.group().replace('$', '').replace(',', '')
                            try:
                                parsed_data["max_tender_value"] = float(value_str)
                                logger.info(f"Extracted max_tender_value: {parsed_data['max_tender_value']}")
                            except:
                                pass
                        break
            
            elif "epu" in line_lower and ":" in line:
                value = line.split(":")[-1].strip()
                if value and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["epu_level"] = value
                    logger.info(f"Extracted epu_level: {value}")
            
            # Extract solution information
            elif any(pattern in line_lower for pattern in ["solution summary", "solution description"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and len(value) > 10 and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["solution_summary"] = value
                    logger.info(f"Extracted solution_summary: {value[:50]}...")
            
            elif any(pattern in line_lower for pattern in ["business purpose", "purpose", "objective"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and len(value) > 10 and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["business_purpose"] = value
                    logger.info(f"Extracted business_purpose: {value[:50]}...")
            
            elif any(pattern in line_lower for pattern in ["scope of work", "scope", "deliverables"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and len(value) > 10 and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["scope_of_work"] = value
                    logger.info(f"Extracted scope_of_work: {value[:50]}...")
            
            # Extract technology information
            elif any(pattern in line_lower for pattern in ["technology stack", "technologies", "tech stack"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    # Split comma-separated technologies
                    tech_list = [tech.strip() for tech in value.split(',') if tech.strip()]
                    if tech_list:
                        parsed_data["technology_stack"] = tech_list
                        logger.info(f"Extracted technology_stack: {tech_list}")
            
            # Extract project information
            elif any(pattern in line_lower for pattern in ["project duration", "duration", "timeline"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["project_duration"] = value
                    logger.info(f"Extracted project_duration: {value}")
            
            elif "managed services" in line_lower:
                line_text = line.lower()
                if any(yes_word in line_text for yes_word in ["yes", "required", "needed"]):
                    parsed_data["managed_services"] = True
                    logger.info("Extracted managed_services: True")
                elif any(no_word in line_text for no_word in ["no", "not required", "not needed"]):
                    parsed_data["managed_services"] = False
                    logger.info("Extracted managed_services: False")
            
            elif any(pattern in line_lower for pattern in ["sla", "service level"]) and ":" in line:
                value = line.split(":")[-1].strip()
                if value and not any(skip in value.lower() for skip in [
                    'not available', 'information not available', 'not found', 'n/a', 'none'
                ]):
                    parsed_data["sla_requirements"] = value
                    logger.info(f"Extracted sla_requirements: {value}")
        
        # Log summary of what was extracted
        extracted_fields = [k for k, v in parsed_data.items() if v is not None and v != [] and k not in ['llm_extraction_text', 'raw_extraction']]
        logger.info(f"Successfully extracted {len(extracted_fields)} fields: {extracted_fields}")
        
        return parsed_data
