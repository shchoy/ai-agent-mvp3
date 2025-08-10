"""
Information Extractor for parsing tender requirements from documents
"""

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import date

from models.tender_models import TenderInformation, CloudPlatform, ITCertification, ProjectType
from services.llm_service import LLMService
from utils.date_utils import DateUtils

logger = logging.getLogger(__name__)


class InformationExtractor:
    """Extracts structured information from tender documents"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.date_utils = DateUtils()
    
    def extract_tender_information(self, documents: Dict[str, str]) -> TenderInformation:
        """
        Extract structured tender information from documents
        
        Args:
            documents: Dictionary of filename -> document content
            
        Returns:
            TenderInformation object with extracted data
        """
        try:
            logger.info("Starting information extraction from tender documents")
            
            # Combine all documents for analysis
            combined_text = self._combine_documents(documents)

            # Use LLM for extraction - this is our primary data source
            llm_extraction = self.llm_service.extract_tender_information(documents)
            
            # Parse and structure the extraction, keeping raw LLM output as primary data
            structured_data = self._parse_llm_extraction(llm_extraction, combined_text)
            
            # Process and validate the data before creating TenderInformation object
            processed_data = self._process_and_validate_data(structured_data)
            
            # Create TenderInformation object with comprehensive extraction
            tender_info = TenderInformation(
                **processed_data,
                raw_documents=documents,
                extraction_confidence=self._calculate_confidence(processed_data)
            )
            
            logger.info("Successfully extracted tender information")
            return tender_info
            
        except Exception as e:
            logger.error(f"Error extracting tender information: {str(e)}")
            # Return empty tender info with error, ensuring all list fields are properly initialized
            return TenderInformation(
                raw_documents=documents,
                extraction_confidence=0.0,
                it_certifications=[],
                technology_stack=[],
                cloud_platforms=[],
                project_types=[]
            )
    
    def _combine_documents(self, documents: Dict[str, str]) -> str:
        """Combine all documents into a single text for analysis"""
        combined = ""
        for filename, content in documents.items():
            combined += f"\n\n=== {filename} ===\n{content}"
        return combined
    
    def _parse_llm_extraction(self, llm_result: str, combined_text: str) -> Dict[str, Any]:
        """Parse the LLM extraction result into structured data"""
        # Use the LLM service's comprehensive parsing function
        # This will provide full structured extraction while preserving the raw LLM output
        parsed_data = self.llm_service.parse_extraction_result(llm_result)
        
        # ENSURE llm_extraction_text is always captured from the raw LLM result
        if 'llm_extraction_text' not in parsed_data or not parsed_data['llm_extraction_text']:
            parsed_data['llm_extraction_text'] = llm_result[:1000] if llm_result else ""
            logger.info(f"Manually added llm_extraction_text: {len(parsed_data['llm_extraction_text'])} characters")
        
        # Also ensure raw_extraction is captured
        if 'raw_extraction' not in parsed_data:
            parsed_data['raw_extraction'] = llm_result
        
        # Debug: Log the LLM extraction text length to verify it's being captured
        llm_text = parsed_data.get('llm_extraction_text', '')
        logger.info(f"LLM extraction text captured: {len(llm_text)} characters")
        
        # Also apply rule-based extraction to fill in any gaps
        rule_data = self._apply_rule_based_extraction(combined_text)
        
        # Merge the results, preferring LLM extraction when available
        merged_data = self._merge_extraction_results(parsed_data, rule_data)
        
        # Debug: Verify the merged data still contains the LLM extraction text
        final_llm_text = merged_data.get('llm_extraction_text', '')
        logger.info(f"Final merged LLM extraction text: {len(final_llm_text)} characters")
        
        return merged_data
    
    def _process_and_validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate extracted data, handling type conversions and validation"""
        from datetime import datetime, date
        
        processed_data = data.copy()
        
        # Convert and validate date fields
        for date_field in ['publish_date', 'response_date']:
            if date_field in processed_data and processed_data[date_field]:
                date_value = processed_data[date_field]
                if isinstance(date_value, str):
                    # Try to parse the date string
                    parsed_date = self._parse_date_safely(date_value)
                    processed_data[date_field] = parsed_date
                    if parsed_date:
                        logger.info(f"Successfully parsed {date_field}: {date_value} -> {parsed_date}")
                    else:
                        logger.warning(f"Failed to parse {date_field}: '{date_value}' - setting to None")
                elif not isinstance(date_value, date):
                    # If it's not a string or date object, set to None
                    processed_data[date_field] = None
                    logger.warning(f"Invalid {date_field} type: {type(date_value)} - setting to None")
        
        # Convert and validate max_tender_value
        if 'max_tender_value' in processed_data and processed_data['max_tender_value']:
            value = processed_data['max_tender_value']
            if isinstance(value, str):
                try:
                    # Remove currency symbols, commas, and extra text
                    import re
                    # Extract numbers from the string
                    clean_value = re.sub(r'[^\d.,]', '', value)
                    clean_value = clean_value.replace(',', '')
                    if clean_value:
                        processed_data['max_tender_value'] = float(clean_value)
                        logger.info(f"Converted max_tender_value: {value} -> {processed_data['max_tender_value']}")
                    else:
                        processed_data['max_tender_value'] = None
                        logger.warning(f"Could not extract number from max_tender_value: '{value}'")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Could not convert max_tender_value '{value}': {str(e)}")
                    processed_data['max_tender_value'] = None
            elif not isinstance(value, (int, float)):
                processed_data['max_tender_value'] = None
                logger.warning(f"Invalid max_tender_value type: {type(value)}")
        
        # Ensure list fields are properly formatted
        list_fields = ['it_certifications', 'technology_stack', 'cloud_platforms', 'project_types']
        for field in list_fields:
            if field in processed_data:
                if processed_data[field] is None:
                    processed_data[field] = []
                elif isinstance(processed_data[field], str):
                    # Convert comma-separated string to list
                    if processed_data[field].strip():
                        processed_data[field] = [item.strip() for item in processed_data[field].split(',') if item.strip()]
                    else:
                        processed_data[field] = []
                elif not isinstance(processed_data[field], list):
                    processed_data[field] = [processed_data[field]] if processed_data[field] else []
        
        # Ensure managed_services is a boolean or None
        if 'managed_services' in processed_data and processed_data['managed_services']:
            if isinstance(processed_data['managed_services'], str):
                value_lower = processed_data['managed_services'].lower()
                processed_data['managed_services'] = value_lower in ['true', 'yes', '1', 'required', 'needed']
            elif not isinstance(processed_data['managed_services'], bool):
                processed_data['managed_services'] = None
        
        # Ensure string fields are proper strings or None
        string_fields = ['company_name', 'contact_person', 'contact_email', 'contact_phone', 
                        'epu_level', 'pricing_schedule', 'solution_summary', 'business_purpose', 
                        'scope_of_work', 'sla_requirements', 'project_duration', 
                        'llm_extraction_text', 'raw_extraction']
        
        for field in string_fields:
            if field in processed_data and processed_data[field] is not None:
                if not isinstance(processed_data[field], str):
                    processed_data[field] = str(processed_data[field])
                elif processed_data[field].strip() == "":
                    processed_data[field] = None
        
        logger.info(f"Data processing complete: {len(processed_data)} fields processed")
        return processed_data
    
    def _parse_date_safely(self, date_str: str) -> Optional[date]:
        """Safely parse a date string using multiple formats"""
        if not date_str or not isinstance(date_str, str):
            return None
        
        # Clean the date string
        date_str = date_str.strip()
        
        # Skip obviously invalid date strings
        if len(date_str) < 6 or date_str.lower() in ['not available', 'n/a', 'none', 'null']:
            return None
        
        # Try multiple date formats
        date_formats = [
            '%Y-%m-%d',           # 2024-12-31
            '%d/%m/%Y',           # 31/12/2024
            '%d-%m-%Y',           # 31-12-2024
            '%m/%d/%Y',           # 12/31/2024
            '%B %d, %Y',          # December 31, 2024
            '%d %B %Y',           # 31 December 2024
            '%b %d, %Y',          # Dec 31, 2024
            '%d %b %Y',           # 31 Dec 2024
            '%Y/%m/%d',           # 2024/12/31
            '%d.%m.%Y',           # 31.12.2024
        ]
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                return parsed_date
            except ValueError:
                continue
        
        # Try using the date_utils if available
        try:
            return self.date_utils.parse_date_from_text(date_str)
        except:
            pass
        
        logger.warning(f"Could not parse date string: '{date_str}'")
        return None

    def _extract_critical_fields_only(self, llm_result: str) -> Dict[str, Any]:
        """Extract only critical fields needed for business logic validation"""
        critical_data = {
            'response_date': None,
            'max_tender_value': None,
        }
        
        lines = llm_result.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Only extract response deadline for timeline validation
            if "response deadline" in line.lower() or "deadline:" in line.lower():
                for j in range(i, min(i+3, len(lines))):
                    check_line = lines[j]
                    if "deadline:" in check_line.lower():
                        date_part = check_line.split(":")[-1].strip()
                        if date_part and not date_part.lower().startswith('not'):
                            critical_data["response_date"] = date_part
                            break
            
            # Only extract tender value for financial validation
            elif "maximum tender" in line.lower() or "tender award value" in line.lower():
                for j in range(i, min(i+3, len(lines))):
                    check_line = lines[j]
                    if "$" in check_line or "sgd" in check_line.lower():
                        # Extract monetary value
                        import re
                        money_match = re.search(r'[\$]?[\d,]+(?:\.\d{2})?', check_line)
                        if money_match:
                            value_str = money_match.group().replace('$', '').replace(',', '')
                            try:
                                critical_data["max_tender_value"] = float(value_str)
                            except:
                                pass
                        break
        
        return critical_data
    
    def _calculate_llm_confidence(self, llm_extraction: str) -> float:
        """Calculate confidence score based on LLM extraction completeness"""
        if not llm_extraction or len(llm_extraction.strip()) < 50:
            return 0.0
        
        # Simple heuristic based on extraction length and structure
        lines = [line.strip() for line in llm_extraction.split('\n') if line.strip()]
        
        # Look for key indicators of complete extraction
        key_indicators = [
            'solution summary', 'business purpose', 'scope of work',
            'technology', 'deadline', 'value', 'company'
        ]
        
        found_indicators = 0
        for indicator in key_indicators:
            if any(indicator.lower() in line.lower() for line in lines):
                found_indicators += 1
        
        # Base confidence on number of key fields found and text length
        base_confidence = found_indicators / len(key_indicators)
        length_factor = min(1.0, len(llm_extraction) / 1000)  # Scale based on content length
        
        return min(0.95, base_confidence * 0.7 + length_factor * 0.3)
    
    def _apply_rule_based_extraction(self, text: str) -> Dict[str, Any]:
        """Apply rule-based extraction for specific fields"""
        extracted = {}
        
        # Extract company/agency name
        extracted['company_name'] = self._extract_company_name(text)
        
        # Extract contact information
        extracted.update(self._extract_contact_info(text))
        
        # Extract dates
        extracted.update(self._extract_dates(text))
        
        # Extract financial information
        extracted.update(self._extract_financial_info(text))
        
        # Extract technical requirements
        extracted.update(self._extract_technical_requirements(text))
        
        # Extract project information
        extracted.update(self._extract_project_info(text))
        
        return extracted
    
    def _extract_company_name(self, text: str) -> Optional[str]:
        """Extract company or agency name"""
        patterns = [
            r'(?:issued by|from|agency|company|organization)[:\s]*([^\n]+)',
            r'([A-Z][A-Za-z\s&]+(?:Agency|Corporation|Company|Ltd|Pte|Inc))',
            r'([A-Z][A-Za-z\s&]+(?:Government|Ministry|Department))',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                company = match.group(1).strip()
                if len(company) > 3 and len(company) < 100:
                    return company
        
        return None
    
    def _extract_contact_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact person, email, and phone"""
        contact_info = {
            'contact_person': None,
            'contact_email': None,
            'contact_phone': None
        }
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            contact_info['contact_email'] = email_matches[0]
        
        # Extract phone
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}',
            r'\b\d{8,15}\b'
        ]
        for pattern in phone_patterns:
            phone_matches = re.findall(pattern, text)
            if phone_matches:
                contact_info['contact_phone'] = phone_matches[0]
                break
        
        # Extract contact person name
        name_patterns = [
            r'contact person[:\s]*([^\n]+)',
            r'contact[:\s]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'person in charge[:\s]*([^\n]+)',
        ]
        for pattern in name_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1).strip()
                if len(name) > 3 and len(name) < 50:
                    contact_info['contact_person'] = name
                    break
        
        return contact_info
    
    def _extract_dates(self, text: str) -> Dict[str, Optional[date]]:
        """Extract publish and response dates"""
        date_info = {
            'publish_date': None,
            'response_date': None
        }
        
        # Look for date patterns with context
        publish_patterns = [
            r'publish(?:ed)?\s+(?:date|on)[:\s]*([^\n]+)',
            r'issue(?:d)?\s+(?:date|on)[:\s]*([^\n]+)',
            r'tender\s+date[:\s]*([^\n]+)',
        ]
        
        response_patterns = [
            r'(?:response|submission|deadline|closing)\s+(?:date|by|on)[:\s]*([^\n]+)',
            r'(?:due|submit)\s+(?:date|by|on)[:\s]*([^\n]+)',
            r'deadline[:\s]*([^\n]+)',
        ]
        
        # Extract publish date
        for pattern in publish_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(1).strip()
                parsed_date = self.date_utils.parse_date_from_text(date_text)
                if parsed_date:
                    date_info['publish_date'] = parsed_date
                    break
        
        # Extract response date
        for pattern in response_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(1).strip()
                parsed_date = self.date_utils.parse_date_from_text(date_text)
                if parsed_date:
                    date_info['response_date'] = parsed_date
                    break
        
        return date_info
    
    def _extract_financial_info(self, text: str) -> Dict[str, Any]:
        """Extract financial information including EPU level and tender value"""
        financial_info = {
            'epu_level': None,
            'max_tender_value': None,
            'pricing_schedule': None
        }
        
        # Extract EPU level
        epu_patterns = [
            r'epu\s+level[:\s]*([^\n]+)',
            r'epu[:\s]*([A-Z0-9]+)',
            r'capacity[:\s]*([A-Z0-9]+)',
        ]
        for pattern in epu_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                epu = match.group(1).strip()
                if epu:
                    financial_info['epu_level'] = epu
                    break
        
        # Extract tender value
        value_patterns = [
            r'(?:tender|contract|project)\s+value[:\s]*sgd?\s*([\d,]+)',
            r'maximum\s+value[:\s]*sgd?\s*([\d,]+)',
            r'budget[:\s]*sgd?\s*([\d,]+)',
            r'sgd?\s*([\d,]+)',
        ]
        for pattern in value_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1).replace(',', '')
                try:
                    value = float(value_str)
                    if value > 1000:  # Reasonable minimum for tender value
                        financial_info['max_tender_value'] = value
                        break
                except ValueError:
                    continue
        
        # Extract pricing schedule information
        pricing_patterns = [
            r'pricing\s+schedule[:\s]*([^\n]+)',
            r'payment\s+terms[:\s]*([^\n]+)',
            r'cost\s+structure[:\s]*([^\n]+)',
        ]
        for pattern in pricing_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                pricing = match.group(1).strip()
                if len(pricing) > 10:
                    financial_info['pricing_schedule'] = pricing
                    break
        
        return financial_info
    
    def _extract_technical_requirements(self, text: str) -> Dict[str, Any]:
        """Extract technical requirements and certifications"""
        tech_info = {
            'it_certifications': [],
            'technology_stack': [],
            'cloud_platforms': [],
            'project_types': []
        }
        
        # Extract IT certifications
        cert_patterns = {
            'soc2': r'soc\s*2',
            'iso27001': r'iso\s*27001',
            'vendor_level_1': r'vendor\s+level\s+1',
            'vendor_level_2': r'vendor\s+level\s+2',
            'vendor_level_3': r'vendor\s+level\s+3',
            'cmmi': r'cmmi',
            'itil': r'itil'
        }
        
        for cert_name, pattern in cert_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                try:
                    tech_info['it_certifications'].append(ITCertification(cert_name))
                except ValueError:
                    pass
        
        # Extract technology stack
        tech_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue',
            'docker', 'kubernetes', 'microservices', 'api', 'rest',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql',
            'hadoop', 'spark', 'kafka', 'elasticsearch', 'redis',
            'tensorflow', 'pytorch', 'machine learning', 'ai', 'ml'
        ]
        
        for keyword in tech_keywords:
            if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
                tech_info['technology_stack'].append(keyword)
        
        # Extract cloud platforms
        cloud_patterns = {
            'aws': r'\b(?:aws|amazon\s+web\s+services)\b',
            'azure': r'\b(?:azure|microsoft\s+azure)\b',
            'gcp': r'\b(?:gcp|google\s+cloud)\b'
        }
        
        for platform_name, pattern in cloud_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                try:
                    tech_info['cloud_platforms'].append(CloudPlatform(platform_name))
                except ValueError:
                    pass
        
        # Extract project types
        project_patterns = {
            'cloud_migration': r'cloud\s+migration',
            'data_platform': r'data\s+platform',
            'big_data': r'big\s+data',
            'serverless': r'serverless',
            'ai_agents': r'ai\s+agent',
            'web_application': r'web\s+application',
            'mobile_application': r'mobile\s+app'
        }
        
        for project_type, pattern in project_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                try:
                    tech_info['project_types'].append(ProjectType(project_type))
                except ValueError:
                    pass
        
        return tech_info
    
    def _extract_project_info(self, text: str) -> Dict[str, Any]:
        """Extract project-related information"""
        project_info = {
            'solution_summary': None,
            'business_purpose': None,
            'scope_of_work': None,
            'managed_services': None,
            'sla_requirements': None,
            'project_duration': None
        }
        
        # Extract solution summary
        summary_patterns = [
            r'solution\s+summary[:\s]*([^\n]{50,500})',
            r'project\s+overview[:\s]*([^\n]{50,500})',
            r'solution\s+description[:\s]*([^\n]{50,500})',
        ]
        for pattern in summary_patterns:
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                project_info['solution_summary'] = matches.group(1).strip()
                break
        
        # Extract business purpose
        purpose_patterns = [
            r'business\s+(?:purpose|benefit)[:\s]*([^\n]{50,500})',
            r'objective[:\s]*([^\n]{50,500})',
            r'goal[:\s]*([^\n]{50,500})',
        ]
        for pattern in purpose_patterns:
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                project_info['business_purpose'] = matches.group(1).strip()
                break
        
        # Extract scope of work
        scope_patterns = [
            r'scope\s+of\s+work[:\s]*([^\n]{50,1000})',
            r'deliverables[:\s]*([^\n]{50,1000})',
            r'services\s+required[:\s]*([^\n]{50,1000})',
        ]
        for pattern in scope_patterns:
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                project_info['scope_of_work'] = matches.group(1).strip()
                break
        
        # Check for managed services
        if re.search(r'managed\s+services', text, re.IGNORECASE):
            project_info['managed_services'] = True
        
        # Extract SLA requirements
        sla_patterns = [
            r'sla[:\s]*([^\n]+)',
            r'service\s+level[:\s]*([^\n]+)',
            r'uptime[:\s]*([^\n]+)',
        ]
        for pattern in sla_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                project_info['sla_requirements'] = matches.group(1).strip()
                break
        
        # Extract project duration
        project_info['project_duration'] = self.date_utils.extract_duration_from_text(text)
        
        return project_info
    
    def _merge_extraction_results(self, llm_data: Dict[str, Any], rule_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge LLM and rule-based extraction results"""
        merged = {}
        
        # Define default values for list fields to prevent None values
        list_field_defaults = {
            'it_certifications': [],
            'technology_stack': [],
            'cloud_platforms': [],
            'project_types': []
        }
        
        # Always preserve these fields from LLM data (they don't exist in rule-based extraction)
        llm_only_fields = {'llm_extraction_text', 'raw_extraction'}
        
        # Combine both dictionaries, preferring rule-based results when available
        all_keys = set(llm_data.keys()) | set(rule_data.keys())
        
        for key in all_keys:
            rule_value = rule_data.get(key)
            llm_value = llm_data.get(key)
            
            # Always preserve LLM-only fields from LLM data
            if key in llm_only_fields:
                merged[key] = llm_value
            # Prefer rule-based extraction when available and valid
            elif rule_value is not None and rule_value != "":
                if isinstance(rule_value, list) and len(rule_value) > 0:
                    merged[key] = rule_value
                elif not isinstance(rule_value, list):
                    merged[key] = rule_value
                else:
                    merged[key] = llm_value
            else:
                merged[key] = llm_value
            
            # Ensure list fields are never None
            if key in list_field_defaults and (merged[key] is None or merged[key] == ""):
                merged[key] = list_field_defaults[key]
        
        return merged
    
    def _calculate_confidence(self, extracted_data: Dict[str, Any]) -> float:
        """Calculate confidence score for extraction quality"""
        total_fields = 0
        filled_fields = 0
        
        important_fields = [
            'company_name', 'response_date', 'solution_summary',
            'max_tender_value', 'technology_stack'
        ]
        
        for field in important_fields:
            total_fields += 1
            value = extracted_data.get(field)
            
            if value is not None:
                if isinstance(value, list) and len(value) > 0:
                    filled_fields += 1
                elif isinstance(value, str) and value.strip():
                    filled_fields += 1
                elif not isinstance(value, (list, str)):
                    filled_fields += 1
        
        return filled_fields / total_fields if total_fields > 0 else 0.0
