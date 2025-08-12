"""
Date and time utilities for tender processing
"""

from datetime import datetime, date, timedelta
from typing import Optional, Union
import re
import logging

logger = logging.getLogger(__name__)


class DateUtils:
    """Utility class for date operations"""
    
    @staticmethod
    def parse_date_from_text(text: str) -> Optional[date]:
        """
        Parse date from text using various formats
        
        Args:
            text: Text containing a date
            
        Returns:
            Parsed date or None if not found
        """
        # Common date patterns
        date_patterns = [
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})\b',  # DD Month YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})\b',  # Month DD, YYYY
        ]
        
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()
                    
                    if len(groups) == 3:
                        if pattern == date_patterns[0]:  # DD/MM/YYYY
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        elif pattern == date_patterns[1]:  # YYYY/MM/DD
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        elif pattern == date_patterns[2]:  # DD Month YYYY
                            day = int(groups[0])
                            month = month_map.get(groups[1][:3].lower())
                            year = int(groups[2])
                        elif pattern == date_patterns[3]:  # Month DD, YYYY
                            month = month_map.get(groups[0][:3].lower())
                            day = int(groups[1])
                            year = int(groups[2])
                        
                        if month and 1 <= month <= 12 and 1 <= day <= 31 and year >= 2020:
                            return date(year, month, day)
                            
                except (ValueError, TypeError):
                    continue
        
        return None
    
    @staticmethod
    def parse_date_safely(date_str: str) -> Optional[date]:
        """
        Safely parse a date string using multiple formats with enhanced validation
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Parsed date or None if parsing fails
        """
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
        
        # Try each format
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, fmt).date()
                return parsed_date
            except ValueError:
                continue
        
        # Fallback to the existing parse_date_from_text method
        try:
            return DateUtils.parse_date_from_text(date_str)
        except Exception:
            pass
        
        logger.warning(f"Could not parse date string: '{date_str}'")
        return None
    
    @staticmethod
    def calculate_working_days(start_date: date, end_date: date) -> int:
        """
        Calculate working days between two dates (excluding weekends)
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of working days
        """
        if start_date > end_date:
            return 0
        
        working_days = 0
        current_date = start_date
        
        while current_date <= end_date:
            # 0 = Monday, 6 = Sunday
            if current_date.weekday() < 5:  # Monday to Friday
                working_days += 1
            current_date += timedelta(days=1)
        
        return working_days
    
    @staticmethod
    def is_sufficient_time(response_date: date, min_days: int = 10) -> bool:
        """
        Check if there's sufficient time from today to response date
        
        Args:
            response_date: Tender response deadline
            min_days: Minimum required working days
            
        Returns:
            True if sufficient time available
        """
        today = date.today()
        if response_date <= today:
            return False
        
        working_days = DateUtils.calculate_working_days(today, response_date)
        return working_days >= min_days
    
    @staticmethod
    def extract_duration_from_text(text: str) -> Optional[str]:
        """
        Extract project duration from text
        
        Args:
            text: Text to search for duration
            
        Returns:
            Extracted duration string or None
        """
        duration_patterns = [
            r'(\d+)\s*(?:month|months|mth|mths)',
            r'(\d+)\s*(?:year|years|yr|yrs)',
            r'(\d+)\s*(?:week|weeks|wk|wks)',
            r'(\d+)\s*(?:day|days)',
            r'duration[:\s]*([^.\n]+)',
            r'project\s+period[:\s]*([^.\n]+)',
            r'contract\s+term[:\s]*([^.\n]+)',
        ]
        
        for pattern in duration_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                duration = match.group(1).strip()
                if duration:
                    return duration
        
        return None
    
    @staticmethod
    def format_date_range(start_date: Optional[date], end_date: Optional[date]) -> str:
        """
        Format a date range for display
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Formatted date range string
        """
        if not start_date and not end_date:
            return "No dates specified"
        elif not start_date:
            return f"Until {end_date.strftime('%d %B %Y')}"
        elif not end_date:
            return f"From {start_date.strftime('%d %B %Y')}"
        else:
            return f"{start_date.strftime('%d %B %Y')} - {end_date.strftime('%d %B %Y')}"
    
    @staticmethod
    def get_current_date_string() -> str:
        """Get current date as formatted string"""
        return datetime.now().strftime("%d %B %Y")
    
    @staticmethod
    def parse_flexible_date(date_input: Union[str, date, datetime]) -> Optional[date]:
        """
        Parse date from various input types
        
        Args:
            date_input: Date in various formats
            
        Returns:
            Parsed date or None
        """
        if isinstance(date_input, date):
            return date_input
        elif isinstance(date_input, datetime):
            return date_input.date()
        elif isinstance(date_input, str):
            return DateUtils.parse_date_from_text(date_input)
        else:
            return None
