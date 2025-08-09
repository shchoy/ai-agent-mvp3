"""
Script to populate Pinecone database with sample historical tender responses
Run this script once to add sample data for testing
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.agents.tender_qualification_agent import TenderQualificationAgent
from src.utils.config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_sample_historical_tenders():
    """Add sample historical tender responses to the database"""
    
    try:
        # Initialize agent
        config = Config()
        config.validate_config()
        agent = TenderQualificationAgent()
        
        # Sample historical tender responses
        sample_tenders = [
            {
                "title": "Government Cloud Migration Project - Ministry of Health",
                "solution_type": "Cloud Migration and Modernization",
                "content": """
                Successfully migrated Ministry of Health's legacy systems to AWS cloud infrastructure.
                The project involved migrating 15 critical applications, implementing new data analytics
                capabilities, and establishing robust security frameworks. The solution included:
                
                - Migration of patient management systems to AWS
                - Implementation of real-time data analytics using AWS services
                - Development of new citizen portal using React and Python
                - Implementation of AI-powered chatbot for citizen services
                - 24/7 managed services with 99.9% uptime SLA
                
                Technologies used: Python, Java, React, AWS (EC2, RDS, Lambda, S3), Docker, Kubernetes,
                Terraform, Jenkins, PostgreSQL, Redis, Elasticsearch.
                
                Project duration: 14 months
                Client satisfaction: Excellent
                All deliverables completed on time and within budget.
                """,
                "technology_stack": ["Python", "Java", "React", "AWS", "Docker", "Kubernetes", "PostgreSQL", "Redis"],
                "project_value": 1800000.0,
                "success_outcome": True,
                "lessons_learned": "Early stakeholder engagement and comprehensive testing were key to success. AWS native services provided better performance than expected."
            },
            
            {
                "title": "Smart City Data Platform - Urban Redevelopment Authority",
                "solution_type": "Big Data Analytics Platform",
                "content": """
                Developed a comprehensive smart city data platform for urban planning and analytics.
                The platform processes real-time data from IoT sensors, traffic systems, and citizen
                applications to provide insights for city planning and operations.
                
                Key features:
                - Real-time data ingestion from 10,000+ IoT sensors
                - Advanced analytics and machine learning models
                - Interactive dashboards for city planners
                - Predictive analytics for traffic and utilities
                - Citizen engagement mobile application
                
                Technologies: Python, Apache Spark, Kafka, MongoDB, TensorFlow, React, Azure,
                Docker, Kubernetes, Prometheus, Grafana.
                
                The platform now processes 50TB of data daily and has improved city planning
                efficiency by 35%.
                """,
                "technology_stack": ["Python", "Apache Spark", "Kafka", "MongoDB", "TensorFlow", "React", "Azure"],
                "project_value": 2500000.0,
                "success_outcome": True,
                "lessons_learned": "Scalable architecture design was crucial. Azure's big data services provided excellent performance for large-scale data processing."
            },
            
            {
                "title": "Digital Transformation - Housing Development Board",
                "solution_type": "Serverless Application Development",
                "content": """
                Built a serverless application ecosystem for housing applications and management.
                The solution replaced legacy mainframe systems with modern, scalable serverless
                architecture on AWS.
                
                Solution components:
                - Serverless APIs using AWS Lambda and API Gateway
                - Event-driven architecture with SQS and SNS
                - Real-time data processing with Kinesis
                - Mobile-first citizen portal
                - AI-powered document processing
                
                Technologies: Python, Node.js, AWS Lambda, DynamoDB, API Gateway, React Native,
                TensorFlow, AWS Textract.
                
                Results:
                - 90% reduction in processing time
                - 60% cost savings compared to previous system
                - Improved citizen satisfaction scores
                """,
                "technology_stack": ["Python", "Node.js", "AWS Lambda", "DynamoDB", "React Native", "TensorFlow"],
                "project_value": 1200000.0,
                "success_outcome": True,
                "lessons_learned": "Serverless architecture provided excellent scalability and cost benefits. Proper event design was critical for system reliability."
            },
            
            {
                "title": "AI-Powered Customer Service Platform - Public Utilities Board",
                "solution_type": "AI Agents and Automation",
                "content": """
                Developed an AI-powered customer service platform with intelligent chatbots
                and automated workflow processing. The system handles 80% of customer inquiries
                automatically and provides seamless escalation to human agents when needed.
                
                Key features:
                - Natural language processing for customer inquiries
                - Multi-channel support (web, mobile, WhatsApp)
                - Intelligent routing and escalation
                - Automated document processing
                - Real-time sentiment analysis
                - Integration with existing billing and CRM systems
                
                Technologies: Python, TensorFlow, BERT, Azure Cognitive Services, React,
                Node.js, MongoDB, Redis, Kubernetes.
                
                Achievements:
                - 80% automation rate for customer inquiries
                - 50% reduction in response time
                - 95% customer satisfaction score
                """,
                "technology_stack": ["Python", "TensorFlow", "Azure", "React", "Node.js", "MongoDB"],
                "project_value": 1600000.0,
                "success_outcome": True,
                "lessons_learned": "Training data quality was crucial for AI model performance. Azure Cognitive Services accelerated development significantly."
            },
            
            {
                "title": "Enterprise Data Warehouse - Inland Revenue Authority",
                "solution_type": "Data Platform and Analytics",
                "content": """
                Implemented a modern data warehouse and analytics platform for tax processing
                and compliance monitoring. The platform consolidates data from multiple sources
                and provides advanced analytics capabilities for fraud detection and compliance.
                
                Technical implementation:
                - Cloud-native data warehouse on Azure Synapse
                - ETL pipelines using Azure Data Factory
                - Machine learning models for fraud detection
                - Self-service analytics with Power BI
                - Real-time monitoring and alerting
                
                Technologies: Azure Synapse, Azure Data Factory, Python, Scala, Power BI,
                Azure ML, Databricks, Kafka.
                
                Note: Project faced delays due to data quality issues and integration challenges
                with legacy systems. Eventually completed 3 months behind schedule.
                """,
                "technology_stack": ["Azure", "Python", "Scala", "Power BI", "Databricks", "Kafka"],
                "project_value": 2200000.0,
                "success_outcome": False,
                "lessons_learned": "Data quality assessment should be done earlier. Legacy system integration was more complex than anticipated. Better change management needed."
            }
        ]
        
        # Add each sample tender to the database
        logger.info("Adding sample historical tender responses to Pinecone database...")
        
        for i, tender in enumerate(sample_tenders, 1):
            logger.info(f"Adding tender {i}/{len(sample_tenders)}: {tender['title']}")
            
            doc_id = agent.add_historical_tender(
                title=tender["title"],
                solution_type=tender["solution_type"],
                content=tender["content"],
                technology_stack=tender["technology_stack"],
                project_value=tender["project_value"],
                success_outcome=tender["success_outcome"],
                lessons_learned=tender["lessons_learned"]
            )
            
            logger.info(f"Added tender with ID: {doc_id}")
        
        logger.info("✅ Successfully added all sample historical tenders!")
        
        # Get database statistics
        stats = agent.get_system_status()
        logger.info(f"Database statistics: {stats.get('database_stats', {})}")
        
    except Exception as e:
        logger.error(f"❌ Error adding sample tenders: {str(e)}")
        print("\nPlease ensure:")
        print("1. Your .env file is properly configured with:")
        print("   - OPENAI_API_KEY")
        print("   - PINECONE_API_KEY") 
        print("   - PINECONE_ENVIRONMENT")
        print("2. Your Pinecone index 'ai-agent-tender-response' exists")
        print("3. All required packages are installed")

if __name__ == "__main__":
    add_sample_historical_tenders()
