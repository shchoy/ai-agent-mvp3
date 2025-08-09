"""
Streamlit Web Application for Tender Qualification AI Agent
"""

import streamlit as st
import os
import logging
from typing import Optional
import tempfile
import json
from datetime import datetime


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from agents.tender_qualification_agent import TenderQualificationAgent
    from models.tender_models import TenderAnalysisResult
    from utils.config import Config
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you're running the app from the MVP3 directory")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Tender Qualification AI Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: bold;
    }
    
    .section-header {
        color: #1f4e79;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .info-box {
        background-color: #cce7ff;
        border: 1px solid #99d6ff;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f4e79;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_agent():
    """Initialize the tender qualification agent"""
    try:
        config = Config()
        config.validate_config()
        agent = TenderQualificationAgent()
        return agent, None
    except Exception as e:
        return None, str(e)

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">🤖 Tender Qualification AI Agent</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Upload tender documents in ZIP format to get intelligent bidding recommendations
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with system information and controls"""
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        # Initialize agent
        agent, error = initialize_agent()
        
        if error:
            st.error(f"❌ Configuration Error: {error}")
            st.markdown("""
            **Required Environment Variables:**
            - `OPENAI_API_KEY`
            - `PINECONE_API_KEY`
            - `PINECONE_ENVIRONMENT`
            """)
            return None
        
        # Get system status
        try:
            status = agent.get_system_status()
            
            # Display status indicators
            if status.get('config_valid', False):
                st.success("✅ Configuration Valid")
            else:
                st.error("❌ Configuration Issues")
            
            if status.get('pinecone_connected', False):
                st.success("✅ Database Connected")
                
                # Show database stats
                db_stats = status.get('database_stats', {})
                if db_stats:
                    st.markdown("### Database Statistics")
                    st.metric("Total Historical Tenders", db_stats.get('total_vectors', 0))
                    st.metric("Vector Dimension", db_stats.get('dimension', 0))
            else:
                st.error("❌ Database Connection Failed")
        
        except Exception as e:
            st.error(f"❌ Status Check Failed: {str(e)}")
        
        st.markdown("---")
        
        # Business criteria info
        st.markdown("## ⚙️ Business Criteria")
        config = Config()
        st.info(f"""
        **Minimum Tender Value:** SGD {config.MAX_TENDER_VALUE:,}
        
        **Minimum Timeline:** {config.MIN_DAYS_TO_SUBMISSION} working days
        
        **Similarity Threshold:** {config.SIMILARITY_THRESHOLD:.0%}
        
        **Target Platforms:** AWS, Azure
        
        **Target Projects:** Cloud Migration, Data Platform, Big Data, Serverless, AI Agents
        """)
        
        return agent

def upload_and_validate_file():
    """Handle file upload and validation"""
    st.markdown('<h2 class="section-header">📁 Upload Tender Documents</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a ZIP file containing tender documents",
        type=['zip'],
        help="Upload a ZIP file containing PDF, DOCX, TXT, or Excel files with tender information"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        return tmp_file_path, uploaded_file.name
    
    return None, None

def display_analysis_results(result: TenderAnalysisResult, filename: str):
    """Display the analysis results"""
    
    # Main recommendation
    st.markdown('<h2 class="section-header">🎯 Bidding Recommendation</h2>', unsafe_allow_html=True)
    
    recommendation = result.recommendation
    
    if recommendation.should_bid:
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ RECOMMEND BIDDING</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ DO NOT RECOMMEND BIDDING</h3>
            <p><strong>Confidence Score:</strong> {recommendation.confidence_score:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Criteria analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Criteria Analysis")
        
        criteria = [
            ("Similarity Threshold", recommendation.similarity_score >= 0.7, f"{recommendation.similarity_score:.1%}"),
            ("Tender Value", recommendation.meets_value_threshold, "✅" if recommendation.meets_value_threshold else "❌"),
            ("Timeline", recommendation.meets_timeline_threshold, "✅" if recommendation.meets_timeline_threshold else "❌"),
        ]
        
        for name, passed, value in criteria:
            status = "✅" if passed else "❌"
            st.markdown(f"**{name}:** {status} {value}")
    
    with col2:
        st.markdown("### 🔧 Technical Compatibility")
        
        tech_criteria = [
            ("Cloud Platform", recommendation.cloud_compatibility),
            ("Project Type", recommendation.project_type_match),
        ]
        
        for name, passed in tech_criteria:
            status = "✅" if passed else "❌"
            st.markdown(f"**{name}:** {status}")
    
    # Detailed reasoning
    st.markdown('<h3 class="section-header">🧠 Analysis Reasoning</h3>', unsafe_allow_html=True)
    st.markdown(recommendation.reasoning)
    
    # Extracted tender information
    with st.expander("📋 Extracted Tender Information", expanded=False):
        tender_info = result.tender_info
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            st.write(f"**Company:** {tender_info.company_name or 'Not specified'}")
            st.write(f"**Contact Person:** {tender_info.contact_person or 'Not specified'}")
            st.write(f"**Contact Email:** {tender_info.contact_email or 'Not specified'}")
            st.write(f"**Publish Date:** {tender_info.publish_date or 'Not specified'}")
            st.write(f"**Response Date:** {tender_info.response_date or 'Not specified'}")
            
            st.markdown("#### Financial Information")
            st.write(f"**EPU Level:** {tender_info.epu_level or 'Not specified'}")
            value_str = f"SGD {tender_info.max_tender_value:,.0f}" if tender_info.max_tender_value else "Not specified"
            st.write(f"**Max Tender Value:** {value_str}")
        
        with col2:
            st.markdown("#### Technical Requirements")
            
            if tender_info.technology_stack:
                st.write("**Technology Stack:**")
                for tech in tender_info.technology_stack:
                    st.write(f"  • {tech}")
            else:
                st.write("**Technology Stack:** Not specified")
            
            if tender_info.cloud_platforms:
                platforms = [p.value for p in tender_info.cloud_platforms]
                st.write(f"**Cloud Platforms:** {', '.join(platforms)}")
            else:
                st.write("**Cloud Platforms:** Not specified")
            
            if tender_info.it_certifications:
                certs = [c.value for c in tender_info.it_certifications]
                st.write(f"**IT Certifications:** {', '.join(certs)}")
            else:
                st.write("**IT Certifications:** Not specified")
        
        if tender_info.solution_summary:
            st.markdown("#### Solution Summary")
            st.write(tender_info.solution_summary)
        
        if tender_info.scope_of_work:
            st.markdown("#### Scope of Work")
            st.write(tender_info.scope_of_work)
    
    # Similar projects
    if recommendation.similar_projects:
        with st.expander(f"🔍 Similar Historical Projects ({len(recommendation.similar_projects)} found)", expanded=False):
            for i, project in enumerate(recommendation.similar_projects[:3], 1):
                st.markdown(f"#### Project {i}: {project.title}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similarity Score", f"{project.similarity_score:.1%}")
                with col2:
                    st.metric("Solution Type", project.solution_type)
                with col3:
                    success_text = "✅ Success" if project.success_outcome else "❌ Unsuccessful" if project.success_outcome is False else "Unknown"
                    st.metric("Outcome", success_text)
                
                if project.technology_stack:
                    st.write(f"**Tech Stack:** {', '.join(project.technology_stack)}")
                
                if project.lessons_learned:
                    st.write(f"**Lessons:** {project.lessons_learned}")
                
                st.markdown("---")
    
    # Risk and success factors
    col1, col2 = st.columns(2)
    
    with col1:
        if recommendation.risk_factors:
            st.markdown("### ⚠️ Risk Factors")
            for risk in recommendation.risk_factors:
                st.markdown(f"• {risk}")
    
    with col2:
        if recommendation.success_factors:
            st.markdown("### ✅ Success Factors")
            for factor in recommendation.success_factors:
                st.markdown(f"• {factor}")
    
    # Missing information
    if recommendation.missing_information:
        st.markdown('<h3 class="section-header">❓ Missing Information</h3>', unsafe_allow_html=True)
        st.warning("The following information needs clarification from the sales person:")
        for info in recommendation.missing_information:
            st.markdown(f"• {info}")
    
    # Sales recommendations
    if recommendation.sales_recommendations:
        st.markdown('<h3 class="section-header">💡 Recommendations for Sales Team</h3>', unsafe_allow_html=True)
        for rec in recommendation.sales_recommendations:
            st.markdown(f"{rec}")
    
    # Processing details
    with st.expander("🔧 Processing Details", expanded=False):
        st.write(f"**Processing Time:** {result.processing_time:.2f} seconds")
        st.write(f"**Files Processed:** {len(result.processed_files)}")
        
        if result.processed_files:
            st.write("**File List:**")
            for file in result.processed_files:
                st.write(f"  • {file}")
        
        if result.errors:
            st.write("**Errors:**")
            for error in result.errors:
                st.error(error)
        
        st.write(f"**Extraction Confidence:** {result.tender_info.extraction_confidence:.1%}")

def main():
    """Main application function"""
    
    # Display header
    display_header()
    
    # Display sidebar and get agent
    agent = display_sidebar()
    
    if agent is None:
        st.error("⚠️ Cannot proceed without proper configuration. Please check your environment variables.")
        st.stop()
    
    # File upload section
    zip_file_path, filename = upload_and_validate_file()
    
    if zip_file_path is None:
        st.markdown("""
        <div class="info-box">
            <h3>📘 How to Use This Tool</h3>
            <ol>
                <li><strong>Prepare Documents:</strong> Collect all tender-related documents (RFP, specifications, terms, etc.)</li>
                <li><strong>Create ZIP File:</strong> Add all documents to a ZIP file</li>
                <li><strong>Upload:</strong> Use the file uploader above to submit your ZIP file</li>
                <li><strong>Review Results:</strong> Get detailed analysis and bidding recommendations</li>
            </ol>
            
            <h4>🔍 What the AI Agent Analyzes:</h4>
            <ul>
                <li>Company/agency information and contact details</li>
                <li>Tender publish and response dates</li>
                <li>Financial requirements and EPU levels</li>
                <li>IT certification requirements</li>
                <li>Technical solution specifications</li>
                <li>Technology stack preferences</li>
                <li>Cloud platform requirements</li>
                <li>Project scope and deliverables</li>
                <li>Service level agreements</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Analysis section
    st.markdown('<h2 class="section-header">🔄 Processing Tender Documents</h2>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing tender documents... This may take a few minutes."):
        try:
            # Perform analysis
            result = agent.analyze_tender(zip_file_path)
            
            # Display results
            display_analysis_results(result, filename)
            
            # Download results option
            st.markdown('<h3 class="section-header">💾 Export Results</h3>', unsafe_allow_html=True)
            
            # Prepare JSON export
            export_data = {
                "filename": filename,
                "analysis_date": datetime.now().isoformat(),
                "recommendation": {
                    "should_bid": result.recommendation.should_bid,
                    "confidence_score": result.recommendation.confidence_score,
                    "reasoning": result.recommendation.reasoning,
                    "criteria_met": {
                        "similarity_threshold": result.recommendation.similarity_score >= 0.7,
                        "value_threshold": result.recommendation.meets_value_threshold,
                        "timeline_threshold": result.recommendation.meets_timeline_threshold,
                        "cloud_compatibility": result.recommendation.cloud_compatibility,
                        "project_type_match": result.recommendation.project_type_match
                    }
                },
                "tender_info": {
                    "company_name": result.tender_info.company_name,
                    "max_tender_value": result.tender_info.max_tender_value,
                    "response_date": result.tender_info.response_date.isoformat() if result.tender_info.response_date else None,
                    "technology_stack": result.tender_info.technology_stack,
                    "solution_summary": result.tender_info.solution_summary
                },
                "missing_information": result.recommendation.missing_information,
                "sales_recommendations": result.recommendation.sales_recommendations
            }
            
            json_data = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="📄 Download Analysis Report (JSON)",
                data=json_data,
                file_name=f"tender_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            logger.error(f"Analysis error: {str(e)}", exc_info=True)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(zip_file_path)
            except:
                pass

if __name__ == "__main__":
    main()
