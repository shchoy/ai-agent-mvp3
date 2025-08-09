# Quick Start Guide - Tender Qualification AI Agent

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
cd MVP3
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the MVP3 directory:
```env
# Required
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Optional customization
MAX_TENDER_VALUE=500000
MIN_DAYS_TO_SUBMISSION=10
SIMILARITY_THRESHOLD=0.7
```

### 3. Set Up Pinecone Index
Create a Pinecone index named `ai-agent-tender-response` with:
- **Dimension**: 1536 (for OpenAI embeddings)
- **Metric**: Cosine similarity

### 4. Add Sample Historical Data (Optional)
```bash
python setup_sample_data.py
```

### 5. Run the Application
```bash
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

## 📋 How to Use

1. **Prepare Documents**: Collect all tender-related documents (RFP, specifications, terms, etc.)
2. **Create ZIP File**: Add all documents to a ZIP file
3. **Upload**: Use the web interface to upload your ZIP file
4. **Review Results**: Get detailed analysis and bidding recommendations

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

## 🔧 Troubleshooting

### Common Issues:

**"Configuration Error"**
- Ensure all required environment variables are set in `.env`
- Check that your API keys are valid

**"Database Connection Failed"** 
- Verify your Pinecone index exists and is accessible
- Check PINECONE_API_KEY and PINECONE_ENVIRONMENT

**"No text could be extracted"**
- Ensure your ZIP file contains supported document types (PDF, DOCX, TXT, XLSX)
- Check that documents are not password-protected or corrupted

**"Import errors"**
- Run `pip install -r requirements.txt` to install all dependencies
- Ensure you're using Python 3.8+

## 📚 Project Structure

```
MVP3/
├── src/                    # Core application code
│   ├── agents/            # Main AI agent
│   ├── core/              # Processing components  
│   ├── models/            # Data models
│   ├── services/          # External services (LLM, Pinecone)
│   └── utils/             # Utilities
├── app/                   # Streamlit web application
├── tests/                 # Unit tests
├── data/                  # Data directories
└── requirements.txt       # Dependencies
```

## 🎯 Key Features

- **Intelligent Document Processing**: Extracts key information from tender documents
- **Historical Similarity Search**: Finds similar past projects using RAG
- **Business Rules Engine**: Applies configurable business criteria
- **Detailed Recommendations**: Provides comprehensive bidding analysis
- **Web Interface**: Easy-to-use Streamlit application
- **Extensible Architecture**: Modular design for easy customization

## 📊 Business Criteria

The agent evaluates tenders against these criteria:

- **Similarity Threshold**: ≥70% similarity with historical projects
- **Minimum Value**: ≥SGD 500,000 
- **Timeline**: ≥10 working days for response preparation
- **Cloud Compatibility**: AWS or Azure deployment
- **Project Types**: Cloud migration, Data platform, Big data, Serverless, AI Agents

## 💡 Customization

You can customize the agent by:

1. **Modifying Business Rules**: Update `Config` class in `src/utils/config.py`
2. **Adding New Document Types**: Extend `FileHandler` in `src/utils/file_handler.py`
3. **Improving Extraction**: Enhance prompts in `LLMService`
4. **Custom UI**: Modify the Streamlit app in `app/streamlit_app.py`

## 🤝 Contributing

To add new features or improve the agent:

1. Create feature branches from main
2. Add comprehensive tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Submit pull requests for review

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test files for usage examples
3. Examine the sample data in `setup_sample_data.py`
4. Check the detailed code documentation
