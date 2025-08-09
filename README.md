# Tender Qualification AI Agent

This AI Agent system qualifies tender documents submitted in ZIP format and provides bidding recommendations based on document analysis and historical data comparison.

## Features

- Extract and analyze tender documents from ZIP files
- Extract key information (company, dates, requirements, etc.)
- Search historical tender responses for similarities using RAG
- Provide intelligent bidding recommendations
- Clean project structure with modular components

## Project Structure

```
MVP3/
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   └── tender_qualification_agent.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   ├── information_extractor.py
│   │   └── similarity_matcher.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── tender_models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pinecone_service.py
│   │   └── llm_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── file_handler.py
│   │   ├── date_utils.py
│   │   └── config.py
│   └── __init__.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── __init__.py
│   └── test_tender_agent.py
├── data/
│   ├── sample_tenders/
│   └── uploads/
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file
3. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Usage

1. Upload ZIP file containing tender documents
2. The agent will extract and analyze the documents
3. Review the extracted information and bidding recommendation
4. Get detailed reasoning for the recommendation

## Configuration

- Minimum tender value: SGD 500,000
- Minimum days to submission: 10 working days
- Similarity threshold: 70%
- Supported cloud platforms: AWS, Azure
