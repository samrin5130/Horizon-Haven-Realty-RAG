# Horizon Haven Realty Knowledge Assistant

A Retrieval-Augmented Generation (RAG) powered knowledge base assistant for Horizon Haven Realty built with Streamlit, LangChain, and Chroma vector database.

## Features

- Interactive chat interface to query the Horizon Haven Realty knowledge base
- Open-source LLM integration with Qwen3:4B via Ollama
- Vector embeddings with Hugging Face's sentence-transformers
- Download option for the knowledge base
- Persistent vector storage

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/horizon-haven-assistant.git
cd horizon-haven-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the knowledge base in place

4. Install and run Ollama:
   - Download from [ollama.ai](https://ollama.ai)
   - Run `ollama pull qwen3:4b` to download the model

## Usage

### Run the application locally:

```bash
streamlit run app.py
```

This will start the Streamlit app on your local machine, typically at http://localhost:8501.

## Architecture

This system uses a RAG (Retrieval-Augmented Generation) architecture:

1. **Embedding**: Documents are split into chunks and embedded using sentence-transformers
2. **Storage**: Embeddings are stored in a Chroma vector database
3. **Retrieval**: When a query is received, the system finds relevant documents
4. **Generation**: The LLM (Qwen3:4B) uses retrieved documents to generate accurate responses

## Project Structure

```
horizon-haven-assistant/
├── app.py                              # Main Streamlit application
├── utils.py                            # Utility functions
├── requirements.txt                    # Dependencies
├── Horizon_Haven_Realty_KnowledgeBase/ # Knowledge base directory
│   ├── Company/
│   ├── Contracts/
│   ├── Employees/
│   └── ...
└── README.md
```
