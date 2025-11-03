# Local LLM using vLLM with AI Agent

This repository powered by **vLLM, FastAPI, Qwen3-4B-Instruct model and Google Search**. 

The agent autonomously decides when to use web search tools to answer questions, providing transparent reasoning and step-by-step thought processes.

## 🎯 What This Repo Is About

This project demonstrates:
- **Local LLM Inference**: Uses vLLM for fast, efficient model inference (Qwen3-4B-Instruct)
- **Autonomous Tool Use**: Agent decides when to search the web based on the question
- **Web Search Integration**: Uses Google Custom Search API with web scraping capabilities
- **REST API**: FastAPI server for easy integration and testing
- **Transparent Reasoning**: View the agent's complete thought process and tool calls


## 🚀 Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (for vLLM)
- Google Custom Search API credentials

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd local_llm_with_ai_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API credentials in `local_ai_agent.py`:
```python
api_key = "<API_KEY>"  # Your Google API key
cx = "<CUSTOM_SEARCH_ENGINE_ID>"  # Your Custom Search Engine ID
```

4. The vLLM model will be downloaded automatically on first run.

## 🏃 How to Start the Server

### Option 1: Endpoints using Uvicorn (Recommended)
```bash
uvicorn local_ai_agent:app --reload --host 0.0.0.0 --port 8000
```
The server will start on `http://localhost:8000`


### Option 2: Run directly
```bash
python local_ai_agent.py
```


## 📁 Repository Structure

- **`local_ai_agent.py`** - FastAPI server implementing the ReAct agent with web search
- **`llm.py`** - Standalone vLLM test script for basic model inference
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This file


The agent **only calls tools when necessary**, making it efficient and transparent.


## 📡 API Endpoints


### Generate Endpoint: `POST /generate`

The main endpoint for generating AI responses using the ReAct framework. The agent **autonomously decides** whether to use the WebSearch tool.

#### JSON Payload

```json
{
  "prompt": "What are the latest AI developments in 2025?"
}
```



## 💻 cURL Command Examples

### Basic Question (May or May Not Use Search)

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"What is quantum computing?\"}"
```

### Question Will Use Web Search

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"What are the latest developments in AI for 2025?\""
```



