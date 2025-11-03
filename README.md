# ReAct AI Agent with Web Search - FastAPI

This FastAPI application implements a **ReAct (Reasoning + Acting)** framework AI agent powered by vLLM. The agent autonomously decides when to use the Web Search tool based on the question.

## What is ReAct?

ReAct is a prompting framework that combines reasoning and acting in language models. The agent:
1. **Thinks** about what to do (Reasoning)
2. **Decides** which action/tool to use (Acting)
3. **Observes** the result
4. **Repeats** until it has enough information to answer

The agent will **only call the WebSearch tool if it determines it's necessary** for answering the question.

## Features

- **ReAct Framework** - Agent autonomously decides when to use tools
- **vLLM Integration** - Fast inference with openai/gpt-oss-20b model
- **Web Search Tool** - DuckDuckGo search (no API key required)
- **Reasoning Transparency** - See the agent's thought process step-by-step
- **FastAPI REST API** - Production-ready with async support
- **Interactive API docs** - Test at `/docs`

## Setup

1. Install dependencies (already in requirements.txt):
```bash
pip install fastapi uvicorn httpx vllm pydantic
```

2. Make sure you have the vLLM model downloaded or accessible.

## Running the Server

Start the FastAPI server:

```bash
uvicorn local_ai_agent:app --reload --host 0.0.0.0 --port 8000
```

Or run directly:
```bash
python local_ai_agent.py
```


**Response:**
```json
{
  "message": "ReAct AI Agent with Web Search Tool is running",
  "model": "openai/gpt-oss-20b",
  "framework": "ReAct (Reasoning + Acting)",
  "tools": ["WebSearch"],
  "endpoints": {
    "/generate": "POST - Generate AI responses using ReAct framework",
    "/docs": "GET - Interactive API documentation"
  }
}
```

### `POST /generate`

Generate AI response using ReAct framework - **Agent autonomously decides whether to use WebSearch tool**

**Request Body:**
```json
{
  "prompt": "What are the latest AI developments in 2025?",
  "max_iterations": 5,
  "temperature": 0.7,
  "top_p": 0.9
}
```

**Parameters:**

- `prompt` (required): The user's question
- `max_iterations` (optional, default: 5): Maximum reasoning steps
- `temperature` (optional, default: 0.7): Sampling temperature
- `top_p` (optional, default: 0.9): Top-p sampling parameter

**Response:**
```json
{
  "prompt": "What are the latest AI developments in 2025?",
  "final_answer": "Based on recent search results...",
  "reasoning_steps": [
    {
      "thought": "I need current information about AI in 2025",
      "action": "WebSearch",
      "action_input": "latest AI developments 2025",
      "observation": "Search Results: ..."
    },
    {
      "thought": "I now have enough information to answer",
      "action": "Final Answer",
      "action_input": "Based on recent developments...",
      "observation": "Task completed"
    }
  ],
  "tool_calls": 1
}
```

## Usage Examples

### Using curl

```bash
# Agent decides if search is needed
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is quantum computing?",
    "max_iterations": 5
  }'

# Simple math - agent likely won't use search
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 15 * 23?",
    "max_iterations": 3
  }'
```

### Using Python (httpx)

```python
import httpx
import asyncio

async def query_agent():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/generate",
            json={
                "prompt": "Who won the latest Nobel Prize in Physics?",
                "max_iterations": 5
            }
        )
        result = response.json()
        
        print(f"Question: {result['prompt']}")
        print(f"Tool Calls: {result['tool_calls']}")
        print(f"Answer: {result['final_answer']}")

asyncio.run(query_agent())
```

### Using the test script

```bash
python test_agent.py
```


## How it Works

The **ReAct Framework** follows this pattern:

```
Question → Thought → Action → Observation → Thought → ... → Final Answer
```

### Example Flow

**Question:** "What are the latest AI developments in 2025?"

1. **Thought:** "I need current information about AI developments in 2025. This requires recent data."
2. **Action:** WebSearch
3. **Action Input:** "latest AI developments 2025"
4. **Observation:** [Search results with recent news]
5. **Thought:** "Now I have enough current information to provide an accurate answer."
6. **Action:** Final Answer
7. **Action Input:** "Based on recent developments, AI in 2025 has seen..."

### When the Agent Uses WebSearch

The agent **autonomously decides** to use the WebSearch tool when:
- ✅ Questions require **current/recent information** (news, events, dates)
- ✅ Questions about **specific facts** it may not know
- ✅ Questions about **real-world current events**

The agent **does NOT use** WebSearch when:
- ❌ Simple **mathematical calculations**
- ❌ **Creative tasks** (stories, poems)
- ❌ **General knowledge** it already possesses
- ❌ **Logical reasoning** that doesn't require external data

## ReAct Prompt Format

The agent uses this structured format internally:

```
Question: [User's question]
Thought: [Agent's reasoning about what to do]
Action: [WebSearch or Final Answer]
Action Input: [Search query or final answer text]
Observation: [Result from the action]
... (repeats until Final Answer)
```


## Interactive Documentation

Visit `http://localhost:8000/docs` for the interactive Swagger UI where you can test the API directly from your browser.

## Notes

- **Autonomous Decision Making**: The agent decides when to use tools based on the question
- **ReAct Framework**: Provides transparent reasoning - you can see the thought process
- **No API Key Required**: Uses DuckDuckGo HTML API for web search
- **Production Ready**: For production, consider using Google Custom Search API or Bing Search API
- **Model Flexibility**: Easy to swap different vLLM-compatible models

## Advantages of ReAct Framework

1. **Transparency**: See exactly how the agent reasons
2. **Efficiency**: Only uses tools when necessary (saves API calls and time)
3. **Flexibility**: Easy to add more tools (Calculator, Database queries, etc.)
4. **Control**: Set max iterations to prevent infinite loops
5. **Debuggable**: Track each reasoning step for troubleshooting

## Customization

### Change the Model

```python
model_path = "your/model/path"  # e.g., "meta-llama/Llama-2-13b-hf"
```

### Add More Tools

```python
async def calculator_tool(expression: str) -> str:
    """Performs mathematical calculations"""
    try:
        result = eval(expression)  # Use safely in production!
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

TOOLS = {
    "WebSearch": {...},
    "Calculator": {
        "function": calculator_tool,
        "description": "Performs mathematical calculations"
    }
}
```

### Adjust Sampling Parameters

```python
sampling_params = SamplingParams(
    temperature=0.7,  # Lower = more focused, Higher = more creative
    top_p=0.9,        # Nucleus sampling
    max_tokens=512,   # Max tokens per generation
    stop=["Observation:"]  # Stop sequence for ReAct
)
```

## Troubleshooting

**Agent not using WebSearch when it should:**
- The model may need better prompting
- Try a larger/better model
- Adjust temperature (lower for more focused reasoning)

**Agent stuck in loops:**
- Reduce `max_iterations`
- Check if the stop sequence is working properly
- Review the model's reasoning steps

**Search not working:**
- DuckDuckGo HTML API may be rate-limited
- Consider switching to a proper API (Google, Bing, SerpAPI)
- Check network connectivity
